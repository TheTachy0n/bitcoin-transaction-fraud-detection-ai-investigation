# ============================================================
# STEP 4 — TOP-K FRAUD DETECTION
# XGBoost vs GraphSAGE vs 90/10 Hybrid
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

XGB_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost_test_predictions.csv"
)

GNN_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphsage_test_predictions.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "top_k_comparison.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

TOP_K_VALUES = [100, 500, 1000]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 4 — TOP-K FRAUD DETECTION")
print("=" * 70)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading XGBoost predictions...")

xgb = pd.read_csv(XGB_PATH)

print("XGBoost rows:", len(xgb))
print("XGBoost columns:", list(xgb.columns))


print("\nLoading GraphSAGE predictions...")

gnn = pd.read_csv(GNN_PATH)

print("GraphSAGE rows:", len(gnn))
print("GraphSAGE columns:", list(gnn.columns))


# ============================================================
# LOAD GROUND-TRUTH TEST DATA
# ============================================================

print("\nLoading test labels...")

test = pd.read_csv(TEST_PATH)

print("Test rows:", len(test))
print("Test columns:", list(test.columns))


# ============================================================
# FIND PROBABILITY COLUMNS
# ============================================================

def find_probability_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not find probability column.\n"
        f"Available columns: {list(df.columns)}"
    )


xgb_probability_column = find_probability_column(
    xgb,
    [
        "xgboost_probability",
        "fraud_probability",
        "probability"
    ]
)


gnn_probability_column = find_probability_column(
    gnn,
    [
        "graphsage_probability",
        "fraud_probability",
        "probability"
    ]
)


print(
    "\nXGBoost probability column:",
    xgb_probability_column
)

print(
    "GraphSAGE probability column:",
    gnn_probability_column
)


# ============================================================
# FIND LABEL COLUMN
# ============================================================

if "label" in test.columns:

    label_column = "label"

elif "class" in test.columns:

    label_column = "class"

else:

    raise ValueError(
        "Could not find label column in test.csv.\n"
        f"Available columns: {list(test.columns)}"
    )


print(
    "Ground-truth label column:",
    label_column
)


# ============================================================
# PREPARE XGBOOST
# ============================================================

xgb_clean = xgb[
    [
        "txId",
        xgb_probability_column
    ]
].copy()


xgb_clean = xgb_clean.rename(
    columns={
        xgb_probability_column:
            "xgb_probability"
    }
)


# ============================================================
# PREPARE GRAPHSAGE
# ============================================================

gnn_clean = gnn[
    [
        "txId",
        gnn_probability_column
    ]
].copy()


gnn_clean = gnn_clean.rename(
    columns={
        gnn_probability_column:
            "gnn_probability"
    }
)


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

test_clean = test[
    [
        "txId",
        label_column
    ]
].copy()


test_clean = test_clean.rename(
    columns={
        label_column:
            "label"
    }
)


# ============================================================
# CONVERT LABELS IF NECESSARY
# ============================================================

# Elliptic:
#
# 1 = illicit / fraud
# 0 = licit / legitimate
#
# If labels are strings such as "1" and "0",
# convert them to integers.

test_clean["label"] = pd.to_numeric(
    test_clean["label"],
    errors="coerce"
)


# ============================================================
# REMOVE UNKNOWN LABELS
# ============================================================

test_clean = test_clean[
    test_clean["label"].isin([0, 1])
].copy()


test_clean["label"] = (
    test_clean["label"]
    .astype(int)
)


# ============================================================
# ALIGN ALL THREE SOURCES
# ============================================================

print("\nAligning predictions with ground truth...")

df = xgb_clean.merge(
    gnn_clean,
    on="txId",
    how="inner"
)

df = df.merge(
    test_clean,
    on="txId",
    how="inner"
)


# ============================================================
# SANITY CHECK
# ============================================================

print(
    "\nAligned transactions:",
    len(df)
)

print(
    "Fraud cases:",
    int(df["label"].sum())
)

print(
    "Legitimate cases:",
    int((df["label"] == 0).sum())
)


if len(df) == 0:

    raise ValueError(
        "No transactions were successfully aligned."
    )


# ============================================================
# CREATE HYBRID SCORE
# ============================================================

df["hybrid_probability"] = (
    XGB_WEIGHT * df["xgb_probability"]
    +
    GNN_WEIGHT * df["gnn_probability"]
)


# ============================================================
# TOP-K EVALUATION
# ============================================================

results = []


for model_name, score_column in [

    ("XGBoost", "xgb_probability"),

    ("GraphSAGE", "gnn_probability"),

    ("90/10 Hybrid", "hybrid_probability")

]:

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)


    ranked = df.sort_values(
        score_column,
        ascending=False
    ).reset_index(drop=True)


    total_fraud = int(
        ranked["label"].sum()
    )


    for k in TOP_K_VALUES:

        # Protect against K > dataset size

        actual_k = min(
            k,
            len(ranked)
        )


        top_k = ranked.head(actual_k)


        fraud_found = int(
            top_k["label"].sum()
        )


        precision_at_k = (
            fraud_found / actual_k
        )


        recall_at_k = (
            fraud_found / total_fraud
        )


        results.append({

            "model": model_name,

            "k": k,

            "fraud_found":
                fraud_found,

            "total_fraud":
                total_fraud,

            "precision_at_k":
                precision_at_k,

            "recall_at_k":
                recall_at_k

        })


        print(
            f"Top-{k:4d} | "
            f"Fraud found: {fraud_found:4d} | "
            f"Precision@{k}: "
            f"{precision_at_k:.4f} | "
            f"Recall@{k}: "
            f"{recall_at_k:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("TOP-K COMPARISON")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT_PATH)


print("\n" + "=" * 70)
print("STEP 4 COMPLETE")
print("=" * 70)