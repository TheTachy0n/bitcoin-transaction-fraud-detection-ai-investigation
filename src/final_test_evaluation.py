# ============================================================
# STEP 7 — FINAL TEST EVALUATION
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

XGB_PATH = (
    RESULTS_DIR /
    "xgboost_test_predictions.csv"
)

GNN_PATH = (
    RESULTS_DIR /
    "graphsage_test_predictions.csv"
)

# If your actual filenames differ, change them above.


# ============================================================
# FINAL CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

HYBRID_THRESHOLD = 0.79

TOP_K_VALUES = [100, 500, 1000]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 7 — FINAL TEST EVALUATION")
print("=" * 70)

print("\nFINAL LOCKED CONFIGURATION")
print("-" * 70)

print(f"XGBoost weight   : {XGB_WEIGHT:.2f}")
print(f"GraphSAGE weight : {GNN_WEIGHT:.2f}")
print(f"Hybrid threshold : {HYBRID_THRESHOLD:.2f}")


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading XGBoost predictions...")

xgb = pd.read_csv(XGB_PATH)

print("Rows:", len(xgb))
print("Columns:", list(xgb.columns))


print("\nLoading GraphSAGE predictions...")

gnn = pd.read_csv(GNN_PATH)

print("Rows:", len(gnn))
print("Columns:", list(gnn.columns))


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

# XGBoost
if "fraud_probability" in xgb.columns:
    xgb_prob_col = "fraud_probability"
elif "xgboost_probability" in xgb.columns:
    xgb_prob_col = "xgboost_probability"
else:
    raise ValueError(
        "Could not find XGBoost probability column."
    )


# GraphSAGE
if "probability" in gnn.columns:
    gnn_prob_col = "probability"
elif "graphsage_probability" in gnn.columns:
    gnn_prob_col = "graphsage_probability"
else:
    raise ValueError(
        "Could not find GraphSAGE probability column."
    )


# Ground truth
if "actual_label" in xgb.columns:
    xgb_label_col = "actual_label"

elif "true_label" in xgb.columns:
    xgb_label_col = "true_label"

elif "label" in xgb.columns:
    xgb_label_col = "label"

else:
    xgb_label_col = None


if "true_label" in gnn.columns:
    gnn_label_col = "true_label"

elif "actual_label" in gnn.columns:
    gnn_label_col = "actual_label"

elif "label" in gnn.columns:
    gnn_label_col = "label"

else:
    gnn_label_col = None


print("\nXGBoost probability column:", xgb_prob_col)
print("GraphSAGE probability column:", gnn_prob_col)


# ============================================================
# PREPARE XGBOOST
# ============================================================

xgb_clean = xgb[
    [
        "txId",
        "timestep",
        xgb_prob_col
    ]
].copy()

xgb_clean = xgb_clean.rename(
    columns={
        xgb_prob_col:
        "xgboost_probability"
    }
)


# ============================================================
# PREPARE GRAPHSAGE
# ============================================================

gnn_clean = gnn[
    [
        "txId",
        "timestep",
        gnn_prob_col
    ]
].copy()

gnn_clean = gnn_clean.rename(
    columns={
        gnn_prob_col:
        "graphsage_probability"
    }
)


# ============================================================
# ALIGN MODELS
# ============================================================

print("\nAligning model predictions...")

df = pd.merge(
    xgb_clean,
    gnn_clean,
    on=[
        "txId",
        "timestep"
    ],
    how="inner"
)


print(
    "Aligned transactions:",
    len(df)
)


# ============================================================
# GET GROUND TRUTH
# ============================================================

if xgb_label_col is not None:

    labels = xgb[
        [
            "txId",
            "timestep",
            xgb_label_col
        ]
    ].copy()

    labels = labels.rename(
        columns={
            xgb_label_col:
            "label"
        }
    )

elif gnn_label_col is not None:

    labels = gnn[
        [
            "txId",
            "timestep",
            gnn_label_col
        ]
    ].copy()

    labels = labels.rename(
        columns={
            gnn_label_col:
            "label"
        }
    )

else:

    raise ValueError(
        "Could not find ground-truth labels "
        "in either prediction file."
    )


# ============================================================
# ALIGN LABELS
# ============================================================

df = pd.merge(
    df,
    labels,
    on=[
        "txId",
        "timestep"
    ],
    how="inner"
)


# Remove duplicate rows if any
df = df.drop_duplicates(
    subset=[
        "txId",
        "timestep"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CLEAN LABELS
# ============================================================

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[
    df["label"].isin([0, 1])
].copy()

df["label"] = df[
    "label"
].astype(int)


# ============================================================
# CLEAN PROBABILITIES
# ============================================================

df["xgboost_probability"] = (
    pd.to_numeric(
        df["xgboost_probability"],
        errors="coerce"
    )
    .clip(0.0, 1.0)
)

df["graphsage_probability"] = (
    pd.to_numeric(
        df["graphsage_probability"],
        errors="coerce"
    )
    .clip(0.0, 1.0)
)


df = df.dropna(
    subset=[
        "xgboost_probability",
        "graphsage_probability",
        "label"
    ]
).reset_index(
    drop=True
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\nGround-truth distribution:")

print(
    df["label"].value_counts()
)

total_fraud = int(
    (df["label"] == 1).sum()
)

total_legitimate = int(
    (df["label"] == 0).sum()
)

print(
    "\nFraud cases:",
    total_fraud
)

print(
    "Legitimate cases:",
    total_legitimate
)


# ============================================================
# CREATE HYBRID SCORE
# ============================================================

df["hybrid_probability"] = (
    XGB_WEIGHT
    * df["xgboost_probability"]
    +
    GNN_WEIGHT
    * df["graphsage_probability"]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    probabilities,
    threshold=0.5
):

    labels = df["label"].values

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {

        "model": name,

        "pr_auc":
            average_precision_score(
                labels,
                probabilities
            ),

        "roc_auc":
            roc_auc_score(
                labels,
                probabilities
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"PR-AUC     : {metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC    : {metrics['roc_auc']:.4f}"
    )

    print(
        f"Precision  : {metrics['precision']:.4f}"
    )

    print(
        f"Recall     : {metrics['recall']:.4f}"
    )

    print(
        f"F1         : {metrics['f1']:.4f}"
    )

    print(
        f"Accuracy   : {metrics['accuracy']:.4f}"
    )

    print("\nConfusion Matrix")

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"TP: {tp}"
    )

    return metrics


# ============================================================
# FINAL MODEL EVALUATION
# ============================================================

xgb_metrics = evaluate_model(
    "XGBoost",
    df["xgboost_probability"].values,
    threshold=0.82
)


gnn_metrics = evaluate_model(
    "GraphSAGE",
    df["graphsage_probability"].values,
    threshold=0.84
)


hybrid_metrics = evaluate_model(
    "90/10 Hybrid",
    df["hybrid_probability"].values,
    threshold=HYBRID_THRESHOLD
)


# ============================================================
# TOP-K EVALUATION
# ============================================================

def top_k_metrics(
    name,
    probabilities
):

    results = []

    order = np.argsort(
        probabilities
    )[::-1]

    labels = df["label"].values

    for k in TOP_K_VALUES:

        k = min(
            k,
            len(labels)
        )

        selected = order[:k]

        fraud_found = int(
            labels[selected].sum()
        )

        precision = (
            fraud_found / k
        )

        recall = (
            fraud_found / total_fraud
        )

        results.append({

            "model": name,

            "k": k,

            "fraud_found":
                fraud_found,

            "total_fraud":
                total_fraud,

            "precision_at_k":
                precision,

            "recall_at_k":
                recall
        })

    return results


top_k_results = []

top_k_results.extend(
    top_k_metrics(
        "XGBoost",
        df["xgboost_probability"].values
    )
)

top_k_results.extend(
    top_k_metrics(
        "GraphSAGE",
        df["graphsage_probability"].values
    )
)

top_k_results.extend(
    top_k_metrics(
        "90/10 Hybrid",
        df["hybrid_probability"].values
    )
)


top_k_df = pd.DataFrame(
    top_k_results
)


# ============================================================
# PRINT TOP-K
# ============================================================

print("\n" + "=" * 70)
print("FINAL TOP-K TEST RESULTS")
print("=" * 70)

for model_name in [
    "XGBoost",
    "GraphSAGE",
    "90/10 Hybrid"
]:

    print(
        f"\n{model_name}"
    )

    subset = top_k_df[
        top_k_df["model"]
        == model_name
    ]

    for _, row in subset.iterrows():

        print(
            f"Top-{int(row['k']):4d} | "
            f"Fraud found: "
            f"{int(row['fraud_found']):4d} | "
            f"Precision: "
            f"{row['precision_at_k']:.4f} | "
            f"Recall: "
            f"{row['recall_at_k']:.4f}"
        )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        xgb_metrics,
        gnn_metrics,
        hybrid_metrics
    ]
)


metrics_path = (
    RESULTS_DIR /
    "final_test_metrics.csv"
)

top_k_path = (
    RESULTS_DIR /
    "final_test_top_k.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)

top_k_df.to_csv(
    top_k_path,
    index=False
)


# ============================================================
# SAVE FINAL SCORES
# ============================================================

output_predictions = df[
    [
        "txId",
        "timestep",
        "label",
        "xgboost_probability",
        "graphsage_probability",
        "hybrid_probability"
    ]
].copy()


output_predictions = output_predictions.sort_values(
    "hybrid_probability",
    ascending=False
)


prediction_path = (
    RESULTS_DIR /
    "final_test_predictions.csv"
)


output_predictions.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)

print("\nSaved:")

print(metrics_path)
print(top_k_path)
print(prediction_path)