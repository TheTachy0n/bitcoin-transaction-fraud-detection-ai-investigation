# ============================================================
# DAY 3 — RISK ENGINE V2
# Evidence-aware risk representation
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "aligned_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "risk_engine_v2_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Keep the current baseline weights for now.
#
# We will optimize these later.

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DAY 3 — RISK ENGINE V2")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nLoaded transactions:", len(df))


# ============================================================
# SANITY CHECK
# ============================================================

required_columns = [
    "txId",
    "timestep",
    "label",
    "xgboost_probability",
    "graphsage_probability"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CLIP PROBABILITIES
# ============================================================

df["xgboost_probability"] = (
    df["xgboost_probability"]
    .clip(0.0, 1.0)
)

df["graphsage_probability"] = (
    df["graphsage_probability"]
    .clip(0.0, 1.0)
)


# ============================================================
# INDIVIDUAL MODEL RISK
# ============================================================

df["transaction_risk"] = (
    df["xgboost_probability"]
)

df["graph_risk"] = (
    df["graphsage_probability"]
)


# ============================================================
# MODEL DISAGREEMENT
# ============================================================

df["model_disagreement"] = (
    df["transaction_risk"]
    - df["graph_risk"]
).abs()


# ============================================================
# MODEL AGREEMENT
# ============================================================

df["model_agreement"] = (
    1.0
    - df["model_disagreement"]
)


# ============================================================
# WEIGHTED COMBINED RISK
# ============================================================

df["risk_score"] = (
    XGB_WEIGHT
    * df["transaction_risk"]
    +
    GNN_WEIGHT
    * df["graph_risk"]
)


# ============================================================
# MODEL RISK LEVELS
# ============================================================

def classify_risk(score):

    if score >= HIGH_THRESHOLD:
        return "HIGH"

    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


df["transaction_risk_level"] = (
    df["transaction_risk"]
    .apply(classify_risk)
)

df["graph_risk_level"] = (
    df["graph_risk"]
    .apply(classify_risk)
)

df["risk_level"] = (
    df["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# MODEL AGREEMENT CATEGORY
# ============================================================

def classify_agreement(row):

    xgb_high = (
        row["transaction_risk"]
        >= HIGH_THRESHOLD
    )

    gnn_high = (
        row["graph_risk"]
        >= HIGH_THRESHOLD
    )

    if xgb_high and gnn_high:
        return "BOTH_HIGH"

    elif xgb_high and not gnn_high:
        return "XGB_HIGH_GNN_LOW"

    elif not xgb_high and gnn_high:
        return "XGB_LOW_GNN_HIGH"

    return "BOTH_LOW"


df["agreement_category"] = (
    df.apply(
        classify_agreement,
        axis=1
    )
)


# ============================================================
# EVIDENCE STRENGTH
# ============================================================

def classify_evidence(row):

    category = row["agreement_category"]

    if category == "BOTH_HIGH":
        return "STRONG_CORROBORATION"

    elif category == "XGB_HIGH_GNN_LOW":
        return "STRONG_TRANSACTION_SIGNAL"

    elif category == "XGB_LOW_GNN_HIGH":
        return "GRAPH_SIGNAL_ONLY"

    return "LOW_SIGNAL"


df["evidence_type"] = (
    df.apply(
        classify_evidence,
        axis=1
    )
)


# ============================================================
# ALERT PRIORITY
# ============================================================

def classify_priority(row):

    xgb = row["transaction_risk"]
    gnn = row["graph_risk"]

    # Both models strongly support fraud
    if (
        xgb >= 0.95
        and gnn >= 0.80
    ):
        return "CRITICAL"

    # Very strong transaction signal
    if (
        xgb >= 0.95
        and gnn < 0.80
    ):
        return "HIGH"

    # Strong graph signal without strong XGB support
    if (
        gnn >= 0.80
        and xgb < 0.50
    ):
        return "MEDIUM"

    # Combined risk is elevated
    if row["risk_score"] >= 0.80:
        return "HIGH"

    if row["risk_score"] >= 0.50:
        return "MEDIUM"

    return "LOW"


df["alert_priority"] = (
    df.apply(
        classify_priority,
        axis=1
    )
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "alert_priority",
        "risk_score"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK ENGINE V2 SUMMARY")
print("=" * 70)

print("\nRisk level:")
print(
    df["risk_level"]
    .value_counts()
)

print("\nAlert priority:")
print(
    df["alert_priority"]
    .value_counts()
)

print("\nEvidence type:")
print(
    df["evidence_type"]
    .value_counts()
)

print("\nAgreement category:")
print(
    df["agreement_category"]
    .value_counts()
)


# ============================================================
# TOP TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 TRANSACTIONS")
print("=" * 70)

display_columns = [
    "txId",
    "timestep",
    "label",
    "transaction_risk",
    "graph_risk",
    "model_agreement",
    "model_disagreement",
    "risk_score",
    "agreement_category",
    "evidence_type",
    "alert_priority"
]

print(
    df[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)

print(
    f"XGBoost weight   : {XGB_WEIGHT:.2f}"
)

print(
    f"GraphSAGE weight : {GNN_WEIGHT:.2f}"
)

print(
    f"High threshold   : {HIGH_THRESHOLD:.2f}"
)

print(
    f"Medium threshold : {MEDIUM_THRESHOLD:.2f}"
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("RISK ENGINE V2 COMPLETE")
print("=" * 70)