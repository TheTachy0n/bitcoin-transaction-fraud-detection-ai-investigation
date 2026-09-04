# ============================================================
# STEP 9 — FINAL RISK ENGINE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "final_risk_engine.csv"
)


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

HIGH_THRESHOLD = 0.79
MEDIUM_THRESHOLD = 0.50

# Threshold used ONLY for determining
# model agreement categories.
AGREEMENT_THRESHOLD = 0.50


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 9 — FINAL RISK ENGINE")
print("=" * 70)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("\nLoading final model predictions...")

df = pd.read_csv(
    INPUT_PATH
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "txId",
    "timestep",
    "label",
    "xgboost_probability",
    "graphsage_probability"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CALCULATE FINAL RISK SCORE
# ============================================================

print("\nCalculating final risk score...")

df["risk_score"] = (
    XGB_WEIGHT
    * df["xgboost_probability"]
    +
    GNN_WEIGHT
    * df["graphsage_probability"]
)


# ============================================================
# VERIFY SCORE RANGE
# ============================================================

if (
    df["risk_score"].min() < 0
    or
    df["risk_score"].max() > 1
):

    raise ValueError(
        "Risk score outside [0,1]"
    )


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(score):

    if score >= HIGH_THRESHOLD:

        return "HIGH"

    elif score >= MEDIUM_THRESHOLD:

        return "MEDIUM"

    else:

        return "LOW"


df["risk_level"] = (
    df["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# MODEL AGREEMENT
# ============================================================

df["xgb_high"] = (
    df["xgboost_probability"]
    >= AGREEMENT_THRESHOLD
)

df["gnn_high"] = (
    df["graphsage_probability"]
    >= AGREEMENT_THRESHOLD
)


# ============================================================
# AGREEMENT CATEGORY
# ============================================================

def determine_agreement(row):

    xgb_high = row["xgb_high"]
    gnn_high = row["gnn_high"]

    if xgb_high and gnn_high:

        return "BOTH_HIGH"

    elif not xgb_high and not gnn_high:

        return "BOTH_LOW"

    elif xgb_high and not gnn_high:

        return "XGB_HIGH_GNN_LOW"

    else:

        return "XGB_LOW_GNN_HIGH"


df["agreement_category"] = (
    df.apply(
        determine_agreement,
        axis=1
    )
)


# ============================================================
# MODEL AGREEMENT SCORE
# ============================================================

df["model_agreement"] = 1 - (
    np.abs(
        df["xgboost_probability"]
        -
        df["graphsage_probability"]
    )
)


df["model_disagreement"] = 1 - (
    df["model_agreement"]
)


# ============================================================
# EVIDENCE TYPE
# ============================================================

def determine_evidence(row):

    category = row[
        "agreement_category"
    ]

    if category == "BOTH_HIGH":

        return "STRONG_CORROBORATION"

    elif category == "XGB_HIGH_GNN_LOW":

        return "STRONG_TRANSACTION_SIGNAL"

    elif category == "XGB_LOW_GNN_HIGH":

        return "GRAPH_SIGNAL_ONLY"

    else:

        return "LOW_SIGNAL"


df["evidence_type"] = (
    df.apply(
        determine_evidence,
        axis=1
    )
)


# ============================================================
# ALERT PRIORITY
# ============================================================

def determine_priority(row):

    risk = row["risk_score"]

    category = row[
        "agreement_category"
    ]

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if (
        risk >= 0.90
        and
        category == "BOTH_HIGH"
    ):

        return "CRITICAL"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    elif risk >= HIGH_THRESHOLD:

        return "HIGH"

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    elif risk >= MEDIUM_THRESHOLD:

        return "MEDIUM"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    else:

        return "LOW"


df["alert_priority"] = (
    df.apply(
        determine_priority,
        axis=1
    )
)


# ============================================================
# FRAUD DECISION
# ============================================================

df["fraud_alert"] = (
    df["risk_score"]
    >= HIGH_THRESHOLD
)


# ============================================================
# SORT BY RISK
# ============================================================

df = df.sort_values(
    "risk_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RISK ENGINE SUMMARY")
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
# FRAUD ALERT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FRAUD ALERT SUMMARY")
print("=" * 70)

print(
    "Total transactions:",
    len(df)
)

print(
    "HIGH risk transactions:",
    (
        df["risk_level"]
        == "HIGH"
    ).sum()
)

print(
    "MEDIUM risk transactions:",
    (
        df["risk_level"]
        == "MEDIUM"
    ).sum()
)

print(
    "LOW risk transactions:",
    (
        df["risk_level"]
        == "LOW"
    ).sum()
)

print(
    "Fraud alerts:",
    df["fraud_alert"].sum()
)


# ============================================================
# TOP 20 TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 HIGHEST-RISK TRANSACTIONS")
print("=" * 70)


display_columns = [
    "txId",
    "timestep",
    "label",
    "xgboost_probability",
    "graphsage_probability",
    "risk_score",
    "risk_level",
    "agreement_category",
    "evidence_type",
    "alert_priority"
]


print(
    df[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# MODEL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL CONTRIBUTION")
print("=" * 70)

print(
    "XGBoost weight:",
    XGB_WEIGHT
)

print(
    "GraphSAGE weight:",
    GNN_WEIGHT
)

print(
    "Average XGBoost probability:",
    f"{df['xgboost_probability'].mean():.6f}"
)

print(
    "Average GraphSAGE probability:",
    f"{df['graphsage_probability'].mean():.6f}"
)

print(
    "Average final risk:",
    f"{df['risk_score'].mean():.6f}"
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "txId",
    "timestep",
    "label",

    "xgboost_probability",
    "graphsage_probability",

    "risk_score",

    "risk_level",
    "alert_priority",

    "agreement_category",

    "model_agreement",
    "model_disagreement",

    "evidence_type",

    "fraud_alert"
]


df[
    output_columns
].to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nSaved:")
print(OUTPUT_PATH)


print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)