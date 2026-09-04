# ============================================================
# STEP 10 — RISK ENGINE
# XGBoost primary + GraphSAGE secondary evidence
# ============================================================

from pathlib import Path

import numpy as np
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "risk_engine_output.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Based on Step 9:
#
# XGBoost is the primary model.
# GraphSAGE is secondary graph evidence.
#
# We deliberately DO NOT use the failed logistic
# meta-classifier fusion here.

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10


# Risk levels

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 10 — RISK ENGINE")
print("=" * 70)


# ============================================================
# LOAD XGBOOST
# ============================================================

print("\nLoading XGBoost predictions...")

xgb = pd.read_csv(XGB_PATH)

print("XGBoost rows:", len(xgb))
print("XGBoost columns:", list(xgb.columns))


# Find XGBoost probability column

xgb_probability_column = None

for column in [
    "fraud_probability",
    "xgboost_probability",
    "probability"
]:

    if column in xgb.columns:

        xgb_probability_column = column
        break


if xgb_probability_column is None:

    raise ValueError(
        "Could not find XGBoost probability column."
    )


print(
    "Using XGBoost probability column:",
    xgb_probability_column
)


# ============================================================
# LOAD GRAPHSAGE
# ============================================================

print("\nLoading GraphSAGE predictions...")

gnn = pd.read_csv(GNN_PATH)

print("GraphSAGE rows:", len(gnn))
print("GraphSAGE columns:", list(gnn.columns))


# Your actual GraphSAGE file uses:
#
# probability
#
# So explicitly support that.

gnn_probability_column = None

for column in [
    "graphsage_probability",
    "probability",
    "fraud_probability"
]:

    if column in gnn.columns:

        gnn_probability_column = column
        break


if gnn_probability_column is None:

    raise ValueError(
        "Could not find GraphSAGE probability column."
    )


print(
    "Using GraphSAGE probability column:",
    gnn_probability_column
)


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

xgb_clean = xgb[
    [
        "txId",
        "timestep",
        xgb_probability_column
    ]
].copy()


gnn_clean = gnn[
    [
        "txId",
        "timestep",
        gnn_probability_column
    ]
].copy()


# Rename columns to standard names

xgb_clean = xgb_clean.rename(
    columns={
        xgb_probability_column:
            "xgboost_probability"
    }
)


gnn_clean = gnn_clean.rename(
    columns={
        gnn_probability_column:
            "graphsage_probability"
    }
)


# ============================================================
# ALIGN MODELS
# ============================================================

print("\nAligning XGBoost and GraphSAGE...")

risk_df = pd.merge(
    xgb_clean,
    gnn_clean[
        [
            "txId",
            "graphsage_probability"
        ]
    ],
    on="txId",
    how="left"
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

print(
    "Total XGBoost transactions:",
    len(xgb_clean)
)

print(
    "Aligned transactions:",
    len(risk_df)
)

missing_gnn = (
    risk_df["graphsage_probability"]
    .isna()
    .sum()
)

print(
    "Missing GraphSAGE probabilities:",
    missing_gnn
)


# ============================================================
# HANDLE MISSING GRAPHSAGE
# ============================================================

# GraphSAGE may not have predictions for every transaction.
#
# Since XGBoost is our primary model, transactions without
# GraphSAGE evidence simply receive zero graph contribution.

risk_df[
    "graphsage_probability"
] = risk_df[
    "graphsage_probability"
].fillna(0.0)


# ============================================================
# SANITY CHECK
# ============================================================

risk_df[
    "xgboost_probability"
] = risk_df[
    "xgboost_probability"
].clip(0.0, 1.0)


risk_df[
    "graphsage_probability"
] = risk_df[
    "graphsage_probability"
].clip(0.0, 1.0)


# ============================================================
# CALCULATE RISK SCORE
# ============================================================

print("\nCalculating risk scores...")

risk_df["risk_score"] = (
    XGB_WEIGHT
    * risk_df["xgboost_probability"]
    +
    GNN_WEIGHT
    * risk_df["graphsage_probability"]
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


risk_df["risk_level"] = (
    risk_df["risk_score"]
    .apply(classify_risk)
)


# ============================================================
# PRIMARY MODEL PREDICTION
# ============================================================

risk_df["xgboost_risk_level"] = (
    risk_df["xgboost_probability"]
    .apply(classify_risk)
)


# ============================================================
# GRAPH EVIDENCE
# ============================================================

risk_df["graph_evidence"] = np.where(
    risk_df["graphsage_probability"] >= 0.80,
    "HIGH",
    np.where(
        risk_df["graphsage_probability"] >= 0.50,
        "MEDIUM",
        "LOW"
    )
)


# ============================================================
# SORT BY RISK
# ============================================================

risk_df = risk_df.sort_values(
    "risk_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

risk_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK ENGINE SUMMARY")
print("=" * 70)

print(
    "\nTransactions:",
    len(risk_df)
)

print(
    "\nRisk level distribution:"
)

print(
    risk_df["risk_level"]
    .value_counts()
)


print(
    "\nRisk score statistics:"
)

print(
    risk_df["risk_score"]
    .describe()
)


# ============================================================
# TOP HIGH-RISK TRANSACTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 20 HIGH-RISK TRANSACTIONS"
)

print(
    "=" * 70
)


display_columns = [
    "txId",
    "timestep",
    "xgboost_probability",
    "graphsage_probability",
    "risk_score",
    "risk_level"
]


print(
    risk_df[
        display_columns
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# MODEL CONTRIBUTION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL CONFIGURATION"
)

print(
    "=" * 70
)

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


# ============================================================
# OUTPUT
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\n" + "=" * 70
)

print(
    "STEP 10 — RISK ENGINE COMPLETE"
)

print(
    "=" * 70
)