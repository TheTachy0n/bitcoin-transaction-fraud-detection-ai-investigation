# ============================================================
# ALIGN XGBOOST + GRAPHSAGE PREDICTIONS
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
    / "xgboost_validation_predictions.csv"
)

GNN_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphsage_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "aligned_validation_predictions.csv"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 60)
print("ALIGNING XGBOOST + GRAPHSAGE PREDICTIONS")
print("=" * 60)

print("\nLoading XGBoost predictions...")

xgb = pd.read_csv(XGB_PATH)

print(
    "XGBoost rows:",
    len(xgb)
)

print("\nLoading GraphSAGE predictions...")

gnn = pd.read_csv(GNN_PATH)

print(
    "GraphSAGE rows:",
    len(gnn)
)


# ============================================================
# KEEP ONLY LABELLED GRAPHSAGE TRANSACTIONS
# ============================================================

gnn_labelled = gnn[
    gnn["label"] != -1
].copy()

print(
    "\nGraphSAGE labelled rows:",
    len(gnn_labelled)
)

print(
    "GraphSAGE unknown rows:",
    (gnn["label"] == -1).sum()
)


# ============================================================
# CHECK DUPLICATE TRANSACTION IDs
# ============================================================

print("\nDuplicate TX IDs:")

print(
    "XGBoost:",
    xgb["txId"].duplicated().sum()
)

print(
    "GraphSAGE:",
    gnn_labelled["txId"].duplicated().sum()
)


# ============================================================
# ALIGN USING TX ID
# ============================================================

aligned = pd.merge(
    xgb[
        [
            "txId",
            "timestep",
            "label",
            "xgboost_probability"
        ]
    ],
    gnn_labelled[
        [
            "txId",
            "graphsage_probability"
        ]
    ],
    on="txId",
    how="inner"
)


# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

print(
    "\nAligned rows:",
    len(aligned)
)

print(
    "XGBoost rows:",
    len(xgb)
)

print(
    "Rows lost:",
    len(xgb) - len(aligned)
)


# ============================================================
# SAVE
# ============================================================

aligned.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    "\nSaved:"
)

print(OUTPUT_PATH)


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print(
    "\nAligned prediction sample:"
)

print(
    aligned.head(10).to_string(
        index=False
    )
)


# ============================================================
# PROBABILITY SUMMARY
# ============================================================

print(
    "\nProbability summary:"
)

print(
    aligned[
        [
            "xgboost_probability",
            "graphsage_probability"
        ]
    ].describe()
)

print(
    "\nAlignment complete."
)