# ============================================================
# DAY 3 — MODEL AGREEMENT ANALYSIS
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
    / "model_agreement_validation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DAY 3 — MODEL AGREEMENT ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nLoaded:", len(df), "transactions")
print("Columns:", list(df.columns))


# ============================================================
# MODEL AGREEMENT
# ============================================================

df["disagreement"] = (
    df["xgboost_probability"]
    - df["graphsage_probability"]
).abs()

df["agreement"] = 1.0 - df["disagreement"]


# ============================================================
# HIGH / LOW MODEL SIGNAL
# ============================================================

THRESHOLD = 0.50

df["xgb_high"] = (
    df["xgboost_probability"] >= THRESHOLD
)

df["gnn_high"] = (
    df["graphsage_probability"] >= THRESHOLD
)


# ============================================================
# AGREEMENT GROUP
# ============================================================

def assign_group(row):

    if row["xgb_high"] and row["gnn_high"]:
        return "XGB HIGH + GNN HIGH"

    elif row["xgb_high"] and not row["gnn_high"]:
        return "XGB HIGH + GNN LOW"

    elif not row["xgb_high"] and row["gnn_high"]:
        return "XGB LOW + GNN HIGH"

    else:
        return "XGB LOW + GNN LOW"


df["agreement_group"] = df.apply(
    assign_group,
    axis=1
)


# ============================================================
# GROUP ANALYSIS
# ============================================================

summary = (
    df
    .groupby("agreement_group")
    .agg(
        transactions=("label", "count"),
        fraud_cases=("label", "sum"),
        fraud_rate=("label", "mean"),
        avg_xgb_probability=(
            "xgboost_probability",
            "mean"
        ),
        avg_gnn_probability=(
            "graphsage_probability",
            "mean"
        ),
        avg_disagreement=(
            "disagreement",
            "mean"
        ),
        avg_agreement=(
            "agreement",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL AGREEMENT GROUPS")
print("=" * 70)

print(
    summary.to_string(index=False)
)


# ============================================================
# OVERALL CORRELATION
# ============================================================

correlation = df[
    [
        "xgboost_probability",
        "graphsage_probability"
    ]
].corr().iloc[0, 1]

print("\n")
print("XGBoost ↔ GraphSAGE probability correlation:")
print(f"{correlation:.4f}")


# ============================================================
# EXTREME DISAGREEMENT
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 MOST DISAGREEING TRANSACTIONS")
print("=" * 70)

top_disagreement = (
    df
    .sort_values(
        "disagreement",
        ascending=False
    )
    [
        [
            "txId",
            "timestep",
            "label",
            "xgboost_probability",
            "graphsage_probability",
            "disagreement",
            "agreement_group"
        ]
    ]
    .head(20)
)

print(
    top_disagreement.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)