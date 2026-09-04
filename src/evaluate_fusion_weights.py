from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


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
    / "fusion_weight_analysis.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DAY 3 — FUSION WEIGHT ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

y_true = df["label"].astype(int).values

xgb = df["xgboost_probability"].values
gnn = df["graphsage_probability"].values


# ============================================================
# WEIGHT CONFIGURATIONS
# ============================================================

xgb_weights = np.arange(
    0.0,
    1.01,
    0.10
)

results = []


# ============================================================
# EVALUATE
# ============================================================

for xgb_weight in xgb_weights:

    gnn_weight = 1.0 - xgb_weight

    risk_score = (
        xgb_weight * xgb
        +
        gnn_weight * gnn
    )

    # Fixed threshold for comparison.
    #
    # We will optimize thresholds separately later.

    predictions = (
        risk_score >= 0.50
    ).astype(int)

    results.append({

        "xgb_weight": xgb_weight,

        "gnn_weight": gnn_weight,

        "pr_auc": average_precision_score(
            y_true,
            risk_score
        ),

        "roc_auc": roc_auc_score(
            y_true,
            risk_score
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("FUSION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST CONFIGURATIONS
# ============================================================

print("\n")
print("=" * 70)
print("BEST BY PR-AUC")
print("=" * 70)

best_pr_auc = (
    results_df
    .sort_values(
        "pr_auc",
        ascending=False
    )
    .head(3)
)

print(
    best_pr_auc.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("BEST BY F1")
print("=" * 70)

best_f1 = (
    results_df
    .sort_values(
        "f1",
        ascending=False
    )
    .head(3)
)

print(
    best_f1.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)
