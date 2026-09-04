# ============================================================
# STEP 5B — PROBABILITY CALIBRATION
# ELLIPTIC BITCOIN FRAUD DETECTION
#
# Purpose:
# Evaluate whether model probabilities are reliable.
#
# Models:
#   1. XGBoost
#   2. GraphSAGE
#   3. 90/10 Hybrid
#
# IMPORTANT:
# We are NOT changing the models yet.
# This step only evaluates calibration.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from sklearn.calibration import (
    calibration_curve
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
    / "calibration_analysis.csv"
)

CALIBRATION_CURVE_PATH = (
    PROJECT_ROOT
    / "results"
    / "calibration_curve_data.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

N_BINS = 10


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 5B — PROBABILITY CALIBRATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading aligned validation predictions...")

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
# CHECK COLUMNS
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
        f"Missing columns: {missing}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()

df = df.dropna(
    subset=[
        "label",
        "xgboost_probability",
        "graphsage_probability"
    ]
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
# CREATE HYBRID
# ============================================================

df["hybrid_probability"] = (
    XGB_WEIGHT
    * df["xgboost_probability"]
    +
    GNN_WEIGHT
    * df["graphsage_probability"]
)


# ============================================================
# GROUND TRUTH
# ============================================================

y_true = (
    df["label"]
    .astype(int)
    .values
)


# ============================================================
# MODEL PROBABILITIES
# ============================================================

models = {

    "XGBoost":
        df["xgboost_probability"].values,

    "GraphSAGE":
        df["graphsage_probability"].values,

    "90/10 Hybrid":
        df["hybrid_probability"].values
}


# ============================================================
# CALIBRATION METRICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CALIBRATION METRICS"
)

print(
    "=" * 70
)


metric_results = []


for model_name, probabilities in models.items():

    # --------------------------------------------------------
    # Brier Score
    # --------------------------------------------------------

    brier = brier_score_loss(
        y_true,
        probabilities
    )


    # --------------------------------------------------------
    # Log Loss
    # --------------------------------------------------------

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1]
    )


    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )


    metric_results.append({

        "model":
            model_name,

        "brier_score":
            brier,

        "log_loss":
            logloss,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc

    })


    print(
        f"\n{model_name}"
    )

    print(
        f"Brier Score : {brier:.6f}"
    )

    print(
        f"Log Loss    : {logloss:.6f}"
    )

    print(
        f"PR-AUC      : {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )


# ============================================================
# CALIBRATION CURVES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CALIBRATION CURVES"
)

print(
    "=" * 70
)


curve_results = []


for model_name, probabilities in models.items():

    fraction_positive, mean_predicted = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=N_BINS,
            strategy="uniform"
        )
    )


    print(
        f"\n{model_name}"
    )


    for i in range(
        len(mean_predicted)
    ):

        print(
            f"Bin {i + 1:02d} | "
            f"Predicted = "
            f"{mean_predicted[i]:.4f} | "
            f"Actual = "
            f"{fraction_positive[i]:.4f}"
        )


        curve_results.append({

            "model":
                model_name,

            "bin":
                i + 1,

            "mean_predicted_probability":
                mean_predicted[i],

            "actual_fraction_positive":
                fraction_positive[i]

        })


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metric_results
)

metrics_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE CALIBRATION CURVE DATA
# ============================================================

curve_df = pd.DataFrame(
    curve_results
)

curve_df.to_csv(
    CALIBRATION_CURVE_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CALIBRATION SUMMARY"
)

print(
    "=" * 70
)

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "INTERPRETATION"
)

print(
    "=" * 70
)

print(
    """
Lower Brier Score = better probability calibration.
Lower Log Loss    = better probability calibration.

Higher PR-AUC      = better fraud ranking.
Higher ROC-AUC     = better discrimination.

Calibration and classification performance are
different properties.

A model can have excellent PR-AUC while still
having poorly calibrated probabilities.
"""
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
    CALIBRATION_CURVE_PATH
)

print(
    "\n" + "=" * 70
)

print(
    "STEP 5B COMPLETE"
)

print(
    "=" * 70
)