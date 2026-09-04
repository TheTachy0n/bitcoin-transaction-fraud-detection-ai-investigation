# ============================================================
# STEP 5A — THRESHOLD OPTIMIZATION
# ELLIPTIC BITCOIN FRAUD DETECTION
#
# Purpose:
# Find the best classification threshold for:
#   1. XGBoost
#   2. GraphSAGE
#   3. 90/10 Hybrid
#
# IMPORTANT:
# Thresholds are optimized ONLY on validation data.
# Test data will be used later for final evaluation.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALIGNED_PATH = (
    PROJECT_ROOT
    / "results"
    / "aligned_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "threshold_optimization.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

# Threshold search range
THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 5A — THRESHOLD OPTIMIZATION")
print("=" * 70)


# ============================================================
# LOAD ALIGNED VALIDATION DATA
# ============================================================

print("\nLoading aligned validation predictions...")

df = pd.read_csv(
    ALIGNED_PATH
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
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "txId",
    "timestep",
    "label",
    "xgboost_probability",
    "graphsage_probability"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    required_columns
].copy()


# Remove rows with missing values

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
# CREATE HYBRID PROBABILITY
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


print("\nGround-truth distribution:")

print(
    pd.Series(y_true)
    .value_counts()
    .sort_index()
)


print(
    "\nFraud cases:",
    np.sum(y_true == 1)
)

print(
    "Legitimate cases:",
    np.sum(y_true == 0)
)


# ============================================================
# MODEL PROBABILITIES
# ============================================================

model_probabilities = {

    "XGBoost":
        df["xgboost_probability"].values,

    "GraphSAGE":
        df["graphsage_probability"].values,

    "90/10 Hybrid":
        df["hybrid_probability"].values
}


# ============================================================
# THRESHOLD OPTIMIZATION FUNCTION
# ============================================================

def optimize_threshold(
    y_true,
    probabilities,
    model_name
):

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )


        results.append({

            "model":
                model_name,

            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "fraud_predictions":
                int(predictions.sum())

        })


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Best threshold according to F1
    # --------------------------------------------------------

    best_row = (
        results_df
        .sort_values(
            "f1",
            ascending=False
        )
        .iloc[0]
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        model_name
    )

    print(
        "=" * 70
    )


    print(
        "\nPR-AUC:",
        f"{average_precision_score(y_true, probabilities):.4f}"
    )


    print(
        "ROC-AUC:",
        f"{roc_auc_score(y_true, probabilities):.4f}"
    )


    print(
        "\nBEST THRESHOLD BY F1"
    )


    print(
        "Threshold:",
        f"{best_row['threshold']:.2f}"
    )


    print(
        "Precision:",
        f"{best_row['precision']:.4f}"
    )


    print(
        "Recall:",
        f"{best_row['recall']:.4f}"
    )


    print(
        "F1:",
        f"{best_row['f1']:.4f}"
    )


    print(
        "Fraud predictions:",
        int(best_row["fraud_predictions"])
    )


    return results_df


# ============================================================
# RUN OPTIMIZATION
# ============================================================

all_results = []


for model_name, probabilities in model_probabilities.items():

    results = optimize_threshold(
        y_true,
        probabilities,
        model_name
    )

    all_results.append(
        results
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

results_df = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# BEST THRESHOLDS SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST THRESHOLDS SUMMARY"
)

print(
    "=" * 70
)


best_thresholds = (
    results_df
    .sort_values(
        ["model", "f1"],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "model",
        as_index=False
    )
    .first()
)


print(
    best_thresholds[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "fraud_predictions"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# OPTIONAL — SHOW TOP THRESHOLDS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 5 THRESHOLDS BY F1"
)

print(
    "=" * 70
)


for model_name in model_probabilities.keys():

    print(
        f"\n{model_name}"
    )

    model_results = (
        results_df[
            results_df["model"]
            == model_name
        ]
        .sort_values(
            "f1",
            ascending=False
        )
        .head(5)
    )


    print(
        model_results[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "fraud_predictions"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STEP 5A COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)