from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support
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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

y = df["label"].values

xgb = df["xgboost_probability"].values
gnn = df["graphsage_probability"].values


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    probabilities
):

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    # Default threshold
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    return {
        "Model": name,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "XGBoost",
        xgb
    )
)


# ------------------------------------------------------------
# GraphSAGE
# ------------------------------------------------------------

results.append(
    evaluate_model(
        "GraphSAGE",
        gnn
    )
)


# ------------------------------------------------------------
# Fusion weights
# ------------------------------------------------------------

weights = [
    0.90,
    0.75,
    0.50,
    0.25,
    0.10
]


for xgb_weight in weights:

    gnn_weight = 1 - xgb_weight

    fused_probability = (
        xgb_weight * xgb
        +
        gnn_weight * gnn
    )

    results.append(
        evaluate_model(
            f"Fusion XGB={xgb_weight:.2f} / GNN={gnn_weight:.2f}",
            fused_probability
        )
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "PR-AUC",
    ascending=False
)


print("=" * 75)
print("XGBOOST + GRAPHSAGE FUSION BENCHMARK")
print("=" * 75)

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "fusion_benchmark.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(
    "\nSaved:"
)

print(output_path)