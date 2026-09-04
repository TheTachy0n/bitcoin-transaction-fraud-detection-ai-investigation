# ============================================================
# STEP 9
# XGBOOST vs HYBRID TEST ANALYSIS
#
# Purpose:
# 1. Evaluate XGBoost on the exact same test transactions
#    used by the Hybrid model.
# 2. Evaluate Hybrid across multiple thresholds.
# 3. Determine whether GraphSAGE should influence the
#    final risk score or remain supporting evidence.
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
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

XGB_TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost_test_predictions.csv"
)

HYBRID_TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "hybrid_fusion_test.csv"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "results"
    / "step9_test_comparison.csv"
)

OUTPUT_THRESHOLDS = (
    PROJECT_ROOT
    / "results"
    / "step9_threshold_analysis.csv"
)


# ============================================================
# HELPER
# ============================================================

def evaluate_at_threshold(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,

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
        ),

        "predicted_fraud": int(
            predictions.sum()
        ),

        "actual_fraud": int(
            y_true.sum()
        )
    }


def ranking_metrics(
    y_true,
    probabilities
):

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 9 — XGBOOST vs HYBRID TEST ANALYSIS")
    print("=" * 75)

    # ========================================================
    # LOAD XGBOOST
    # ========================================================

    print("\nLoading XGBoost test predictions...")

    xgb = pd.read_csv(
        XGB_TEST_PATH
    )

    print(
        "XGBoost rows:",
        len(xgb)
    )

    # --------------------------------------------------------
    # Normalize XGBoost column names
    # --------------------------------------------------------

    if "fraud_probability" in xgb.columns:

        xgb = xgb.rename(
            columns={
                "fraud_probability":
                    "xgboost_probability"
            }
        )

    if "actual_label" in xgb.columns:

        xgb = xgb.rename(
            columns={
                "actual_label":
                    "label"
            }
        )

    required_xgb = [
        "txId",
        "timestep",
        "label",
        "xgboost_probability"
    ]

    missing = [
        c
        for c in required_xgb
        if c not in xgb.columns
    ]

    if missing:

        raise ValueError(
            f"Missing XGBoost columns: {missing}"
        )

    xgb = xgb[
        required_xgb
    ].copy()

    xgb = xgb.drop_duplicates(
        subset=["txId"]
    )


    # ========================================================
    # LOAD HYBRID
    # ========================================================

    print("\nLoading Hybrid test predictions...")

    hybrid = pd.read_csv(
        HYBRID_TEST_PATH
    )

    print(
        "Hybrid rows:",
        len(hybrid)
    )

    required_hybrid = [
        "txId",
        "hybrid_probability"
    ]

    missing = [
        c
        for c in required_hybrid
        if c not in hybrid.columns
    ]

    if missing:

        raise ValueError(
            f"Missing Hybrid columns: {missing}"
        )

    hybrid = hybrid[
        [
            "txId",
            "hybrid_probability"
        ]
    ].copy()

    hybrid = hybrid.drop_duplicates(
        subset=["txId"]
    )


    # ========================================================
    # ALIGN
    # ========================================================

    print("\nAligning XGBoost and Hybrid...")

    comparison = pd.merge(
        xgb,
        hybrid,
        on="txId",
        how="inner"
    )

    print(
        "Common transactions:",
        len(comparison)
    )

    print(
        "XGBoost rows not matched:",
        len(xgb) - len(comparison)
    )

    print(
        "Hybrid rows not matched:",
        len(hybrid) - len(comparison)
    )

    if len(comparison) == 0:

        raise ValueError(
            "No common transactions found."
        )


    # ========================================================
    # LABELS
    # ========================================================

    y = comparison[
        "label"
    ].astype(int).values

    xgb_prob = comparison[
        "xgboost_probability"
    ].astype(float).values

    hybrid_prob = comparison[
        "hybrid_probability"
    ].astype(float).values


    # ========================================================
    # BASIC DATA CHECK
    # ========================================================

    print("\n" + "=" * 75)
    print("TEST DATA CHECK")
    print("=" * 75)

    print(
        "Total transactions:",
        len(y)
    )

    print(
        "Actual fraud:",
        int(y.sum())
    )

    print(
        "Actual legitimate:",
        int((y == 0).sum())
    )

    print(
        "Fraud rate:",
        f"{y.mean():.4%}"
    )


    # ========================================================
    # RANKING PERFORMANCE
    # ========================================================

    print("\n" + "=" * 75)
    print("RANKING PERFORMANCE")
    print("=" * 75)

    xgb_ranking = ranking_metrics(
        y,
        xgb_prob
    )

    hybrid_ranking = ranking_metrics(
        y,
        hybrid_prob
    )

    print(
        "\nXGBoost:"
    )

    print(
        f"PR-AUC  : "
        f"{xgb_ranking['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC : "
        f"{xgb_ranking['roc_auc']:.4f}"
    )

    print(
        "\nHybrid:"
    )

    print(
        f"PR-AUC  : "
        f"{hybrid_ranking['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC : "
        f"{hybrid_ranking['roc_auc']:.4f}"
    )

    print(
        "\nHybrid improvement over XGBoost:"
    )

    print(
        f"PR-AUC difference : "
        f"{hybrid_ranking['pr_auc'] - xgb_ranking['pr_auc']:+.4f}"
    )

    print(
        f"ROC-AUC difference: "
        f"{hybrid_ranking['roc_auc'] - xgb_ranking['roc_auc']:+.4f}"
    )


    # ========================================================
    # XGBOOST THRESHOLD ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("XGBOOST THRESHOLD ANALYSIS")
    print("=" * 75)

    thresholds = np.arange(
        0.10,
        0.96,
        0.05
    )

    xgb_threshold_results = []

    for threshold in thresholds:

        result = evaluate_at_threshold(
            y,
            xgb_prob,
            float(threshold)
        )

        result["model"] = "XGBoost"

        xgb_threshold_results.append(
            result
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"Flagged={result['predicted_fraud']}"
        )


    # ========================================================
    # HYBRID THRESHOLD ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)
    print("HYBRID THRESHOLD ANALYSIS")
    print("=" * 75)

    hybrid_threshold_results = []

    for threshold in thresholds:

        result = evaluate_at_threshold(
            y,
            hybrid_prob,
            float(threshold)
        )

        result["model"] = "Hybrid"

        hybrid_threshold_results.append(
            result
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"Flagged={result['predicted_fraud']}"
        )


    # ========================================================
    # COMBINE THRESHOLD RESULTS
    # ========================================================

    threshold_df = pd.DataFrame(
        xgb_threshold_results
        + hybrid_threshold_results
    )

    threshold_df = threshold_df[
        [
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "predicted_fraud",
            "actual_fraud"
        ]
    ]

    threshold_df.to_csv(
        OUTPUT_THRESHOLDS,
        index=False
    )


    # ========================================================
    # BEST F1 THRESHOLD — TEST
    # ========================================================

    print("\n" + "=" * 75)
    print("BEST TEST F1 THRESHOLDS")
    print("=" * 75)

    xgb_best = max(
        xgb_threshold_results,
        key=lambda x: x["f1"]
    )

    hybrid_best = max(
        hybrid_threshold_results,
        key=lambda x: x["f1"]
    )

    print(
        "\nXGBoost best test F1:"
    )

    print(
        f"Threshold : "
        f"{xgb_best['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{xgb_best['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{xgb_best['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{xgb_best['f1']:.4f}"
    )

    print(
        f"Flagged   : "
        f"{xgb_best['predicted_fraud']}"
    )

    print(
        "\nHybrid best test F1:"
    )

    print(
        f"Threshold : "
        f"{hybrid_best['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{hybrid_best['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{hybrid_best['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{hybrid_best['f1']:.4f}"
    )

    print(
        f"Flagged   : "
        f"{hybrid_best['predicted_fraud']}"
    )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    print("\n" + "=" * 75)
    print("CONFUSION MATRICES AT BEST TEST F1")
    print("=" * 75)

    xgb_best_predictions = (
        xgb_prob
        >= xgb_best["threshold"]
    ).astype(int)

    hybrid_best_predictions = (
        hybrid_prob
        >= hybrid_best["threshold"]
    ).astype(int)

    print("\nXGBoost:")
    print(
        confusion_matrix(
            y,
            xgb_best_predictions
        )
    )

    print("\nHybrid:")
    print(
        confusion_matrix(
            y,
            hybrid_best_predictions
        )
    )


    # ========================================================
    # MODEL AGREEMENT / DISAGREEMENT
    # ========================================================

    print("\n" + "=" * 75)
    print("MODEL DISAGREEMENT")
    print("=" * 75)

    # Use 0.5 only for disagreement analysis.
    # This is NOT the final risk threshold.

    xgb_class = (
        xgb_prob >= 0.5
    ).astype(int)

    hybrid_class = (
        hybrid_prob >= 0.5
    ).astype(int)

    disagreement = (
        xgb_class != hybrid_class
    )

    print(
        "Disagreements:",
        int(disagreement.sum())
    )

    print(
        "Disagreement rate:",
        f"{disagreement.mean():.4%}"
    )

    if disagreement.sum() > 0:

        disagreement_labels = y[
            disagreement
        ]

        print(
            "Fraud among disagreements:",
            int(
                disagreement_labels.sum()
            )
        )

        print(
            "Fraud rate among disagreements:",
            f"{disagreement_labels.mean():.4%}"
        )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "pr_auc": xgb_ranking["pr_auc"],
                "roc_auc": xgb_ranking["roc_auc"]
            },
            {
                "model": "Hybrid",
                "pr_auc": hybrid_ranking["pr_auc"],
                "roc_auc": hybrid_ranking["roc_auc"]
            }
        ]
    )

    summary.to_csv(
        OUTPUT_METRICS,
        index=False
    )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 9 COMPLETE")
    print("=" * 75)

    if (
        hybrid_ranking["pr_auc"]
        > xgb_ranking["pr_auc"]
    ):

        print(
            "\nRESULT:"
        )

        print(
            "Hybrid improves ranking performance "
            "over XGBoost."
        )

        print(
            "GraphSAGE may be useful inside the "
            "final risk score."
        )

    else:

        print(
            "\nRESULT:"
        )

        print(
            "Hybrid does NOT improve ranking "
            "performance over XGBoost."
        )

        print(
            "GraphSAGE should likely be treated "
            "as secondary graph evidence rather "
            "than a major component of the risk score."
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_METRICS
    )

    print(
        OUTPUT_THRESHOLDS
    )


if __name__ == "__main__":
    main()