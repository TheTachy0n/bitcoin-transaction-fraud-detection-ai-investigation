# ============================================================
# STEP 8
# HYBRID RISK FUSION MODEL
# XGBoost + GraphSAGE -> Logistic Regression
# ============================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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

VALIDATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "aligned_validation_predictions.csv"
)

XGB_TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost_test_predictions.csv"
)

GNN_TEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "graphsage_test_predictions_v2.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hybrid_fusion.pkl"
)

VALIDATION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "hybrid_fusion_validation.csv"
)

TEST_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "hybrid_fusion_test.csv"
)

METRICS_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "hybrid_fusion_metrics.csv"
)


# ============================================================
# HELPER
# ============================================================

def calculate_metrics(y_true, probabilities, threshold=0.5):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
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
    }


def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID RISK FUSION")
    print("=" * 70)

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print("\nLoading aligned validation predictions...")

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        "Validation rows:",
        len(validation)
    )

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
        if col not in validation.columns
    ]

    if missing:

        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    # ========================================================
    # PREPARE META FEATURES
    # ========================================================

    X_validation = validation[
        [
            "xgboost_probability",
            "graphsage_probability"
        ]
    ].values

    y_validation = validation[
        "label"
    ].astype(int).values

    print("\nMeta-feature matrix:")
    print(
        "Shape:",
        X_validation.shape
    )

    print(
        "\nFeatures:"
    )

    print(
        "1. XGBoost probability"
    )

    print(
        "2. GraphSAGE probability"
    )

    # ========================================================
    # TRAIN LOGISTIC REGRESSION
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING HYBRID META-CLASSIFIER")
    print("=" * 70)

    fusion_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    fusion_model.fit(
        X_validation,
        y_validation
    )

    print("\nTraining complete.")

    # ========================================================
    # MODEL COEFFICIENTS
    # ========================================================

    print("\nLearned coefficients:")

    print(
        "XGBoost coefficient:",
        fusion_model.coef_[0][0]
    )

    print(
        "GraphSAGE coefficient:",
        fusion_model.coef_[0][1]
    )

    print(
        "Intercept:",
        fusion_model.intercept_[0]
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_probabilities = (
        fusion_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # ========================================================
    # OPTIMIZE VALIDATION THRESHOLD
    # ========================================================

    best_threshold, best_f1 = (
        find_best_threshold(
            y_validation,
            validation_probabilities
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        best_threshold
    )

    print("\n" + "=" * 70)
    print("HYBRID VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"PR-AUC     : "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Precision  : "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1         : "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{best_threshold:.2f}"
    )

    # ========================================================
    # VALIDATION CONFUSION MATRIX
    # ========================================================

    validation_predictions = (
        validation_probabilities
        >= best_threshold
    ).astype(int)

    print("\nValidation confusion matrix:")

    print(
        confusion_matrix(
            y_validation,
            validation_predictions
        )
    )

    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    validation_output = validation[
        [
            "txId",
            "timestep",
            "label"
        ]
    ].copy()

    validation_output[
        "xgboost_probability"
    ] = validation[
        "xgboost_probability"
    ]

    validation_output[
        "graphsage_probability"
    ] = validation[
        "graphsage_probability"
    ]

    validation_output[
        "hybrid_probability"
    ] = validation_probabilities

    validation_output[
        "hybrid_prediction"
    ] = validation_predictions

    validation_output.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    # ========================================================
    # LOAD TEST PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING TEST PREDICTIONS")
    print("=" * 70)

    print("\nLoading XGBoost test predictions...")

    xgb_test = pd.read_csv(
        XGB_TEST_PATH
    )

    print(
        "XGBoost rows:",
        len(xgb_test)
    )

    print("\nLoading GraphSAGE test predictions...")

    gnn_test = pd.read_csv(
        GNN_TEST_PATH
    )

    print(
        "GraphSAGE rows:",
        len(gnn_test)
    )

    # ========================================================
    # NORMALIZE COLUMN NAMES
    # ========================================================

    # XGBoost prediction file from the baseline uses:
    # fraud_probability / actual_label
    #
    # GraphSAGE uses:
    # graphsage_probability / label

    if "fraud_probability" in xgb_test.columns:

        xgb_test = xgb_test.rename(
            columns={
                "fraud_probability":
                    "xgboost_probability"
            }
        )

    if "actual_label" in xgb_test.columns:

        xgb_test = xgb_test.rename(
            columns={
                "actual_label":
                    "label"
            }
        )

    # ========================================================
    # ALIGN TEST DATA
    # ========================================================

    print("\nAligning test predictions by txId...")

    xgb_columns = [
        "txId",
        "timestep",
        "label",
        "xgboost_probability"
    ]

    gnn_columns = [
        "txId",
        "timestep",
        "label",
        "graphsage_probability"
    ]

    xgb_test = xgb_test[
        xgb_columns
    ].copy()

    gnn_test = gnn_test[
        gnn_columns
    ].copy()

    # Remove duplicate IDs if present

    xgb_test = xgb_test.drop_duplicates(
        subset=["txId"]
    )

    gnn_test = gnn_test.drop_duplicates(
        subset=["txId"]
    )

    # Inner join guarantees that the two
    # probabilities refer to the same transaction.

    test = pd.merge(
        xgb_test,
        gnn_test[
            [
                "txId",
                "graphsage_probability"
            ]
        ],
        on="txId",
        how="inner"
    )

    print(
        "Aligned test rows:",
        len(test)
    )

    print(
        "XGBoost rows lost:",
        len(xgb_test) - len(test)
    )

    print(
        "GraphSAGE rows lost:",
        len(gnn_test) - len(test)
    )

    if len(test) == 0:

        raise ValueError(
            "No overlapping test transactions found."
        )

    # ========================================================
    # TEST META FEATURES
    # ========================================================

    X_test = test[
        [
            "xgboost_probability",
            "graphsage_probability"
        ]
    ].values

    y_test = test[
        "label"
    ].astype(int).values

    # ========================================================
    # GENERATE HYBRID TEST PROBABILITIES
    # ========================================================

    print("\nGenerating hybrid test probabilities...")

    test_probabilities = (
        fusion_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    # ========================================================
    # TEST METRICS
    # ========================================================

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold
    )

    print("\n" + "=" * 70)
    print("HYBRID TEST RESULTS")
    print("=" * 70)

    print(
        f"PR-AUC     : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1         : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{best_threshold:.2f}"
    )

    print("\nTest confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_output = test.copy()

    test_output[
        "hybrid_probability"
    ] = test_probabilities

    test_output[
        "hybrid_prediction"
    ] = test_predictions

    test_output.to_csv(
        TEST_OUTPUT,
        index=False
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            fusion_model,
            f
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_rows = [

        {
            "model": "Hybrid Fusion",
            "dataset": "validation",
            "threshold": best_threshold,
            **validation_metrics
        },

        {
            "model": "Hybrid Fusion",
            "dataset": "test",
            "threshold": best_threshold,
            **test_metrics
        }

    ]

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        METRICS_OUTPUT,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Validation:",
        VALIDATION_OUTPUT
    )

    print(
        "Test:",
        TEST_OUTPUT
    )

    print(
        "Metrics:",
        METRICS_OUTPUT
    )

    print("\n" + "=" * 70)
    print("HYBRID RISK FUSION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()