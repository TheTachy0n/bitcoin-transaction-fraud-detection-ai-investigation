# ============================================================
# XGBOOST BASELINE — ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/xgboost_best.pkl"
PREDICTION_PATH = "results/xgboost_test_predictions.csv"
METRICS_PATH = "results/xgboost_metrics.csv"
CURVE_PATH = "results/xgboost_training_curve.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data():

    print("Loading train.csv...")
    train = pd.read_csv(TRAIN_PATH)

    print("Shape:", train.shape)

    print("Loading validation.csv...")
    val = pd.read_csv(VAL_PATH)

    print("Shape:", val.shape)

    print("Loading test.csv...")
    test = pd.read_csv(TEST_PATH)

    print("Shape:", test.shape)

    return train, val, test


def prepare_features(df):

    # Remove metadata and target columns.
    #
    # We ONLY want the 165 transaction features.

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    X = df[feature_columns]

    y = df["label"]

    return X, y


def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_f1 = 0.0

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


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            predictions
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
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        )
    }

    cm = confusion_matrix(
        y_true,
        predictions
    )

    return metrics, cm


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("XGBOOST BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train, val, test = load_data()

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)

    print("\nFeature matrix:")

    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    print("\nTraining labels:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # HANDLE CLASS IMBALANCE
    # --------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\nClass imbalance:")
    print("Negative samples:", negative_count)
    print("Positive samples:", positive_count)
    print(
        "scale_pos_weight:",
        round(scale_pos_weight, 4)
    )

    # --------------------------------------------------------
    # CREATE XGBOOST MODEL
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=1000,

        learning_rate=0.03,

        max_depth=6,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        gamma=0,

        reg_alpha=0.1,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        tree_method="hist",

        random_state=42,

        n_jobs=-1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],

        verbose=50
    )

    print("\nTraining complete.")

    # --------------------------------------------------------
    # TRAINING CURVE
    # --------------------------------------------------------

    evals_result = model.evals_result()

    train_aucpr = evals_result["validation_0"]["aucpr"]
    val_aucpr = evals_result["validation_1"]["aucpr"]

    curve = pd.DataFrame({

        "iteration": np.arange(
            len(train_aucpr)
        ),

        "train_aucpr": train_aucpr,

        "validation_aucpr": val_aucpr

    })

    os.makedirs("results", exist_ok=True)

    curve.to_csv(
        CURVE_PATH,
        index=False
    )

    # --------------------------------------------------------
    # BEST ITERATION
    # --------------------------------------------------------

    if hasattr(model, "best_iteration"):

        print("\nBest iteration:")
        print(model.best_iteration)

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # VALIDATION @ 0.5
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    val_metrics_05, val_cm_05 = calculate_metrics(
        y_val,
        val_probabilities,
        0.5
    )

    print("\nValidation metrics @ threshold 0.5:")

    for metric, value in val_metrics_05.items():

        print(
            f"{metric:12s}: {value:.4f}"
        )

    print("\nValidation confusion matrix:")
    print(val_cm_05)

    # --------------------------------------------------------
    # FIND BEST VALIDATION THRESHOLD
    # --------------------------------------------------------

    best_threshold, best_val_f1 = find_best_threshold(
        y_val,
        val_probabilities
    )

    print("\nBest validation threshold:")

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation F1: {best_val_f1:.4f}"
    )

    # --------------------------------------------------------
    # VALIDATION @ OPTIMIZED THRESHOLD
    # --------------------------------------------------------

    val_metrics, val_cm = calculate_metrics(
        y_val,
        val_probabilities,
        best_threshold
    )

    print("\nValidation metrics @ optimized threshold:")

    for metric, value in val_metrics.items():

        print(
            f"{metric:12s}: {value:.4f}"
        )

    print("\nValidation confusion matrix:")
    print(val_cm)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics, test_cm = calculate_metrics(
        y_test,
        test_probabilities,
        best_threshold
    )

    print(
        f"\nTest metrics @ threshold "
        f"{best_threshold:.2f}:"
    )

    for metric, value in test_metrics.items():

        print(
            f"{metric:12s}: {value:.4f}"
        )

    print("\nTest confusion matrix:")
    print(test_cm)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(
        os.path.abspath(MODEL_PATH)
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    prediction_df = pd.DataFrame({

        "txId": test["txId"],

        "timestep": test["timestep"],

        "actual_label": y_test,

        "fraud_probability": test_probabilities,

        "predicted_label": test_predictions

    })

    os.makedirs("results", exist_ok=True)

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False
    )

    print("\nTest predictions saved:")
    print(
        os.path.abspath(PREDICTION_PATH)
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_row = {

        "model": "XGBoost",

        "threshold": best_threshold,

        "best_iteration": getattr(
            model,
            "best_iteration",
            None
        ),

        "accuracy": test_metrics["accuracy"],

        "precision": test_metrics["precision"],

        "recall": test_metrics["recall"],

        "f1": test_metrics["f1"],

        "roc_auc": test_metrics["roc_auc"],

        "pr_auc": test_metrics["pr_auc"]

    }

    metrics_df = pd.DataFrame(
        [metrics_row]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print("\nMetrics saved:")
    print(
        os.path.abspath(METRICS_PATH)
    )

    # --------------------------------------------------------
    # SAVE THRESHOLD INFORMATION
    # --------------------------------------------------------

    threshold_info = {

        "optimized_threshold": float(
            best_threshold
        ),

        "validation_f1": float(
            best_val_f1
        ),

        "test_f1": float(
            test_metrics["f1"]
        )

    }

    with open(
        "results/xgboost_threshold.json",
        "w"
    ) as f:

        json.dump(
            threshold_info,
            f,
            indent=4
        )

    print("\n" + "=" * 60)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()