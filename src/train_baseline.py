from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(filename):
    path = DATA_DIR / filename

    print(f"Loading {filename}...")
    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Extract transaction features.

    Columns:
        txId       -> identifier, excluded
        timestep   -> temporal information, excluded initially
        remaining  -> 165 transaction features
        label      -> target
    """

    X = df.drop(columns=["txId", "timestep", "class", "label"])

    y = df["label"]

    return X, y


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, threshold=0.5):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "pr_auc": average_precision_score(
            y, probabilities
        ),
    }

    cm = confusion_matrix(y, predictions)

    return metrics, cm, probabilities, predictions


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y,
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

    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df = load_split("train.csv")
    val_df = load_split("validation.csv")
    test_df = load_split("test.csv")

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    print("\nFeature matrix:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\nTraining labels:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Validation - default threshold
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    val_metrics, val_cm, _, _ = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.5
    )

    print("\nValidation metrics @ threshold 0.5:")

    for metric, value in val_metrics.items():
        print(f"{metric:12s}: {value:.4f}")

    print("\nValidation confusion matrix:")
    print(val_cm)

    # --------------------------------------------------------
    # Find threshold using validation ONLY
    # --------------------------------------------------------

    best_threshold, best_val_f1 = find_best_threshold(
        model,
        X_val,
        y_val
    )

    print("\nBest validation threshold:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Validation F1: {best_val_f1:.4f}")

    # --------------------------------------------------------
    # Final validation metrics
    # --------------------------------------------------------

    val_metrics_tuned, val_cm_tuned, _, _ = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=best_threshold
    )

    print("\nValidation metrics @ optimized threshold:")

    for metric, value in val_metrics_tuned.items():
        print(f"{metric:12s}: {value:.4f}")

    print("\nValidation confusion matrix:")
    print(val_cm_tuned)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST")
    print("=" * 60)

    test_metrics, test_cm, test_probabilities, test_predictions = (
        evaluate_model(
            model,
            X_test,
            y_test,
            threshold=best_threshold
        )
    )

    print(
        f"\nTest metrics @ threshold {best_threshold:.2f}:"
    )

    for metric, value in test_metrics.items():
        print(f"{metric:12s}: {value:.4f}")

    print("\nTest confusion matrix:")
    print(test_cm)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / "logistic_regression.pkl"

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:")
    print(model_path)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "txId": test_df["txId"],
        "y_true": y_test,
        "probability": test_probabilities,
        "prediction": test_predictions
    })

    predictions_path = (
        RESULTS_DIR / "logistic_regression_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("\nTest predictions saved:")
    print(predictions_path)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "threshold": best_threshold,
            **test_metrics
        }
    ])

    metrics_path = (
        RESULTS_DIR / "logistic_regression_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("\nMetrics saved:")
    print(metrics_path)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()