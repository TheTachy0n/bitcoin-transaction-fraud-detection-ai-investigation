# ============================================================
# XGBOOST INFERENCE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_best.pkl"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded.")

    return model


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    feature_columns = [
        col
        for col in df.columns
        if col.startswith("feature_")
    ]

    X = df[feature_columns]

    return X


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def predict(model, df):

    X = prepare_features(df)

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities


# ============================================================
# PREDICT DATASET
# ============================================================

def predict_dataset(model, path):

    df = pd.read_csv(path)

    probabilities = predict(model, df)

    predictions = df[
        ["txId", "timestep", "label"]
    ].copy()

    predictions["xgboost_probability"] = probabilities

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("XGBOOST INFERENCE")
    print("=" * 60)

    model = load_model()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nGenerating validation predictions...")

    val_predictions = predict_dataset(
        model,
        VAL_PATH
    )

    print(
        "Validation shape:",
        val_predictions.shape
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    test_predictions = predict_dataset(
        model,
        TEST_PATH
    )

    print(
        "Test shape:",
        test_predictions.shape
    )

    # --------------------------------------------------------
    # DISPLAY SAMPLE
    # --------------------------------------------------------

    print("\nValidation sample:")

    print(
        val_predictions.head(10).to_string(
            index=False
        )
    )

    print("\nTest sample:")

    print(
        test_predictions.head(10).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BASIC PROBABILITY CHECK
    # --------------------------------------------------------

    print("\nProbability statistics:")

    print(
        val_predictions[
            "xgboost_probability"
        ].describe()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_dir = PROJECT_ROOT / "results"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_output = (
        output_dir /
        "xgboost_validation_predictions.csv"
    )

    test_output = (
        output_dir /
        "xgboost_test_predictions_v2.csv"
    )

    val_predictions.to_csv(
        val_output,
        index=False
    )

    test_predictions.to_csv(
        test_output,
        index=False
    )

    print("\nSaved:")

    print(val_output)
    print(test_output)

    print("\nXGBoost inference complete.")


if __name__ == "__main__":
    main()