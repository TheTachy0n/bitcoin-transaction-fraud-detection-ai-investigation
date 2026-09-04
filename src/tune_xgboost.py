# ============================================================
# XGBOOST HYPERPARAMETER TUNING
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

import os
import time
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"

RESULTS_PATH = "results/xgboost_tuning_results.csv"
MODEL_DIR = "models/xgb_tuning"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading training data...")
    train = pd.read_csv(TRAIN_PATH)

    print("Training shape:", train.shape)

    print("Loading validation data...")
    val = pd.read_csv(VAL_PATH)

    print("Validation shape:", val.shape)

    return train, val


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    X = df[feature_columns]
    y = df["label"]

    return X, y


# ============================================================
# FIND BEST F1 THRESHOLD
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train, val = load_data()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)

    print("\nFeature matrices:")

    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count /
        positive_count
    )

    print("\nClass distribution:")

    print(
        "Negative:",
        negative_count
    )

    print(
        "Positive:",
        positive_count
    )

    print(
        "scale_pos_weight:",
        round(
            scale_pos_weight,
            4
        )
    )

    # --------------------------------------------------------
    # EXPERIMENT CONFIGURATIONS
    # --------------------------------------------------------

    experiments = [

        {
            "name": "baseline",

            "max_depth": 6,
            "min_child_weight": 3,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },

        {
            "name": "depth_4",

            "max_depth": 4,
            "min_child_weight": 3,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },

        {
            "name": "depth_8",

            "max_depth": 8,
            "min_child_weight": 3,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },

        {
            "name": "min_child_1",

            "max_depth": 6,
            "min_child_weight": 1,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },

        {
            "name": "min_child_7",

            "max_depth": 6,
            "min_child_weight": 7,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },

        {
            "name": "strong_regularization",

            "max_depth": 6,
            "min_child_weight": 5,

            "learning_rate": 0.03,

            "subsample": 0.8,
            "colsample_bytree": 0.8,

            "reg_alpha": 0.5,
            "reg_lambda": 2.0
        }
    ]

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    results = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    for i, config in enumerate(
        experiments,
        start=1
    ):

        name = config["name"]

        print("\n")
        print("=" * 70)

        print(
            f"EXPERIMENT {i}/{len(experiments)}: "
            f"{name}"
        )

        print("=" * 70)

        print(
            "max_depth:",
            config["max_depth"]
        )

        print(
            "min_child_weight:",
            config["min_child_weight"]
        )

        print(
            "learning_rate:",
            config["learning_rate"]
        )

        print(
            "reg_alpha:",
            config["reg_alpha"]
        )

        print(
            "reg_lambda:",
            config["reg_lambda"]
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = XGBClassifier(

            n_estimators=1000,

            learning_rate=config[
                "learning_rate"
            ],

            max_depth=config[
                "max_depth"
            ],

            min_child_weight=config[
                "min_child_weight"
            ],

            subsample=config[
                "subsample"
            ],

            colsample_bytree=config[
                "colsample_bytree"
            ],

            gamma=0,

            reg_alpha=config[
                "reg_alpha"
            ],

            reg_lambda=config[
                "reg_lambda"
            ],

            objective="binary:logistic",

            eval_metric="aucpr",

            scale_pos_weight=(
                scale_pos_weight
            ),

            tree_method="hist",

            random_state=42,

            n_jobs=-1
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        start_time = time.time()

        print("\nTraining...")

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ],

            verbose=False
        )

        elapsed = (
            time.time() -
            start_time
        )

        print(
            f"Training time: "
            f"{elapsed:.2f} seconds"
        )

        # ----------------------------------------------------
        # VALIDATION PROBABILITIES
        # ----------------------------------------------------

        val_probabilities = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        pr_auc = (
            average_precision_score(
                y_val,
                val_probabilities
            )
        )

        roc_auc = (
            roc_auc_score(
                y_val,
                val_probabilities
            )
        )

        threshold, best_f1 = (
            find_best_threshold(
                y_val,
                val_probabilities
            )
        )

        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        precision = (
            precision_score(
                y_val,
                predictions,
                zero_division=0
            )
        )

        recall = (
            recall_score(
                y_val,
                predictions,
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # BEST ITERATION
        # ----------------------------------------------------

        best_iteration = getattr(
            model,
            "best_iteration",
            None
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        model_path = os.path.join(
            MODEL_DIR,
            f"{name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        result = {

            "model": name,

            "pr_auc": pr_auc,

            "roc_auc": roc_auc,

            "precision": precision,

            "recall": recall,

            "f1": best_f1,

            "threshold": threshold,

            "best_iteration":
                best_iteration,

            "training_seconds":
                elapsed,

            "max_depth":
                config["max_depth"],

            "min_child_weight":
                config["min_child_weight"],

            "learning_rate":
                config["learning_rate"],

            "subsample":
                config["subsample"],

            "colsample_bytree":
                config[
                    "colsample_bytree"
            ],

            "reg_alpha":
                config["reg_alpha"],

            "reg_lambda":
                config["reg_lambda"]
        }

        results.append(result)

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print("\nValidation result:")

        print(
            f"PR-AUC     : {pr_auc:.4f}"
        )

        print(
            f"ROC-AUC    : {roc_auc:.4f}"
        )

        print(
            f"Precision  : {precision:.4f}"
        )

        print(
            f"Recall     : {recall:.4f}"
        )

        print(
            f"F1         : {best_f1:.4f}"
        )

        print(
            f"Threshold  : {threshold:.2f}"
        )

        print(
            f"Model saved: {model_path}"
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("XGBOOST TUNING RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "threshold"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        "Model:",
        best_model["model"]
    )

    print(
        f"PR-AUC: "
        f"{best_model['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{best_model['roc_auc']:.4f}"
    )

    print(
        f"F1: "
        f"{best_model['f1']:.4f}"
    )

    print(
        f"Threshold: "
        f"{best_model['threshold']:.2f}"
    )

    print("\nResults saved:")
    print(
        os.path.abspath(
            RESULTS_PATH
        )
    )

    print("\nTuning complete.")


if __name__ == "__main__":
    main()