# ============================================================
# FINAL XGBOOST MODEL — ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
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

MODEL_PATH = "models/xgboost_final.pkl"
THRESHOLD_PATH = "results/xgboost_final_threshold.json"


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
# FIND BEST THRESHOLD
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
    print("FINAL XGBOOST MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    train, val = load_data()

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)

    print("\nFeature matrices:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\nClass distribution:")
    print("Negative:", negative_count)
    print("Positive:", positive_count)

    print(
        "scale_pos_weight:",
        round(scale_pos_weight, 4)
    )

    # --------------------------------------------------------
    # FINAL MODEL CONFIGURATION
    # --------------------------------------------------------
    #
    # Selected from our optimization experiments:
    #
    # depth4_lr003_1500
    #
    # PR-AUC : 0.9336
    # ROC-AUC: 0.9748
    # F1     : 0.9243
    #
    # --------------------------------------------------------

    print("\nFinal configuration:")

    print("n_estimators       : 1500")
    print("learning_rate      : 0.03")
    print("max_depth          : 4")
    print("min_child_weight   : 3")
    print("reg_alpha          : 0.1")
    print("reg_lambda         : 1.0")
    print("subsample          : 0.8")
    print("colsample_bytree   : 0.8")

    model = XGBClassifier(

        n_estimators=1500,

        learning_rate=0.03,

        max_depth=4,

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

    print("\n" + "=" * 70)
    print("TRAINING FINAL XGBOOST")
    print("=" * 70)

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],

        verbose=100
    )

    print("\nTraining complete.")

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating validation probabilities...")

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    # --------------------------------------------------------
    # OPTIMIZE THRESHOLD
    # --------------------------------------------------------

    best_threshold, best_f1 = find_best_threshold(
        y_val,
        val_probabilities
    )

    val_predictions = (
        val_probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        val_predictions
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL XGBOOST VALIDATION RESULTS")
    print("=" * 70)

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
        f"F1         : {f1:.4f}"
    )

    print(
        f"Threshold  : {best_threshold:.2f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # BEST ITERATION
    # --------------------------------------------------------

    if hasattr(model, "best_iteration"):

        print(
            "\nBest iteration:",
            model.best_iteration
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nFinal model saved:")
    print(
        os.path.abspath(MODEL_PATH)
    )

    # --------------------------------------------------------
    # SAVE THRESHOLD INFORMATION
    # --------------------------------------------------------

    threshold_info = {

        "model": "XGBoost",

        "n_estimators": 1500,

        "learning_rate": 0.03,

        "max_depth": 4,

        "min_child_weight": 3,

        "reg_alpha": 0.1,

        "reg_lambda": 1.0,

        "optimized_threshold": float(
            best_threshold
        ),

        "validation_pr_auc": float(
            pr_auc
        ),

        "validation_roc_auc": float(
            roc_auc
        ),

        "validation_precision": float(
            precision
        ),

        "validation_recall": float(
            recall
        ),

        "validation_f1": float(
            f1
        ),

        "best_iteration": int(
            getattr(
                model,
                "best_iteration",
                1499
            )
        )
    }

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        THRESHOLD_PATH,
        "w"
    ) as f:

        json.dump(
            threshold_info,
            f,
            indent=4
        )

    print("\nThreshold/configuration saved:")
    print(
        os.path.abspath(THRESHOLD_PATH)
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL XGBOOST TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()