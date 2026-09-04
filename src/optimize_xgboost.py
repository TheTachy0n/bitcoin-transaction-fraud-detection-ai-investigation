import os
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

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"

RESULTS_PATH = "results/xgboost_optimization_results.csv"


def prepare_features(df):
    feature_columns = [
        col for col in df.columns
        if col.startswith("feature_")
    ]

    return df[feature_columns], df["label"]


def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():

    print("=" * 70)
    print("XGBOOST TRAINING OPTIMIZATION")
    print("=" * 70)

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\nTraining:", X_train.shape)
    print("Validation:", X_val.shape)

    print(
        "scale_pos_weight:",
        round(scale_pos_weight, 4)
    )

    # --------------------------------------------------------
    # CONTROLLED EXPERIMENTS
    # --------------------------------------------------------

    experiments = [

        {
            "name": "depth4_lr003_500",
            "max_depth": 4,
            "min_child_weight": 3,
            "learning_rate": 0.03,
            "n_estimators": 500
        },

        {
            "name": "depth4_lr003_1000",
            "max_depth": 4,
            "min_child_weight": 3,
            "learning_rate": 0.03,
            "n_estimators": 1000
        },

        {
            "name": "depth4_lr003_1500",
            "max_depth": 4,
            "min_child_weight": 3,
            "learning_rate": 0.03,
            "n_estimators": 1500
        },

        {
            "name": "depth6_child1_lr003_500",
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.03,
            "n_estimators": 500
        },

        {
            "name": "depth6_child1_lr003_1000",
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.03,
            "n_estimators": 1000
        },

        {
            "name": "depth6_child1_lr003_1500",
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.03,
            "n_estimators": 1500
        },

        {
            "name": "depth6_child1_lr002_1500",
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.02,
            "n_estimators": 1500
        },

        {
            "name": "depth6_child1_lr005_1000",
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.05,
            "n_estimators": 1000
        }
    ]

    results = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    for i, config in enumerate(experiments, 1):

        print("\n" + "=" * 70)

        print(
            f"EXPERIMENT {i}/{len(experiments)}: "
            f"{config['name']}"
        )

        print("=" * 70)

        model = XGBClassifier(

            n_estimators=config["n_estimators"],

            learning_rate=config["learning_rate"],

            max_depth=config["max_depth"],

            min_child_weight=config[
                "min_child_weight"
            ],

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

        print("\nTraining...")

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ],

            verbose=False
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_val
        )[:, 1]

        pr_auc = average_precision_score(
            y_val,
            probabilities
        )

        roc_auc = roc_auc_score(
            y_val,
            probabilities
        )

        threshold, f1 = find_best_threshold(
            y_val,
            probabilities
        )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        # ----------------------------------------------------
        # BEST ITERATION / CURVE
        # ----------------------------------------------------

        evals_result = model.evals_result()

        val_curve = evals_result[
            "validation_0"
        ]["aucpr"]

        best_iteration = int(
            np.argmax(val_curve)
        )

        best_curve_pr_auc = float(
            np.max(val_curve)
        )

        result = {

            "model": config["name"],

            "n_estimators": config[
                "n_estimators"
            ],

            "learning_rate": config[
                "learning_rate"
            ],

            "max_depth": config[
                "max_depth"
            ],

            "min_child_weight": config[
                "min_child_weight"
            ],

            "best_iteration": best_iteration,

            "curve_best_pr_auc":
                best_curve_pr_auc,

            "pr_auc": pr_auc,

            "roc_auc": roc_auc,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "threshold": threshold
        }

        results.append(result)

        print("\nResult:")

        print(
            f"PR-AUC    : {pr_auc:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1        : {f1:.4f}"
        )

        print(
            f"Threshold : {threshold:.2f}"
        )

        print(
            f"Best curve iteration: "
            f"{best_iteration}"
        )

        print(
            f"Best curve PR-AUC: "
            f"{best_curve_pr_auc:.4f}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "best_iteration",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "threshold"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(
        os.path.abspath(
            RESULTS_PATH
        )
    )


if __name__ == "__main__":
    main()