import os
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/xgboost_final.pkl"
TRAIN_PATH = "data/processed/train.csv"
OUTPUT_PATH = "results/xgboost_feature_importance.csv"


def main():

    print("=" * 70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training data...")

    train = pd.read_csv(
        TRAIN_PATH
    )

    feature_columns = [
        col
        for col in train.columns
        if col.startswith("feature_")
    ]

    print(
        "Number of features:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # GET FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": feature_columns,

        "importance": importance

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # NORMALIZED IMPORTANCE
    # --------------------------------------------------------

    total = (
        importance_df["importance"]
        .sum()
    )

    importance_df[
        "importance_percent"
    ] = (
        importance_df["importance"]
        / total
        * 100
    )

    importance_df[
        "cumulative_percent"
    ] = (
        importance_df[
            "importance_percent"
        ].cumsum()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY TOP FEATURES
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 20 FEATURES")
    print("=" * 70)

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # TOP 10 CONTRIBUTION
    # --------------------------------------------------------

    top10 = (
        importance_df
        .head(10)
        ["importance_percent"]
        .sum()
    )

    top20 = (
        importance_df
        .head(20)
        ["importance_percent"]
        .sum()
    )

    print("\n")
    print(
        f"Top 10 features account for "
        f"{top10:.2f}% of total importance."
    )

    print(
        f"Top 20 features account for "
        f"{top20:.2f}% of total importance."
    )

    # --------------------------------------------------------
    # FEATURE COUNT FOR 80%
    # --------------------------------------------------------

    count_80 = (
        importance_df[
            "cumulative_percent"
        ] <= 80
    ).sum()

    print(
        f"\nFeatures needed to reach "
        f"80% importance: {count_80}"
    )

    print("\nSaved:")
    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )

    print("\nFeature importance analysis complete.")


if __name__ == "__main__":
    main()