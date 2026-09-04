from data_loader import load_elliptic_data


# Temporal split boundaries
TRAIN_END = 34
VAL_END = 41
TEST_END = 49


def create_temporal_split(features, classes):
    """
    Create a temporal train/validation/test split.

    Train:      timesteps 1-34
    Validation: timesteps 35-41
    Test:       timesteps 42-49

    Unknown transactions are excluded from the supervised
    train/validation/test sets.
    """

    # --------------------------------------------------------
    # Rename feature columns
    # --------------------------------------------------------

    feature_columns = (
        ["txId", "timestep"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    features = features.copy()
    features.columns = feature_columns

    # --------------------------------------------------------
    # Merge labels onto the COMPLETE feature dataset
    # --------------------------------------------------------

    data = features.merge(
        classes,
        on="txId",
        how="left"
    )

    # --------------------------------------------------------
    # Remove unknown transactions for supervised learning
    # --------------------------------------------------------

    known_data = data[
        data["class"] != "unknown"
    ].copy()

    # --------------------------------------------------------
    # Convert labels
    #
    # Elliptic:
    # 1 = illicit
    # 2 = licit
    #
    # Our ML target:
    # 1 = illicit
    # 0 = licit
    # --------------------------------------------------------

    known_data["label"] = known_data["class"].map({
        "1": 1,
        "2": 0,
        1: 1,
        2: 0
    })

    # --------------------------------------------------------
    # Temporal split
    # --------------------------------------------------------

    train = known_data[
        known_data["timestep"] <= TRAIN_END
    ].copy()

    validation = known_data[
        (known_data["timestep"] > TRAIN_END) &
        (known_data["timestep"] <= VAL_END)
    ].copy()

    test = known_data[
        (known_data["timestep"] > VAL_END) &
        (known_data["timestep"] <= TEST_END)
    ].copy()

    return train, validation, test


def print_split_summary(train, validation, test):

    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT SUMMARY")
    print("=" * 60)

    for name, dataset in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test)
    ]:

        print(f"\n{name}")
        print("-" * 40)

        print(f"Transactions: {len(dataset):,}")

        print(
            f"Timestep range: "
            f"{dataset['timestep'].min()} - "
            f"{dataset['timestep'].max()}"
        )

        print("\nClass distribution:")
        print(
            dataset["label"]
            .value_counts()
            .sort_index()
            .rename({
                0: "Licit",
                1: "Illicit"
            })
        )

        print("\nClass percentages:")
        print(
            dataset["label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .rename({
                0: "Licit",
                1: "Illicit"
            })
        )


if __name__ == "__main__":

    features, edges, classes = load_elliptic_data()

    train, validation, test = create_temporal_split(
        features,
        classes
    )

    print_split_summary(
        train,
        validation,
        test
    )

    # Save processed splits
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("\n" + "=" * 60)
    print("SPLITS SAVED")
    print("=" * 60)

    print(f"Train:      {PROCESSED_DIR / 'train.csv'}")
    print(f"Validation: {PROCESSED_DIR / 'validation.csv'}")
    print(f"Test:       {PROCESSED_DIR / 'test.csv'}")