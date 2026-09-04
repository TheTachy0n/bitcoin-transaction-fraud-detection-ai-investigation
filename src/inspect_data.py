from data_loader import load_elliptic_data


def inspect_dataset():
    features, edges, classes = load_elliptic_data()

    print("\n" + "=" * 60)
    print("FEATURE DATASET")
    print("=" * 60)

    print("\nShape:")
    print(features.shape)

    print("\nFirst 5 rows:")
    print(features.head().to_string())

    print("\nFirst 10 columns of first row:")
    print(features.iloc[0, :10].tolist())

    print("\nColumn 0 unique values:")
    print(features.iloc[:, 0].nunique())

    print("\nColumn 1 unique values:")
    print(features.iloc[:, 1].nunique())

    print("\nColumn 0 value counts:")
    print(features.iloc[:, 0].value_counts().head(10))

    print("\nColumn 1 value counts:")
    print(features.iloc[:, 1].value_counts().sort_index())

    print("\n" + "=" * 60)
    print("CLASS DATASET")
    print("=" * 60)

    print("\nClass distribution:")
    print(classes["class"].value_counts(dropna=False))

    print("\nClass percentages:")
    print(
        classes["class"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    print("\n" + "=" * 60)
    print("EDGE DATASET")
    print("=" * 60)

    print("\nShape:")
    print(edges.shape)

    print("\nFirst 5 edges:")
    print(edges.head().to_string())


if __name__ == "__main__":
    inspect_dataset()