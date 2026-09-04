from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_elliptic_data():
    """
    Load the three raw Elliptic Bitcoin dataset files.

    Returns:
        features: Transaction features and timestep
        edges: Transaction graph edges
        classes: Transaction labels
    """

    features_path = RAW_DATA_DIR / "elliptic_txs_features.csv"
    edges_path = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"
    classes_path = RAW_DATA_DIR / "elliptic_txs_classes.csv"

    print("Loading Elliptic dataset...")

    features = pd.read_csv(features_path, header=None)
    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path)

    print("\nDataset loaded successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Edges shape: {edges.shape}")
    print(f"Classes shape: {classes.shape}")

    return features, edges, classes


if __name__ == "__main__":
    load_elliptic_data()