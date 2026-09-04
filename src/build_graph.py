import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = "data/raw/elliptic_txs_features.csv"
EDGE_PATH = "data/raw/elliptic_txs_edgelist.csv"
CLASS_PATH = "data/raw/elliptic_txs_classes.csv"

OUTPUT_PATH = "data/processed/graph_data.pkl"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 60)

    features = pd.read_csv(
        FEATURE_PATH,
        header=None
    )

    edges = pd.read_csv(
        EDGE_PATH
    )

    classes = pd.read_csv(
        CLASS_PATH
    )

    print("\nFeatures shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Classes shape:", classes.shape)

    return features, edges, classes


# ============================================================
# PREPARE NODE DATA
# ============================================================

def prepare_nodes(features, classes):

    # Original structure:
    #
    # column 0 = txId
    # column 1 = timestep
    # columns 2–166 = 165 features

    feature_columns = (
        ["txId", "timestep"]
        + [f"feature_{i}" for i in range(165)]
    )

    features.columns = feature_columns

    # --------------------------------------------------------
    # Merge labels
    # --------------------------------------------------------

    nodes = features.merge(
        classes,
        on="txId",
        how="left"
    )

    # --------------------------------------------------------
    # Convert labels
    #
    # 1 = illicit
    # 2 = licit
    # unknown = -1
    # --------------------------------------------------------

    nodes["label"] = nodes["class"].map({
        1: 1,
        2: 0,
        "1": 1,
        "2": 0,
        "unknown": -1
    })

    return nodes


# ============================================================
# CREATE NODE INDEX
# ============================================================

def create_node_mapping(nodes):

    tx_ids = nodes["txId"].values

    tx_to_idx = {
        int(tx_id): idx
        for idx, tx_id in enumerate(tx_ids)
    }

    idx_to_tx = {
        idx: int(tx_id)
        for idx, tx_id in enumerate(tx_ids)
    }

    return tx_to_idx, idx_to_tx


# ============================================================
# BUILD EDGE INDEX
# ============================================================

def build_edges(edges, tx_to_idx):

    print("\nBuilding edge index...")

    edge_list = []

    missing_edges = 0

    for _, row in edges.iterrows():

        tx1 = int(row["txId1"])
        tx2 = int(row["txId2"])

        if tx1 not in tx_to_idx or tx2 not in tx_to_idx:

            missing_edges += 1

            continue

        idx1 = tx_to_idx[tx1]
        idx2 = tx_to_idx[tx2]

        edge_list.append(
            [idx1, idx2]
        )

        # ----------------------------------------------------
        # GraphSAGE will use an undirected graph.
        #
        # The raw Elliptic edge list represents connections.
        # Therefore add the reverse direction for message
        # passing.
        # ----------------------------------------------------

        edge_list.append(
            [idx2, idx1]
        )

    edge_index = np.array(
        edge_list,
        dtype=np.int64
    ).T

    print("Original edges:", len(edges))
    print("Missing edges:", missing_edges)
    print("Directed message-passing edges:", edge_index.shape[1])

    return edge_index


# ============================================================
# CREATE TEMPORAL MASKS
# ============================================================

def create_masks(nodes):

    timestep = nodes["timestep"].values

    train_mask = (
        (timestep >= 1)
        & (timestep <= 34)
    )

    val_mask = (
        (timestep >= 35)
        & (timestep <= 41)
    )

    test_mask = (
        (timestep >= 42)
        & (timestep <= 49)
    )

    return (
        train_mask,
        val_mask,
        test_mask
    )


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_masks(
    nodes,
    train_mask,
    val_mask,
    test_mask
):

    print("\n" + "=" * 60)
    print("TEMPORAL MASK VERIFICATION")
    print("=" * 60)

    print(
        "Train nodes:",
        train_mask.sum()
    )

    print(
        "Validation nodes:",
        val_mask.sum()
    )

    print(
        "Test nodes:",
        test_mask.sum()
    )

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    print(
        "\nTrain ∩ Validation:",
        np.sum(train_mask & val_mask)
    )

    print(
        "Train ∩ Test:",
        np.sum(train_mask & test_mask)
    )

    print(
        "Validation ∩ Test:",
        np.sum(val_mask & test_mask)
    )

    # --------------------------------------------------------
    # Check coverage
    # --------------------------------------------------------

    print(
        "\nTotal nodes covered:",
        (
            train_mask
            | val_mask
            | test_mask
        ).sum()
    )

    print(
        "Total nodes:",
        len(nodes)
    )

    # --------------------------------------------------------
    # Timestep ranges
    # --------------------------------------------------------

    print(
        "\nTrain timesteps:",
        nodes.loc[
            train_mask,
            "timestep"
        ].min(),
        "→",
        nodes.loc[
            train_mask,
            "timestep"
        ].max()
    )

    print(
        "Validation timesteps:",
        nodes.loc[
            val_mask,
            "timestep"
        ].min(),
        "→",
        nodes.loc[
            val_mask,
            "timestep"
        ].max()
    )

    print(
        "Test timesteps:",
        nodes.loc[
            test_mask,
            "timestep"
        ].min(),
        "→",
        nodes.loc[
            test_mask,
            "timestep"
        ].max()
    )


# ============================================================
# SAVE GRAPH
# ============================================================

def save_graph(
    nodes,
    edge_index,
    tx_to_idx,
    idx_to_tx,
    train_mask,
    val_mask,
    test_mask
):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    graph_data = {

        "nodes": nodes,

        "edge_index": edge_index,

        "tx_to_idx": tx_to_idx,

        "idx_to_tx": idx_to_tx,

        "train_mask": train_mask,

        "val_mask": val_mask,

        "test_mask": test_mask
    }

    with open(
        OUTPUT_PATH,
        "wb"
    ) as f:

        pickle.dump(
            graph_data,
            f
        )

    print("\n" + "=" * 60)
    print("GRAPH SAVED")
    print("=" * 60)

    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    features, edges, classes = load_data()

    nodes = prepare_nodes(
        features,
        classes
    )

    print(
        "\nNode table shape:",
        nodes.shape
    )

    tx_to_idx, idx_to_tx = create_node_mapping(
        nodes
    )

    edge_index = build_edges(
        edges,
        tx_to_idx
    )

    (
        train_mask,
        val_mask,
        test_mask
    ) = create_masks(
        nodes
    )

    verify_masks(
        nodes,
        train_mask,
        val_mask,
        test_mask
    )

    save_graph(
        nodes,
        edge_index,
        tx_to_idx,
        idx_to_tx,
        train_mask,
        val_mask,
        test_mask
    )


if __name__ == "__main__":
    main()