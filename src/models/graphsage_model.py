# ============================================================
# GRAPHSAGE INFERENCE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_data.pkl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "graphsage_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
)


# ============================================================
# CONFIGURATION
# ============================================================

HIDDEN_DIM = 128
DROPOUT = 0.2


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            2
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    print("Loading graph...")

    with open(
        GRAPH_PATH,
        "rb"
    ) as f:

        graph = pickle.load(f)

    nodes = graph["nodes"]

    edge_index_np = graph["edge_index"]

    train_mask_np = graph["train_mask"]

    val_mask_np = graph["val_mask"]

    test_mask_np = graph["test_mask"]

    print(
        "Nodes:",
        len(nodes)
    )

    print(
        "Edges:",
        edge_index_np.shape[1]
    )

    return (
        graph,
        nodes,
        edge_index_np,
        train_mask_np,
        val_mask_np,
        test_mask_np
    )


# ============================================================
# PREPARE GRAPH DATA
# ============================================================

def prepare_graph(
    nodes,
    edge_index_np,
    train_mask_np,
    val_mask_np,
    test_mask_np
):

    feature_columns = [
        column
        for column in nodes.columns
        if column.startswith("feature_")
    ]

    print(
        "Number of features:",
        len(feature_columns)
    )

    X = nodes[
        feature_columns
    ].values.astype(
        np.float32
    )

    y = nodes[
        "label"
    ].values.astype(
        np.int64
    )

    x = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long
    )

    edge_index = torch.tensor(
        edge_index_np,
        dtype=torch.long
    )

    train_mask = torch.tensor(
        train_mask_np,
        dtype=torch.bool
    )

    val_mask = torch.tensor(
        val_mask_np,
        dtype=torch.bool
    )

    test_mask = torch.tensor(
        test_mask_np,
        dtype=torch.bool
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y_tensor
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data = data.to(DEVICE)

    return data, len(feature_columns)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading GraphSAGE checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # --------------------------------------------------------
    # Read architecture information saved during Day 1
    # --------------------------------------------------------

    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    dropout = checkpoint["dropout"]

    print("Checkpoint information:")
    print("Input dimension:", input_dim)
    print("Hidden dimension:", hidden_dim)
    print("Dropout:", dropout)
    print("Best epoch:", checkpoint["best_epoch"])
    print("Best validation PR-AUC:", checkpoint["best_val_pr_auc"])

    # --------------------------------------------------------
    # Recreate EXACT Day 1 architecture
    # --------------------------------------------------------

    model = GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load ONLY the trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("GraphSAGE model loaded successfully.")

    return model


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

@torch.no_grad()
def generate_predictions(
    model,
    data
):

    logits = model(
        data.x,
        data.edge_index
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[:, 1]

    return probabilities.cpu().numpy()


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

def create_predictions(
    nodes,
    probabilities,
    mask
):

    mask = np.asarray(mask)

    predictions = nodes.loc[
        mask,
        [
            "txId",
            "timestep",
            "label"
        ]
    ].copy()

    predictions[
        "graphsage_probability"
    ] = probabilities[mask]

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRAPHSAGE INFERENCE")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # LOAD GRAPH
    # --------------------------------------------------------

    (
        graph,
        nodes,
        edge_index_np,
        train_mask_np,
        val_mask_np,
        test_mask_np
    ) = load_graph()

    # --------------------------------------------------------
    # PREPARE GRAPH
    # --------------------------------------------------------

    (
        data,
        input_dim
    ) = prepare_graph(
        nodes,
        edge_index_np,
        train_mask_np,
        val_mask_np,
        test_mask_np
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = generate_predictions(
        model,
        data
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = create_predictions(
        nodes,
        probabilities,
        val_mask_np
    )

    print(
        "\nValidation shape:",
        val_predictions.shape
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = create_predictions(
        nodes,
        probabilities,
        test_mask_np
    )

    print(
        "Test shape:",
        test_predictions.shape
    )

    # --------------------------------------------------------
    # DISPLAY
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
    # PROBABILITY STATISTICS
    # --------------------------------------------------------

    print(
        "\nValidation probability statistics:"
    )

    print(
        val_predictions[
            "graphsage_probability"
        ].describe()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    val_output = (
        OUTPUT_DIR
        / "graphsage_validation_predictions.csv"
    )

    test_output = (
        OUTPUT_DIR
        / "graphsage_test_predictions_v2.csv"
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

    print("\nGraphSAGE inference complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()