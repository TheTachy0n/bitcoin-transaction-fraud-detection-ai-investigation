# ============================================================
# GRAPHSAGE BASELINE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    accuracy_score,
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

GRAPH_PATH = "data/processed/graph_data.pkl"

MODEL_PATH = "models/graphsage_best.pt"

PREDICTION_PATH = (
    "results/graphsage_test_predictions.csv"
)

METRICS_PATH = (
    "results/graphsage_metrics.csv"
)


SEED = 42

HIDDEN_DIM = 128

DROPOUT = 0.2

LEARNING_RATE = 0.005

WEIGHT_DECAY = 1e-4

EPOCHS = 300

PATIENCE = 30


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("GRAPHSAGE BASELINE")
print("=" * 60)

print("\nDevice:", DEVICE)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph...")

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


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    column
    for column in nodes.columns
    if column.startswith("feature_")
]


print(
    "\nNumber of features:",
    len(feature_columns)
)


X = nodes[
    feature_columns
].values.astype(
    np.float32
)


# ============================================================
# LABELS
# ============================================================

y = nodes[
    "label"
].values.astype(
    np.int64
)


# ============================================================
# PYTORCH TENSORS
# ============================================================

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


# ============================================================
# GRAPH DATA
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
    y=y_tensor
)

data.train_mask = train_mask

data.val_mask = val_mask

data.test_mask = test_mask


data = data.to(
    DEVICE
)


print("\nGraph moved to:", DEVICE)


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
# CREATE MODEL
# ============================================================

model = GraphSAGE(
    input_dim=len(feature_columns),
    hidden_dim=HIDDEN_DIM,
    dropout=DROPOUT
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = y[
    train_mask_np
]

negative_count = np.sum(
    train_labels == 0
)

positive_count = np.sum(
    train_labels == 1
)

pos_weight = (
    negative_count /
    positive_count
)


print("\nTraining class distribution:")

print(
    "Licit:",
    negative_count
)

print(
    "Illicit:",
    positive_count
)

print(
    "Positive class weight:",
    round(
        pos_weight,
        4
    )
)


class_weights = torch.tensor(
    [
        1.0,
        pos_weight
    ],
    dtype=torch.float32,
    device=DEVICE
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, mask):

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

    # --------------------------------------------------------
    # IMPORTANT:
    # Only evaluate nodes with known labels
    # label -1 = unknown
    # --------------------------------------------------------

    valid_mask = mask & (data.y != -1)

    y_true = (
        data.y[valid_mask]
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities[valid_mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[valid_mask]
        .cpu()
        .numpy()
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            )
    }

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return (
        metrics,
        cm,
        y_prob
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)


best_val_pr_auc = -np.inf

best_epoch = 0

patience_counter = 0

best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index
    )

    loss = criterion(
        logits[
            data.train_mask
        ],
        data.y[
            data.train_mask
        ]
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_metrics, _, _ = evaluate(
        model,
        data.val_mask
    )


    current_pr_auc = (
        val_metrics["pr_auc"]
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if current_pr_auc > best_val_pr_auc:

        best_val_pr_auc = (
            current_pr_auc
        )

        best_epoch = epoch

        patience_counter = 0

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
        or patience_counter == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Val PR-AUC: "
            f"{current_pr_auc:.4f} | "
            f"Best: "
            f"{best_val_pr_auc:.4f}"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch "
            f"{epoch}"
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation PR-AUC:",
    round(
        best_val_pr_auc,
        4
    )
)


model.load_state_dict(
    best_state
)


# ============================================================
# VALIDATION
# ============================================================

val_metrics, val_cm, _ = evaluate(
    model,
    data.val_mask
)


print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)


for key, value in val_metrics.items():

    print(
        f"{key:12s}: "
        f"{value:.4f}"
    )


print("\nValidation confusion matrix:")

print(val_cm)


# ============================================================
# TEST
# ============================================================

test_metrics, test_cm, test_prob = evaluate(
    model,
    data.test_mask
)


print("\n" + "=" * 60)
print("TEST")
print("=" * 60)


for key, value in test_metrics.items():

    print(
        f"{key:12s}: "
        f"{value:.4f}"
    )


print("\nTest confusion matrix:")

print(test_cm)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            len(feature_columns),

        "hidden_dim":
            HIDDEN_DIM,

        "dropout":
            DROPOUT,

        "best_epoch":
            best_epoch,

        "best_val_pr_auc":
            best_val_pr_auc
    },
    MODEL_PATH
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_valid_mask = (
    test_mask_np
    & (y != -1)
)

test_indices = np.where(
    test_valid_mask
)[0]


predictions_df = pd.DataFrame({

    "txId":
        nodes.iloc[
            test_indices
        ]["txId"].values,

    "timestep":
        nodes.iloc[
            test_indices
        ]["timestep"].values,

    "true_label":
        y[
            test_indices
        ],

    "probability":
        test_prob,

    "prediction":
        (
            test_prob >= 0.5
        ).astype(int)
})


predictions_df.to_csv(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": "GraphSAGE",
        "split": "validation",
        **val_metrics
    },
    {
        "model": "GraphSAGE",
        "split": "test",
        **test_metrics
    }
])


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("GRAPHSAGE BASELINE COMPLETE")
print("=" * 60)

print("\nModel saved:")
print(
    os.path.abspath(
        MODEL_PATH
    )
)

print("\nPredictions saved:")
print(
    os.path.abspath(
        PREDICTION_PATH
    )
)

print("\nMetrics saved:")
print(
    os.path.abspath(
        METRICS_PATH
    )
)