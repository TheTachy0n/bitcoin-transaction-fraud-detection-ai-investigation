# ============================================================
# STEP 7C
# LEAKAGE-SAFE HISTORICAL GRAPH FEATURES
# ============================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_data.pkl"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

XGB_TRAIN_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost_test_predictions.csv"
)

XGB_VAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "xgboost_validation_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "historical_graph_features.csv"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("LEAKAGE-SAFE HISTORICAL GRAPH FEATURES")
print("=" * 70)

print("\nLoading graph...")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)

nodes = graph["nodes"]
edge_index = np.asarray(graph["edge_index"])

print("Nodes:", len(nodes))
print("Edges:", edge_index.shape[1])


# ============================================================
# LOAD SPLITS
# ============================================================

print("\nLoading temporal splits...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# DETERMINE TIMESTEP COLUMN
# ============================================================

# graph_data contains timestep information.
node_timestep = dict(
    zip(
        nodes["txId"],
        nodes["timestep"]
    )
)


# ============================================================
# DETERMINE LABELS
# ============================================================

node_label = dict(
    zip(
        nodes["txId"],
        nodes["label"]
    )
)


# ============================================================
# XGBOOST HISTORICAL PREDICTIONS
# ============================================================

# We use previously generated XGBoost predictions
# only as historical risk information.
#
# IMPORTANT:
# For a transaction at timestep T, we will only use
# predictions from timesteps < T.

historical_risk = {}


# Training predictions are not currently available from
# the inference files, so initialize the dictionary with
# validation/test-safe information when available.

if XGB_TRAIN_PATH.exists():

    train_xgb = pd.read_csv(
        XGB_TRAIN_PATH
    )

    if "xgboost_probability" in train_xgb.columns:

        for _, row in train_xgb.iterrows():

            historical_risk[
                int(row["txId"])
            ] = float(
                row["xgboost_probability"]
            )


if XGB_VAL_PATH.exists():

    val_xgb = pd.read_csv(
        XGB_VAL_PATH
    )

    if "xgboost_probability" in val_xgb.columns:

        for _, row in val_xgb.iterrows():

            historical_risk[
                int(row["txId"])
            ] = float(
                row["xgboost_probability"]
            )


print(
    "\nHistorical XGBoost predictions loaded:",
    len(historical_risk)
)


# ============================================================
# BUILD ADJACENCY LIST
# ============================================================

print("\nBuilding adjacency list...")

num_nodes = len(nodes)

neighbors = [
    set()
    for _ in range(num_nodes)
]


for source, target in zip(
    edge_index[0],
    edge_index[1]
):

    source = int(source)
    target = int(target)

    neighbors[source].add(target)
    neighbors[target].add(source)


# Map txId → node index

txid_to_index = dict(
    zip(
        nodes["txId"],
        range(num_nodes)
    )
)


# ============================================================
# SORT NODES TEMPORALLY
# ============================================================

print("Sorting nodes by timestep...")

node_records = nodes[
    [
        "txId",
        "timestep",
        "label"
    ]
].copy()

node_records = node_records.sort_values(
    "timestep"
)


# ============================================================
# HISTORICAL INFORMATION STORAGE
# ============================================================

# These dictionaries represent information that has become
# available before the current timestep.

known_labels = {}
known_risks = {}


# ============================================================
# FEATURE GENERATION
# ============================================================

results = []


print("\nGenerating historical graph features...")


for timestep, timestep_group in node_records.groupby(
    "timestep",
    sort=True
):

    print(
        f"Processing timestep {timestep}..."
    )

    # --------------------------------------------------------
    # FIRST:
    # Generate features using information from EARLIER
    # timesteps only.
    # --------------------------------------------------------

    for _, row in timestep_group.iterrows():

        txid = int(row["txId"])

        node_idx = txid_to_index[txid]

        current_neighbors = neighbors[
            node_idx
        ]

        historical_neighbors = []

        for neighbor_idx in current_neighbors:

            neighbor_txid = int(
                nodes.iloc[
                    neighbor_idx
                ]["txId"]
            )

            neighbor_timestep = int(
                nodes.iloc[
                    neighbor_idx
                ]["timestep"]
            )

            # STRICT temporal protection
            if neighbor_timestep < timestep:

                historical_neighbors.append(
                    neighbor_txid
                )


        # ----------------------------------------------------
        # HISTORICAL NEIGHBOR COUNT
        # ----------------------------------------------------

        historical_neighbor_count = len(
            historical_neighbors
        )


        # ----------------------------------------------------
        # HISTORICAL FRAUD COUNT
        # ----------------------------------------------------

        fraud_labels = []

        for neighbor_txid in historical_neighbors:

            if neighbor_txid in known_labels:

                label = known_labels[
                    neighbor_txid
                ]

                if label in [0, 1]:

                    fraud_labels.append(
                        label
                    )


        historical_fraud_count = sum(
            label == 1
            for label in fraud_labels
        )


        # ----------------------------------------------------
        # HISTORICAL FRAUD RATE
        # ----------------------------------------------------

        if len(fraud_labels) > 0:

            historical_fraud_rate = (
                historical_fraud_count
                / len(fraud_labels)
            )

        else:

            historical_fraud_rate = 0.0


        # ----------------------------------------------------
        # HISTORICAL XGBOOST RISK
        # ----------------------------------------------------

        neighbor_risks = []

        for neighbor_txid in historical_neighbors:

            if neighbor_txid in known_risks:

                neighbor_risks.append(
                    known_risks[
                        neighbor_txid
                    ]
                )


        if len(neighbor_risks) > 0:

            mean_neighbor_xgb_risk = float(
                np.mean(
                    neighbor_risks
                )
            )

            max_neighbor_xgb_risk = float(
                np.max(
                    neighbor_risks
                )
            )

        else:

            mean_neighbor_xgb_risk = 0.0
            max_neighbor_xgb_risk = 0.0


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        results.append(
            {
                "txId": txid,

                "timestep": timestep,

                "label": row["label"],

                "historical_neighbor_count":
                    historical_neighbor_count,

                "historical_fraud_count":
                    historical_fraud_count,

                "historical_fraud_rate":
                    historical_fraud_rate,

                "mean_neighbor_xgb_risk":
                    mean_neighbor_xgb_risk,

                "max_neighbor_xgb_risk":
                    max_neighbor_xgb_risk
            }
        )


    # ========================================================
    # SECOND:
    # AFTER prediction features are generated for this
    # timestep, make this timestep's labelled information
    # available to FUTURE timesteps.
    # ========================================================

    for _, row in timestep_group.iterrows():

        txid = int(row["txId"])

        label = row["label"]

        if label in [0, 1]:

            known_labels[
                txid
            ] = int(label)


        if txid in historical_risk:

            known_risks[
                txid
            ] = historical_risk[
                txid
            ]


# ============================================================
# CREATE DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved:"
)

print(OUTPUT_PATH)


# ============================================================
# SUMMARY
# ============================================================

print("\nFeature summary:")

print(
    features_df[
        [
            "historical_neighbor_count",
            "historical_fraud_count",
            "historical_fraud_rate",
            "mean_neighbor_xgb_risk",
            "max_neighbor_xgb_risk"
        ]
    ].describe()
)


# ============================================================
# CHECK TEMPORAL COVERAGE
# ============================================================

print(
    "\nFeatures by timestep:"
)

print(
    features_df.groupby(
        "timestep"
    )[
        [
            "historical_neighbor_count",
            "historical_fraud_count",
            "historical_fraud_rate",
            "mean_neighbor_xgb_risk"
        ]
    ].mean().head(15)
)


print(
    "\nHistorical graph feature generation complete."
)