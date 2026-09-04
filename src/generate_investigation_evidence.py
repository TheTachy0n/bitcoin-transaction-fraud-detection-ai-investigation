# ============================================================
# STEP 10 — INVESTIGATION EVIDENCE GENERATOR
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_ENGINE_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_risk_engine.csv"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_data.pkl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "investigation_evidence.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Transactions at or above this score are sent for investigation.
RISK_THRESHOLD = 0.79

# Only store this many neighbor IDs in the CSV.
# Statistics are calculated using ALL neighbors.
TOP_NEIGHBORS = 10


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STEP 10 — INVESTIGATION EVIDENCE GENERATOR")
print("=" * 70)


# ============================================================
# LOAD RISK ENGINE
# ============================================================

print("\nLoading final risk engine...")

risk_df = pd.read_csv(
    RISK_ENGINE_PATH
)

print(
    "Rows:",
    len(risk_df)
)

print(
    "Columns:",
    list(risk_df.columns)
)


# ============================================================
# BUILD RISK LOOKUP
# ============================================================
#
# This allows us to obtain the MODEL-PREDICTED risk of any
# neighboring transaction.
#
# IMPORTANT:
# We deliberately use predicted risk rather than the actual
# fraud label. The actual label must not be used as evidence
# during investigation.
# ============================================================

tx_to_risk = {
    int(row["txId"]): float(row["risk_score"])
    for _, row in risk_df.iterrows()
}

print(
    "Risk lookup entries:",
    len(tx_to_risk)
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph data...")

with open(
    GRAPH_PATH,
    "rb"
) as f:

    graph = pickle.load(f)


nodes = graph["nodes"]

edge_index = graph["edge_index"]


print(
    "Graph nodes:",
    len(nodes)
)

print(
    "Graph edges:",
    edge_index.shape[1]
)


# ============================================================
# BUILD TRANSACTION LOOKUP
# ============================================================

nodes = nodes.copy()

nodes["txId"] = nodes["txId"].astype(int)

tx_to_idx = {
    int(tx_id): idx
    for idx, tx_id in enumerate(
        nodes["txId"].values
    )
}

idx_to_tx = {
    idx: int(tx_id)
    for idx, tx_id in enumerate(
        nodes["txId"].values
    )
}


# ============================================================
# BUILD NEIGHBOR MAP
# ============================================================

print("\nBuilding graph neighborhood map...")

neighbor_map = {}

for source, target in zip(
    edge_index[0],
    edge_index[1]
):

    source = int(source)
    target = int(target)

    if source not in neighbor_map:

        neighbor_map[source] = []

    neighbor_map[source].append(
        target
    )


print(
    "Nodes with graph neighbors:",
    len(neighbor_map)
)


# ============================================================
# SELECT HIGH-RISK TRANSACTIONS
# ============================================================

print("\nSelecting transactions for investigation...")

investigation_df = risk_df[
    risk_df["risk_score"] >= RISK_THRESHOLD
].copy()


investigation_df = investigation_df.sort_values(
    "risk_score",
    ascending=False
)


print(
    "Transactions selected:",
    len(investigation_df)
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in nodes.columns
    if column.startswith("feature_")
]

print(
    "Transaction features:",
    len(feature_columns)
)


# ============================================================
# GENERATE EVIDENCE
# ============================================================

evidence_records = []


for _, row in investigation_df.iterrows():

    tx_id = int(
        row["txId"]
    )

    # --------------------------------------------------------
    # Basic model information
    # --------------------------------------------------------

    xgb_probability = float(
        row["xgboost_probability"]
    )

    graphsage_probability = float(
        row["graphsage_probability"]
    )

    risk_score = float(
        row["risk_score"]
    )

    risk_level = row[
        "risk_level"
    ]

    agreement_category = row[
        "agreement_category"
    ]

    evidence_type = row[
        "evidence_type"
    ]

    alert_priority = row[
        "alert_priority"
    ]

    timestep = int(
        row["timestep"]
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------
    #
    # Kept ONLY for later evaluation.
    #
    # It is NOT used to calculate investigation evidence.
    # --------------------------------------------------------

    actual_label = int(
        row["label"]
    )


    # --------------------------------------------------------
    # Locate transaction in graph
    # --------------------------------------------------------

    node_idx = tx_to_idx.get(
        tx_id
    )


    if node_idx is None:

        print(
            f"Warning: transaction {tx_id} "
            f"not found in graph"
        )

        continue


    # --------------------------------------------------------
    # Transaction features
    # --------------------------------------------------------

    transaction_node = nodes.iloc[
        node_idx
    ]


    feature_values = {}

    for feature in feature_columns:

        value = transaction_node[
            feature
        ]

        try:

            feature_values[feature] = float(
                value
            )

        except (TypeError, ValueError):

            feature_values[feature] = None


    # --------------------------------------------------------
    # Graph neighbors
    # --------------------------------------------------------

    all_neighbor_indices = neighbor_map.get(
        node_idx,
        []
    )


    # Remove duplicate neighbors
    all_neighbor_indices = list(
        dict.fromkeys(
            all_neighbor_indices
        )
    )


    # --------------------------------------------------------
    # Collect neighbor information
    # --------------------------------------------------------

    neighbor_ids_all = []

    neighbor_risks = []


    for neighbor_idx in all_neighbor_indices:

        neighbor_tx_id = idx_to_tx.get(
            neighbor_idx
        )

        if neighbor_tx_id is None:

            continue


        neighbor_ids_all.append(
            neighbor_tx_id
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Use predicted risk, NOT actual fraud label.
        # ----------------------------------------------------

        neighbor_risk = tx_to_risk.get(
            neighbor_tx_id
        )

        if neighbor_risk is not None:

            neighbor_risks.append(
                neighbor_risk
            )


    # --------------------------------------------------------
    # Total neighbors
    # --------------------------------------------------------

    neighbor_count = len(
        neighbor_ids_all
    )


    # --------------------------------------------------------
    # Only retain a limited number of IDs for the CSV.
    #
    # Statistics below still use ALL neighbors.
    # --------------------------------------------------------

    neighbor_ids = neighbor_ids_all[
        :TOP_NEIGHBORS
    ]


    # --------------------------------------------------------
    # Neighbor risk statistics
    # --------------------------------------------------------

    risk_count = len(
        neighbor_risks
    )


    if risk_count > 0:

        neighbor_avg_risk = float(
            np.mean(
                neighbor_risks
            )
        )

        neighbor_max_risk = float(
            np.max(
                neighbor_risks
            )
        )

        high_risk_neighbor_count = sum(
            risk >= RISK_THRESHOLD
            for risk in neighbor_risks
        )

        neighbor_high_risk_rate = (
            high_risk_neighbor_count
            / risk_count
        )

    else:

        neighbor_avg_risk = 0.0

        neighbor_max_risk = 0.0

        high_risk_neighbor_count = 0

        neighbor_high_risk_rate = 0.0


    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    model_difference = abs(
        xgb_probability
        - graphsage_probability
    )


    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    if agreement_category == "BOTH_HIGH":

        evidence_summary = (
            "Both XGBoost and GraphSAGE "
            "assign high fraud risk, providing "
            "strong corroborating model evidence."
        )

    elif agreement_category == "XGB_HIGH_GNN_LOW":

        evidence_summary = (
            "XGBoost identifies strong "
            "transaction-level fraud risk, "
            "while GraphSAGE provides weaker "
            "graph-based evidence."
        )

    elif agreement_category == "XGB_LOW_GNN_HIGH":

        evidence_summary = (
            "GraphSAGE identifies strong "
            "graph-based risk despite a lower "
            "XGBoost transaction-level risk."
        )

    else:

        evidence_summary = (
            "The models show relatively low "
            "fraud risk based on their predictions."
        )


    # --------------------------------------------------------
    # Create record
    # --------------------------------------------------------

    record = {

        # -----------------------------------------------
        # Transaction
        # -----------------------------------------------

        "txId":
            tx_id,

        "timestep":
            timestep,

        # Ground truth retained for evaluation only
        "actual_label":
            actual_label,


        # -----------------------------------------------
        # Risk
        # -----------------------------------------------

        "xgboost_probability":
            xgb_probability,

        "graphsage_probability":
            graphsage_probability,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "alert_priority":
            alert_priority,


        # -----------------------------------------------
        # Model agreement
        # -----------------------------------------------

        "agreement_category":
            agreement_category,

        "evidence_type":
            evidence_type,

        "model_probability_difference":
            model_difference,


        # -----------------------------------------------
        # Graph evidence
        # -----------------------------------------------

        "neighbor_count":
            neighbor_count,

        "high_risk_neighbor_count":
            high_risk_neighbor_count,

        "neighbor_avg_risk":
            neighbor_avg_risk,

        "neighbor_max_risk":
            neighbor_max_risk,

        "neighbor_high_risk_rate":
            neighbor_high_risk_rate,

        "neighbor_tx_ids":
            ",".join(
                map(
                    str,
                    neighbor_ids
                )
            ),


        # -----------------------------------------------
        # Explanation
        # -----------------------------------------------

        "evidence_summary":
            evidence_summary
    }


    # --------------------------------------------------------
    # Add transaction features
    # --------------------------------------------------------

    for feature, value in feature_values.items():

        record[
            feature
        ] = value


    evidence_records.append(
        record
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

evidence_df = pd.DataFrame(
    evidence_records
)


# ============================================================
# SORT BY RISK
# ============================================================

if len(evidence_df) > 0:

    evidence_df = evidence_df.sort_values(
        "risk_score",
        ascending=False
    )


# ============================================================
# SAVE
# ============================================================

evidence_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INVESTIGATION EVIDENCE SUMMARY")
print("=" * 70)


print(
    "\nEvidence records:",
    len(evidence_df)
)


if len(evidence_df) > 0:

    print(
        "\nRisk distribution:"
    )

    print(
        evidence_df[
            "risk_level"
        ].value_counts()
    )


    print(
        "\nAlert priority:"
    )

    print(
        evidence_df[
            "alert_priority"
        ].value_counts()
    )


    print(
        "\nEvidence type:"
    )

    print(
        evidence_df[
            "evidence_type"
        ].value_counts()
    )


    print(
        "\nAgreement category:"
    )

    print(
        evidence_df[
            "agreement_category"
        ].value_counts()
    )


    print(
        "\nAverage neighbor count:",
        round(
            evidence_df[
                "neighbor_count"
            ].mean(),
            2
        )
    )


    print(
        "Average neighbor risk:",
        round(
            evidence_df[
                "neighbor_avg_risk"
            ].mean(),
            4
        )
    )


    print(
        "Average high-risk neighbor rate:",
        round(
            evidence_df[
                "neighbor_high_risk_rate"
            ].mean(),
            4
        )
    )


# ============================================================
# TOP TRANSACTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 10 INVESTIGATION CASES"
)

print(
    "=" * 70
)


display_columns = [

    "txId",

    "timestep",

    "risk_score",

    "risk_level",

    "alert_priority",

    "agreement_category",

    "evidence_type",

    "xgboost_probability",

    "graphsage_probability",

    "neighbor_count",

    "high_risk_neighbor_count",

    "neighbor_avg_risk",

    "neighbor_high_risk_rate",

    "evidence_summary"
]


if len(evidence_df) > 0:

    print(
        evidence_df[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\n" + "=" * 70
)

print(
    "STEP 10 COMPLETE"
)

print(
    "=" * 70
)