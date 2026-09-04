# ============================================================
# GRAPH FEATURE ANALYSIS
# Elliptic Bitcoin Fraud Detection
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "graph_features_analysis.csv"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 60)
print("GRAPH FEATURE ANALYSIS")
print("=" * 60)

print("\nLoading graph...")

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)


nodes = graph["nodes"]
edge_index = np.asarray(graph["edge_index"])


print(
    "Nodes:",
    len(nodes)
)

print(
    "Edge index shape:",
    edge_index.shape
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nNode columns:")

print(
    nodes.columns.tolist()
)


# ============================================================
# EXTRACT EDGES
# ============================================================

sources = edge_index[0]
targets = edge_index[1]

num_nodes = len(nodes)


# ============================================================
# DEGREE FEATURES
# ============================================================

print("\nCalculating degree features...")


# Total degree
degree = np.bincount(
    np.concatenate(
        [sources, targets]
    ),
    minlength=num_nodes
)


# Out-degree
out_degree = np.bincount(
    sources,
    minlength=num_nodes
)


# In-degree
in_degree = np.bincount(
    targets,
    minlength=num_nodes
)


# ============================================================
# UNIQUE NEIGHBORS
# ============================================================

print("Calculating unique neighbor counts...")


neighbors = [
    set()
    for _ in range(num_nodes)
]


for source, target in zip(
    sources,
    targets
):

    neighbors[source].add(target)
    neighbors[target].add(source)


neighbor_count = np.array(
    [
        len(n)
        for n in neighbors
    ],
    dtype=np.int32
)


# ============================================================
# 2-HOP NEIGHBORHOOD
# ============================================================

print(
    "Calculating 2-hop neighborhood sizes..."
)

two_hop_size = np.zeros(
    num_nodes,
    dtype=np.int32
)


for node in range(num_nodes):

    first_hop = neighbors[node]

    second_hop = set()

    for neighbor in first_hop:

        second_hop.update(
            neighbors[neighbor]
        )

    second_hop.discard(node)

    two_hop_size[node] = len(
        second_hop
    )


# ============================================================
# BUILD FEATURE DATAFRAME
# ============================================================

graph_features = pd.DataFrame(
    {
        "txId": nodes["txId"].values,

        "timestep": nodes[
            "timestep"
        ].values,

        "label": nodes[
            "label"
        ].values,

        "degree": degree,

        "in_degree": in_degree,

        "out_degree": out_degree,

        "neighbor_count": neighbor_count,

        "two_hop_neighbor_count":
            two_hop_size
    }
)


# ============================================================
# SAVE
# ============================================================

graph_features.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved graph features:"
)

print(OUTPUT_PATH)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\nFeature statistics:")

print(
    graph_features[
        [
            "degree",
            "in_degree",
            "out_degree",
            "neighbor_count",
            "two_hop_neighbor_count"
        ]
    ].describe()
)


# ============================================================
# LABELLED DATA ONLY
# ============================================================

labelled = graph_features[
    graph_features["label"].isin(
        [0, 1]
    )
].copy()


print(
    "\nLabelled transactions:",
    len(labelled)
)


# ============================================================
# COMPARE LICIT / ILLICIT
# ============================================================

print(
    "\nGraph feature means by class:"
)

class_summary = (
    labelled
    .groupby("label")[
        [
            "degree",
            "in_degree",
            "out_degree",
            "neighbor_count",
            "two_hop_neighbor_count"
        ]
    ]
    .mean()
)


print(class_summary)


# ============================================================
# MEDIANS
# ============================================================

print(
    "\nGraph feature medians by class:"
)

class_medians = (
    labelled
    .groupby("label")[
        [
            "degree",
            "in_degree",
            "out_degree",
            "neighbor_count",
            "two_hop_neighbor_count"
        ]
    ]
    .median()
)


print(class_medians)


# ============================================================
# ZERO-DEGREE ANALYSIS
# ============================================================

print(
    "\nZero-degree transactions:"
)

zero_degree = (
    labelled["degree"] == 0
)

print(
    "Total:",
    zero_degree.sum()
)

print(
    "Fraction:",
    zero_degree.mean()
)


print(
    "\nZero-degree by class:"
)

print(
    pd.crosstab(
        labelled["label"],
        zero_degree,
        normalize="index"
    )
)


print(
    "\nGraph feature analysis complete."
)