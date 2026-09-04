import pandas as pd
import numpy as np

FEATURE_PATH = "data/raw/elliptic_txs_features.csv"
EDGE_PATH = "data/raw/elliptic_txs_edgelist.csv"


print("Loading data...")

features = pd.read_csv(
    FEATURE_PATH,
    header=None
)

edges = pd.read_csv(
    EDGE_PATH
)

features.columns = (
    ["txId", "timestep"]
    + [f"feature_{i}" for i in range(1, 166)]
)

# ------------------------------------------------------------
# Timestep lookup
# ------------------------------------------------------------

timestep_lookup = dict(
    zip(
        features["txId"],
        features["timestep"]
    )
)

# ------------------------------------------------------------
# Map edge endpoints to timesteps
# ------------------------------------------------------------

edges["t1"] = edges["txId1"].map(
    timestep_lookup
)

edges["t2"] = edges["txId2"].map(
    timestep_lookup
)

# Remove edges where a transaction was not found

edges = edges.dropna(
    subset=["t1", "t2"]
)

# ------------------------------------------------------------
# Edge timestep relationship
# ------------------------------------------------------------

edges["same_timestep"] = (
    edges["t1"] == edges["t2"]
)

edges["forward_edge"] = (
    edges["t1"] < edges["t2"]
)

edges["backward_edge"] = (
    edges["t1"] > edges["t2"]
)

print("\n" + "=" * 60)
print("TEMPORAL EDGE ANALYSIS")
print("=" * 60)

print("\nTotal valid edges:")
print(len(edges))

print("\nSame timestep:")
print(
    edges["same_timestep"].sum()
)

print("\nEarlier → later:")
print(
    edges["forward_edge"].sum()
)

print("\nLater → earlier:")
print(
    edges["backward_edge"].sum()
)

print("\nCross-timestep edges:")
print(
    (~edges["same_timestep"]).sum()
)

# ------------------------------------------------------------
# Distribution of timestep gaps
# ------------------------------------------------------------

edges["timestep_gap"] = (
    edges["t2"] - edges["t1"]
).abs()

print("\nTimestep gap distribution:")

print(
    edges["timestep_gap"]
    .value_counts()
    .sort_index()
    .head(20)
)

print("\nMaximum timestep gap:")

print(
    edges["timestep_gap"].max()
)

# ------------------------------------------------------------
# Cross-boundary edges
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SPLIT BOUNDARY ANALYSIS")
print("=" * 60)

boundaries = [34, 41]

for boundary in boundaries:

    crossing = (
        (
            edges["t1"] <= boundary
        )
        &
        (
            edges["t2"] > boundary
        )
    )

    print(
        f"\nBoundary {boundary}:"
    )

    print(
        "Edges crossing into future:",
        crossing.sum()
    )

print("\nAnalysis complete.")