import pickle
import numpy as np


GRAPH_PATH = "data/processed/graph_data.pkl"


print("=" * 60)
print("LOADING SAVED GRAPH")
print("=" * 60)

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)


nodes = graph["nodes"]
edge_index = graph["edge_index"]

train_mask = graph["train_mask"]
val_mask = graph["val_mask"]
test_mask = graph["test_mask"]


print("\nNode table:")
print(nodes.shape)

print("\nEdge index:")
print(edge_index.shape)

print("\nMasks:")
print("Train:", train_mask.shape)
print("Validation:", val_mask.shape)
print("Test:", test_mask.shape)


# ============================================================
# BASIC GRAPH CHECKS
# ============================================================

print("\n" + "=" * 60)
print("GRAPH SANITY CHECKS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Node count
# ------------------------------------------------------------

assert len(nodes) == 203769

print("\n✓ Correct number of nodes")


# ------------------------------------------------------------
# 2. Edge count
# ------------------------------------------------------------

assert edge_index.shape[0] == 2

print("✓ Edge index has correct shape")


# ------------------------------------------------------------
# 3. No invalid node indices
# ------------------------------------------------------------

assert edge_index.min() >= 0
assert edge_index.max() < len(nodes)

print("✓ All edge indices are valid")


# ------------------------------------------------------------
# 4. Masks don't overlap
# ------------------------------------------------------------

assert not np.any(train_mask & val_mask)
assert not np.any(train_mask & test_mask)
assert not np.any(val_mask & test_mask)

print("✓ Train/validation/test masks do not overlap")


# ------------------------------------------------------------
# 5. Masks cover all nodes
# ------------------------------------------------------------

covered = (
    train_mask
    | val_mask
    | test_mask
)

assert covered.sum() == len(nodes)

print("✓ Every node belongs to exactly one split")


# ------------------------------------------------------------
# 6. Check labels
# ------------------------------------------------------------

print("\nLabel distribution:")

print(
    nodes["label"].value_counts(
        dropna=False
    )
)


# ------------------------------------------------------------
# 7. Check unknown labels
# ------------------------------------------------------------

unknown_count = (
    nodes["label"] == -1
).sum()

print(
    "\nUnknown nodes:",
    unknown_count
)


# ------------------------------------------------------------
# 8. Check timestep ranges
# ------------------------------------------------------------

print("\nTimestep ranges:")

print(
    "Train:",
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
    "Validation:",
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
    "Test:",
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


print("\n" + "=" * 60)
print("GRAPH VERIFICATION PASSED")
print("=" * 60)