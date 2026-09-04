from pathlib import Path

import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "graph_features_analysis.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

# Only labelled transactions
df = df[df["label"].isin([0, 1])].copy()

y = df["label"].values


# ============================================================
# FEATURES
# ============================================================

features = [
    "degree",
    "in_degree",
    "out_degree",
    "neighbor_count",
    "two_hop_neighbor_count"
]


# ============================================================
# EVALUATE
# ============================================================

results = []


for feature in features:

    values = df[feature].values

    # Fraud has LOWER values, so invert the feature.
    fraud_score = -values

    pr_auc = average_precision_score(
        y,
        fraud_score
    )

    roc_auc = roc_auc_score(
        y,
        fraud_score
    )

    results.append(
        {
            "Feature": feature,
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "PR-AUC",
    ascending=False
)


print("=" * 65)
print("GRAPH FEATURE PREDICTIVE POWER")
print("=" * 65)

print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "graph_feature_predictive_power.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print()
print("Saved:")
print(output_path)