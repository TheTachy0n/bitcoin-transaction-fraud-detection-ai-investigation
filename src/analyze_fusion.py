# ============================================================
# XGBOOST + GRAPHSAGE FUSION ANALYSIS
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "aligned_validation_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("XGBOOST + GRAPHSAGE FUSION ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(
    "\nTransactions:",
    len(df)
)


# ============================================================
# CORRELATION
# ============================================================

correlation = df[
    [
        "xgboost_probability",
        "graphsage_probability"
    ]
].corr().iloc[0, 1]

print(
    "\nPrediction correlation:",
    correlation
)


# ============================================================
# PREDICTION DISAGREEMENT
# ============================================================

df["probability_difference"] = (
    df["xgboost_probability"]
    - df["graphsage_probability"]
).abs()

print(
    "\nProbability difference statistics:"
)

print(
    df["probability_difference"].describe()
)


# ============================================================
# STRONG DISAGREEMENTS
# ============================================================

print(
    "\nTop 20 strongest disagreements:"
)

disagreements = (
    df.sort_values(
        "probability_difference",
        ascending=False
    )
    .head(20)
)

print(
    disagreements[
        [
            "txId",
            "timestep",
            "label",
            "xgboost_probability",
            "graphsage_probability",
            "probability_difference"
        ]
    ].to_string(index=False)
)


# ============================================================
# HIGH-RISK DISAGREEMENTS
# ============================================================

print(
    "\nCases where XGBoost is LOW but GraphSAGE is HIGH:"
)

xgb_low_gnn_high = df[
    (df["xgboost_probability"] < 0.2)
    &
    (df["graphsage_probability"] > 0.8)
]

print(
    "Count:",
    len(xgb_low_gnn_high)
)

print(
    xgb_low_gnn_high[
        [
            "txId",
            "timestep",
            "label",
            "xgboost_probability",
            "graphsage_probability"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# REVERSE DISAGREEMENT
# ============================================================

print(
    "\nCases where XGBoost is HIGH but GraphSAGE is LOW:"
)

gnn_low_xgb_high = df[
    (df["xgboost_probability"] > 0.8)
    &
    (df["graphsage_probability"] < 0.2)
]

print(
    "Count:",
    len(gnn_low_xgb_high)
)

print(
    gnn_low_xgb_high[
        [
            "txId",
            "timestep",
            "label",
            "xgboost_probability",
            "graphsage_probability"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# DISAGREEMENT QUALITY
# ============================================================

print(
    "\nLabel distribution of XGB-low / GNN-high:"
)

if len(xgb_low_gnn_high) > 0:

    print(
        xgb_low_gnn_high["label"]
        .value_counts(
            normalize=True
        )
    )


print(
    "\nLabel distribution of XGB-high / GNN-low:"
)

if len(gnn_low_xgb_high) > 0:

    print(
        gnn_low_xgb_high["label"]
        .value_counts(
            normalize=True
        )
    )


print(
    "\nAnalysis complete."
)