# ============================================================
# STEP 8 — FINAL MODEL SELECTION + SANITY CHECKS
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_PREDICTIONS = (
    RESULTS_DIR
    / "final_test_predictions.csv"
)

THRESHOLD_RESULTS = (
    RESULTS_DIR
    / "threshold_optimization.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "final_model_selection_analysis.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

XGB_WEIGHT = 0.90
GNN_WEIGHT = 0.10

HYBRID_THRESHOLD = 0.79


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 8 — FINAL MODEL SELECTION + SANITY CHECKS")
print("=" * 70)


# ============================================================
# LOAD FINAL TEST PREDICTIONS
# ============================================================

print("\nLoading final test predictions...")

df = pd.read_csv(
    FINAL_PREDICTIONS
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

# Expected columns from Step 7:
#
# txId
# timestep
# label
# xgboost_probability
# graphsage_probability
# hybrid_probability
# ...


required_columns = [
    "txId",
    "timestep",
    "label",
    "xgboost_probability",
    "graphsage_probability"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CREATE HYBRID SCORE
# ============================================================

df["hybrid_probability"] = (
    XGB_WEIGHT
    * df["xgboost_probability"]
    +
    GNN_WEIGHT
    * df["graphsage_probability"]
)


# ============================================================
# BASIC SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("8.1 BASIC SANITY CHECK")
print("=" * 70)

print(
    "Total transactions:",
    len(df)
)

print(
    "Unique transactions:",
    df["txId"].nunique()
)

if df["txId"].nunique() != len(df):

    print(
        "WARNING: Duplicate transaction IDs detected!"
    )

else:

    print(
        "✓ No duplicate transaction IDs"
    )


print(
    "\nGround-truth distribution:"
)

print(
    df["label"].value_counts()
)


# ============================================================
# CHECK PROBABILITY RANGES
# ============================================================

for column in [
    "xgboost_probability",
    "graphsage_probability",
    "hybrid_probability"
]:

    minimum = df[column].min()
    maximum = df[column].max()

    print(
        f"\n{column}"
    )

    print(
        "Min:",
        minimum
    )

    print(
        "Max:",
        maximum
    )

    if minimum < 0 or maximum > 1:

        print(
            "WARNING: Probability outside [0,1]"
        )

    else:

        print(
            "✓ Valid probability range"
        )


# ============================================================
# MODEL DISAGREEMENT
# ============================================================

print("\n" + "=" * 70)
print("8.2 MODEL DISAGREEMENT ANALYSIS")
print("=" * 70)


# Use 0.5 only for disagreement analysis.
# This is NOT the final production threshold.

df["xgb_high"] = (
    df["xgboost_probability"] >= 0.5
)

df["gnn_high"] = (
    df["graphsage_probability"] >= 0.5
)


both_high = (
    df["xgb_high"]
    & df["gnn_high"]
)

both_low = (
    ~df["xgb_high"]
    & ~df["gnn_high"]
)

xgb_high_gnn_low = (
    df["xgb_high"]
    & ~df["gnn_high"]
)

xgb_low_gnn_high = (
    ~df["xgb_high"]
    & df["gnn_high"]
)


print(
    "\nBoth LOW:",
    both_low.sum(),
    f"({both_low.mean():.2%})"
)

print(
    "Both HIGH:",
    both_high.sum(),
    f"({both_high.mean():.2%})"
)

print(
    "XGB HIGH / GNN LOW:",
    xgb_high_gnn_low.sum(),
    f"({xgb_high_gnn_low.mean():.2%})"
)

print(
    "XGB LOW / GNN HIGH:",
    xgb_low_gnn_high.sum(),
    f"({xgb_low_gnn_high.mean():.2%})"
)


# ============================================================
# AGREEMENT ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("8.3 DISAGREEMENT QUALITY")
print("=" * 70)


def analyze_group(
    group,
    name
):

    subset = df[group]

    if len(subset) == 0:

        print(
            f"\n{name}: no transactions"
        )

        return

    fraud_count = subset["label"].sum()

    fraud_rate = (
        fraud_count
        / len(subset)
    )

    print(
        f"\n{name}"
    )

    print(
        "Transactions:",
        len(subset)
    )

    print(
        "Fraud:",
        int(fraud_count)
    )

    print(
        "Fraud rate:",
        f"{fraud_rate:.2%}"
    )

    print(
        "Average XGB risk:",
        f"{subset['xgboost_probability'].mean():.4f}"
    )

    print(
        "Average GNN risk:",
        f"{subset['graphsage_probability'].mean():.4f}"
    )


analyze_group(
    both_low,
    "BOTH LOW"
)

analyze_group(
    both_high,
    "BOTH HIGH"
)

analyze_group(
    xgb_high_gnn_low,
    "XGBoost HIGH / GraphSAGE LOW"
)

analyze_group(
    xgb_low_gnn_high,
    "XGBoost LOW / GraphSAGE HIGH"
)


# ============================================================
# HYBRID IMPACT
# ============================================================

print("\n" + "=" * 70)
print("8.4 HYBRID CONTRIBUTION")
print("=" * 70)


df["hybrid_minus_xgb"] = (
    df["hybrid_probability"]
    - df["xgboost_probability"]
)

print(
    "\nAverage change:",
    f"{df['hybrid_minus_xgb'].mean():.6f}"
)

print(
    "Average absolute change:",
    f"{df['hybrid_minus_xgb'].abs().mean():.6f}"
)

print(
    "Maximum absolute change:",
    f"{df['hybrid_minus_xgb'].abs().max():.6f}"
)

print(
    "Transactions changed by > 0.01:",
    (
        df["hybrid_minus_xgb"].abs() > 0.01
    ).sum()
)

print(
    "Transactions changed by > 0.05:",
    (
        df["hybrid_minus_xgb"].abs() > 0.05
    ).sum()
)


# ============================================================
# TOP-K RANKING CHANGES
# ============================================================

print("\n" + "=" * 70)
print("8.5 TOP-K RANKING IMPACT")
print("=" * 70)


for k in [100, 500, 1000]:

    xgb_top = set(
        df.nlargest(
            k,
            "xgboost_probability"
        )["txId"]
    )

    hybrid_top = set(
        df.nlargest(
            k,
            "hybrid_probability"
        )["txId"]
    )

    overlap = (
        len(xgb_top & hybrid_top)
        / k
    )

    print(
        f"Top-{k} ranking overlap: "
        f"{overlap:.2%}"
    )


# ============================================================
# HYBRID THRESHOLD
# ============================================================

df["hybrid_prediction"] = (
    df["hybrid_probability"]
    >= HYBRID_THRESHOLD
)


print("\n" + "=" * 70)
print("8.6 FINAL HYBRID THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    HYBRID_THRESHOLD
)

print(
    "Transactions classified HIGH:",
    df["hybrid_prediction"].sum()
)

print(
    "Actual fraud among HIGH:",
    df.loc[
        df["hybrid_prediction"],
        "label"
    ].sum()
)


# ============================================================
# FINAL MODEL DECISION
# ============================================================

print("\n" + "=" * 70)
print("8.7 FINAL MODEL DECISION")
print("=" * 70)

print(
    """
PRIMARY FRAUD RANKING MODEL
---------------------------
XGBoost

REASON
------
XGBoost achieved the strongest standalone
validation and test ranking performance.

GraphSAGE is therefore NOT replacing XGBoost.


GRAPH MODEL
-----------
GraphSAGE

ROLE
----
Supporting graph-based signal.

It captures transaction relationships
that feature-based XGBoost does not explicitly model.


FINAL RISK SCORE
----------------
90% XGBoost
10% GraphSAGE


FINAL HYBRID THRESHOLD
----------------------
0.79


ARCHITECTURE
------------
Transaction
      |
      +----> XGBoost --------+
      |                      |
      +----> GraphSAGE ------+----> Hybrid Risk Score
                             |
                             v
                       Risk Decision
"""
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "xgb_weight": [XGB_WEIGHT],

    "graphsage_weight": [GNN_WEIGHT],

    "hybrid_threshold": [
        HYBRID_THRESHOLD
    ],

    "transactions": [
        len(df)
    ],

    "fraud_cases": [
        int(df["label"].sum())
    ],

    "both_low": [
        int(both_low.sum())
    ],

    "both_high": [
        int(both_high.sum())
    ],

    "xgb_high_gnn_low": [
        int(xgb_high_gnn_low.sum())
    ],

    "xgb_low_gnn_high": [
        int(xgb_low_gnn_high.sum())
    ],

    "average_hybrid_change": [
        df["hybrid_minus_xgb"].mean()
    ],

    "average_absolute_hybrid_change": [
        df["hybrid_minus_xgb"].abs().mean()
    ],

    "top100_xgb_hybrid_overlap": [
        len(
            set(
                df.nlargest(
                    100,
                    "xgboost_probability"
                )["txId"]
            )
            &
            set(
                df.nlargest(
                    100,
                    "hybrid_probability"
                )["txId"]
            )
        ) / 100
    ],

    "top500_xgb_hybrid_overlap": [
        len(
            set(
                df.nlargest(
                    500,
                    "xgboost_probability"
                )["txId"]
            )
            &
            set(
                df.nlargest(
                    500,
                    "hybrid_probability"
                )["txId"]
            )
        ) / 500
    ],

    "top1000_xgb_hybrid_overlap": [
        len(
            set(
                df.nlargest(
                    1000,
                    "xgboost_probability"
                )["txId"]
            )
            &
            set(
                df.nlargest(
                    1000,
                    "hybrid_probability"
                )["txId"]
            )
        ) / 1000
    ]

})


summary.to_csv(
    OUTPUT_PATH,
    index=False
)


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
    "STEP 8 COMPLETE"
)

print(
    "=" * 70
)