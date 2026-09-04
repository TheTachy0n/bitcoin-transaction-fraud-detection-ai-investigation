# ============================================================
# SELECT REPRESENTATIVE END-TO-END TEST TRANSACTIONS
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
    / "final_risk_engine.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("REPRESENTATIVE TRANSACTION TEST SELECTION")
print("=" * 70)

print("\nLoading risk engine results...")

df = pd.read_csv(INPUT_PATH)

print(f"Total transactions available: {len(df):,}")


# ============================================================
# INSPECT COLUMNS
# ============================================================

print("\nAvailable columns:")

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

df.columns = [
    str(column).strip()
    for column in df.columns
]


# ============================================================
# DISPLAY RISK DISTRIBUTION
# ============================================================

if "risk_level" in df.columns:

    print("\nRisk distribution:")

    print(
        df["risk_level"]
        .value_counts(dropna=False)
    )


# ============================================================
# DISPLAY AGREEMENT DISTRIBUTION
# ============================================================

if "agreement_category" in df.columns:

    print("\nModel agreement distribution:")

    print(
        df["agreement_category"]
        .value_counts(dropna=False)
    )


# ============================================================
# HELPER
# ============================================================

def select_rows(
    dataframe,
    condition,
    count=5
):

    subset = dataframe.loc[
        condition
    ].copy()

    if subset.empty:
        return subset

    # Prefer diverse risk scores rather than
    # taking only the highest-ranked transactions.
    if "risk_score" in subset.columns:

        subset = subset.sort_values(
            "risk_score"
        )

        if len(subset) > count:

            indexes = (
                pd.Series(
                    range(len(subset))
                )
                .sample(
                    n=count,
                    random_state=42
                )
                .sort_values()
                .values
            )

            subset = subset.iloc[
                indexes
            ]

    return subset.head(count)


# ============================================================
# SELECT HIGH
# ============================================================

high = select_rows(
    df,
    df["risk_level"].astype(str).str.upper() == "HIGH",
    5
)


# ============================================================
# SELECT MEDIUM
# ============================================================

medium = select_rows(
    df,
    df["risk_level"].astype(str).str.upper() == "MEDIUM",
    5
)


# ============================================================
# SELECT LOW
# ============================================================

low = select_rows(
    df,
    df["risk_level"].astype(str).str.upper() == "LOW",
    5
)


# ============================================================
# SELECT MODEL DISAGREEMENT
# ============================================================

if "agreement_category" in df.columns:

    disagreement = select_rows(
        df,
        df["agreement_category"]
        .astype(str)
        .str.upper()
        .isin(
            [
                "XGB_HIGH_GNN_LOW",
                "XGB_LOW_GNN_HIGH"
            ]
        ),
        5
    )

else:

    disagreement = pd.DataFrame()


# ============================================================
# PRINT SELECTED TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("SELECTED TEST CASES")
print("=" * 70)


def print_cases(
    name,
    dataframe
):

    print(
        f"\n{name} ({len(dataframe)} cases)"
    )

    if dataframe.empty:

        print(
            "  No transactions found."
        )

        return

    for _, row in dataframe.iterrows():

        tx_id = row.get(
            "txId",
            row.get(
                "transaction_id",
                "UNKNOWN"
            )
        )

        risk = row.get(
            "risk_score",
            "UNKNOWN"
        )

        level = row.get(
            "risk_level",
            "UNKNOWN"
        )

        agreement = row.get(
            "agreement_category",
            "UNKNOWN"
        )

        xgb = row.get(
            "xgboost_probability",
            "UNKNOWN"
        )

        gnn = row.get(
            "graphsage_probability",
            "UNKNOWN"
        )

        print(
            f"  TX={tx_id} | "
            f"risk={risk} | "
            f"level={level} | "
            f"agreement={agreement} | "
            f"XGB={xgb} | "
            f"GNN={gnn}"
        )


print_cases(
    "HIGH RISK",
    high
)

print_cases(
    "MEDIUM RISK",
    medium
)

print_cases(
    "LOW RISK",
    low
)

print_cases(
    "MODEL DISAGREEMENT",
    disagreement
)


# ============================================================
# SAVE TEST SUITE
# ============================================================

test_cases = []

for category, dataframe in [
    ("HIGH", high),
    ("MEDIUM", medium),
    ("LOW", low),
    ("DISAGREEMENT", disagreement),
]:

    for _, row in dataframe.iterrows():

        tx_id = row.get(
            "txId",
            row.get(
                "transaction_id",
                None
            )
        )

        if tx_id is not None:

            test_cases.append(
                {
                    "category": category,
                    "transaction_id": int(tx_id)
                }
            )


output_path = (
    PROJECT_ROOT
    / "results"
    / "representative_test_cases.json"
)


import json

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        test_cases,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    f"Selected {len(test_cases)} representative transactions."
)

print(
    f"Saved to: {output_path}"
)

print(
    "=" * 70
)