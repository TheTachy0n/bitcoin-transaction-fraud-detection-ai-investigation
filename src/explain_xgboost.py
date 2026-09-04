# ============================================================
# STEP 11 — XGBOOST SHAP EXPLAINABILITY
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_best.pkl"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_data.pkl"
)

INVESTIGATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "investigation_evidence.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "shap_explanations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_FEATURES = 5


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STEP 11 — XGBOOST SHAP EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded:",
    MODEL_PATH
)


# ============================================================
# LOAD GRAPH DATA
# ============================================================

print("\nLoading graph data...")

with open(
    GRAPH_PATH,
    "rb"
) as f:

    graph = pickle.load(f)


nodes = graph["nodes"].copy()

nodes["txId"] = nodes["txId"].astype(int)


print(
    "Graph nodes:",
    len(nodes)
)


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

feature_columns = [
    column
    for column in nodes.columns
    if column.startswith("feature_")
]


print(
    "Feature count:",
    len(feature_columns)
)

print(
    "First features:",
    feature_columns[:5]
)

print(
    "Last features:",
    feature_columns[-5:]
)


# ============================================================
# LOAD INVESTIGATION CASES
# ============================================================

print("\nLoading investigation cases...")

investigation_df = pd.read_csv(
    INVESTIGATION_PATH
)

print(
    "Investigation cases:",
    len(investigation_df)
)


# ============================================================
# BUILD TRANSACTION LOOKUP
# ============================================================

tx_to_idx = {
    int(tx_id): idx
    for idx, tx_id in enumerate(
        nodes["txId"].values
    )
}


# ============================================================
# PREPARE HIGH-RISK TRANSACTIONS
# ============================================================

transaction_ids = []

feature_rows = []


for tx_id in investigation_df["txId"]:

    tx_id = int(tx_id)

    node_idx = tx_to_idx.get(
        tx_id
    )

    if node_idx is None:

        print(
            f"Warning: transaction {tx_id} "
            "not found in graph."
        )

        continue


    transaction_node = nodes.iloc[
        node_idx
    ]


    transaction_ids.append(
        tx_id
    )


    feature_rows.append(
        transaction_node[
            feature_columns
        ].values
    )


X = pd.DataFrame(
    feature_rows,
    columns=feature_columns
)


print(
    "\nSHAP input shape:",
    X.shape
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    X
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

# Depending on the SHAP / XGBoost versions,
# shap_values may occasionally be returned in a
# slightly different format.

if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[-1]


shap_values = np.asarray(
    shap_values
)


print(
    "SHAP values shape:",
    shap_values.shape
)


# ============================================================
# BUILD EXPLANATION RECORDS
# ============================================================

explanation_records = []


for row_idx, tx_id in enumerate(
    transaction_ids
):

    transaction_shap = shap_values[
        row_idx
    ]

    transaction_features = X.iloc[
        row_idx
    ]


    # --------------------------------------------------------
    # Rank features by absolute SHAP magnitude
    # --------------------------------------------------------

    ranked_indices = np.argsort(
        np.abs(
            transaction_shap
        )
    )[::-1]


    selected_count = 0


    for feature_idx in ranked_indices:

        shap_value = float(
            transaction_shap[
                feature_idx
            ]
        )


        # ----------------------------------------------------
        # We primarily want features that push toward fraud.
        #
        # Positive SHAP = pushes toward class 1.
        # ----------------------------------------------------

        if shap_value <= 0:

            continue


        feature_name = feature_columns[
            feature_idx
        ]

        feature_value = float(
            transaction_features[
                feature_name
            ]
        )


        explanation_records.append({

            "txId":
                tx_id,

            "feature_rank":
                selected_count + 1,

            "feature":
                feature_name,

            "feature_value":
                feature_value,

            "shap_value":
                shap_value,

            "direction":
                "INCREASES_FRAUD_RISK"

        })


        selected_count += 1


        if selected_count >= TOP_FEATURES:

            break


    # --------------------------------------------------------
    # If fewer than TOP_FEATURES positive features exist,
    # include the strongest remaining features.
    # --------------------------------------------------------

    if selected_count < TOP_FEATURES:

        for feature_idx in ranked_indices:

            shap_value = float(
                transaction_shap[
                    feature_idx
                ]
            )


            if shap_value > 0:

                continue


            feature_name = feature_columns[
                feature_idx
            ]

            feature_value = float(
                transaction_features[
                    feature_name
                ]
            )


            explanation_records.append({

                "txId":
                    tx_id,

                "feature_rank":
                    selected_count + 1,

                "feature":
                    feature_name,

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "direction":
                    "DECREASES_FRAUD_RISK"

            })


            selected_count += 1


            if selected_count >= TOP_FEATURES:

                break


# ============================================================
# CREATE DATAFRAME
# ============================================================

shap_df = pd.DataFrame(
    explanation_records
)


# ============================================================
# SORT
# ============================================================

if len(shap_df) > 0:

    shap_df = shap_df.sort_values(
        [
            "txId",
            "feature_rank"
        ]
    )


# ============================================================
# SAVE
# ============================================================

shap_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SHAP EXPLANATION SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nTransactions explained:",
    shap_df["txId"].nunique()
    if len(shap_df) > 0
    else 0
)


print(
    "Explanation records:",
    len(shap_df)
)


if len(shap_df) > 0:

    print(
        "\nTop SHAP features across cases:"
    )

    print(
        shap_df[
            "feature"
        ].value_counts().head(10)
    )


    print(
        "\nSample explanations:"
    )

    print(
        shap_df.head(20).to_string(
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
    "STEP 11 COMPLETE"
)

print(
    "=" * 70
)