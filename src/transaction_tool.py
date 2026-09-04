# ============================================================
# STEP 12 — TRANSACTION INVESTIGATION TOOL
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import json
import pickle

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

INVESTIGATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "investigation_evidence.csv"
)

SHAP_PATH = (
    PROJECT_ROOT
    / "results"
    / "shap_explanations.csv"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_data.pkl"
)


# ============================================================
# TRANSACTION TOOL
# ============================================================

class TransactionTool:

    def __init__(self):

        print("Initializing Transaction Tool...")

        # ----------------------------------------------------
        # Load risk engine
        # ----------------------------------------------------

        print("Loading risk engine...")

        if not RISK_ENGINE_PATH.exists():

            raise FileNotFoundError(
                f"Risk engine file not found:\n"
                f"{RISK_ENGINE_PATH}"
            )

        self.risk_df = pd.read_csv(
            RISK_ENGINE_PATH
        )

        self.risk_df["txId"] = (
            self.risk_df["txId"]
            .astype(int)
        )


        # ----------------------------------------------------
        # Load investigation evidence
        # ----------------------------------------------------

        print("Loading investigation evidence...")

        if not INVESTIGATION_PATH.exists():

            raise FileNotFoundError(
                f"Investigation evidence file not found:\n"
                f"{INVESTIGATION_PATH}"
            )

        self.investigation_df = pd.read_csv(
            INVESTIGATION_PATH
        )

        self.investigation_df["txId"] = (
            self.investigation_df["txId"]
            .astype(int)
        )


        # ----------------------------------------------------
        # Load SHAP explanations
        # ----------------------------------------------------

        print("Loading SHAP explanations...")

        if not SHAP_PATH.exists():

            raise FileNotFoundError(
                f"SHAP explanation file not found:\n"
                f"{SHAP_PATH}"
            )

        self.shap_df = pd.read_csv(
            SHAP_PATH
        )

        self.shap_df["txId"] = (
            self.shap_df["txId"]
            .astype(int)
        )


        # ----------------------------------------------------
        # Load graph data ONCE
        # ----------------------------------------------------

        print("Loading graph data...")

        if not GRAPH_PATH.exists():

            raise FileNotFoundError(
                f"Graph data file not found:\n"
                f"{GRAPH_PATH}"
            )

        with open(
            GRAPH_PATH,
            "rb"
        ) as f:

            self.graph = pickle.load(f)


        self.nodes = self.graph["nodes"].copy()


        self.nodes["txId"] = (
            self.nodes["txId"]
            .astype(int)
        )


        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        self.feature_columns = [

            column

            for column in self.nodes.columns

            if column.startswith(
                "feature_"
            )
        ]


        # ----------------------------------------------------
        # Build transaction lookup
        # ----------------------------------------------------

        self.node_lookup = {

            int(row["txId"]): row

            for _, row in self.nodes.iterrows()

        }


        # ----------------------------------------------------
        # Build risk lookup
        # ----------------------------------------------------

        self.risk_lookup = {

            int(row["txId"]): row

            for _, row in self.risk_df.iterrows()

        }


        # ----------------------------------------------------
        # Build investigation lookup
        # ----------------------------------------------------

        self.investigation_lookup = {

            int(row["txId"]): row

            for _, row in self.investigation_df.iterrows()

        }


        # ----------------------------------------------------
        # Print statistics
        # ----------------------------------------------------

        print(
            "Graph nodes:",
            len(self.nodes)
        )

        print(
            "Transaction features:",
            len(self.feature_columns)
        )

        print(
            "Risk records:",
            len(self.risk_lookup)
        )

        print(
            "Investigation records:",
            len(self.investigation_lookup)
        )

        print(
            "SHAP records:",
            len(self.shap_df)
        )

        print("Transaction Tool ready.")


    # ========================================================
    # INVESTIGATE
    #
    # Public interface used by AI Investigator
    # ========================================================

    def investigate(
        self,
        tx_id
    ):

        tx_id = int(tx_id)

        return self.get_transaction(
            tx_id
        )


    # ========================================================
    # GET TRANSACTION
    # ========================================================

    def get_transaction(
        self,
        tx_id
    ):

        tx_id = int(tx_id)


        # ----------------------------------------------------
        # Check transaction
        # ----------------------------------------------------

        if tx_id not in self.risk_lookup:

            return {

                "success": False,

                "error": (
                    f"Transaction {tx_id} "
                    "was not found in the risk engine."
                )
            }


        risk_row = self.risk_lookup[
            tx_id
        ]


        # ====================================================
        # BASIC TRANSACTION INFORMATION
        # ====================================================

        result = {

            "success": True,

            "transaction": {

                "txId":
                    tx_id,

                "timestep":
                    int(
                        risk_row["timestep"]
                    ),

                "label":
                    int(
                        risk_row["label"]
                    )
            },


            # ------------------------------------------------
            # Risk
            # ------------------------------------------------

            "risk": {

                "risk_score":
                    float(
                        risk_row["risk_score"]
                    ),

                "risk_level":
                    str(
                        risk_row["risk_level"]
                    ),

                "alert_priority":
                    str(
                        risk_row["alert_priority"]
                    ),

                "fraud_alert":
                    str(
                        risk_row["fraud_alert"]
                    )
            },


            # ------------------------------------------------
            # Model evidence
            # ------------------------------------------------

            "model_evidence": {

                "xgboost_probability":
                    float(
                        risk_row[
                            "xgboost_probability"
                        ]
                    ),

                "graphsage_probability":
                    float(
                        risk_row[
                            "graphsage_probability"
                        ]
                    ),

                "agreement_category":
                    str(
                        risk_row[
                            "agreement_category"
                        ]
                    ),

                "model_agreement":
                    float(
                        risk_row[
                            "model_agreement"
                        ]
                    ),

                "model_disagreement":
                    float(
                        risk_row[
                            "model_disagreement"
                        ]
                    ),

                "evidence_type":
                    str(
                        risk_row[
                            "evidence_type"
                        ]
                    )
            }
        }


        # ====================================================
        # GRAPH EVIDENCE
        # ====================================================

        if tx_id in self.investigation_lookup:

            graph_row = (
                self.investigation_lookup[
                    tx_id
                ]
            )


            result[
                "graph_evidence"
            ] = {

                "neighbor_count":
                    safe_int(
                        graph_row[
                            "neighbor_count"
                        ]
                    ),

                "high_risk_neighbor_count":
                    safe_int(
                        graph_row[
                            "high_risk_neighbor_count"
                        ]
                    ),

                "neighbor_avg_risk":
                    safe_float(
                        graph_row[
                            "neighbor_avg_risk"
                        ]
                    ),

                "neighbor_high_risk_rate":
                    safe_float(
                        graph_row[
                            "neighbor_high_risk_rate"
                        ]
                    ),

                "neighbor_tx_ids":
                    parse_neighbor_ids(
                        graph_row[
                            "neighbor_tx_ids"
                        ]
                    ),

                "evidence_summary":
                    str(
                        graph_row[
                            "evidence_summary"
                        ]
                    )
            }

        else:

            result[
                "graph_evidence"
            ] = {

                "available":
                    False,

                "neighbor_count":
                    0,

                "high_risk_neighbor_count":
                    0,

                "neighbor_avg_risk":
                    0.0,

                "neighbor_high_risk_rate":
                    0.0,

                "neighbor_tx_ids":
                    [],

                "evidence_summary":
                    "No investigation evidence available."
            }


        # ====================================================
        # SHAP EVIDENCE
        # ====================================================

        transaction_shap = (

            self.shap_df[
                self.shap_df["txId"] == tx_id
            ]

            .sort_values(
                "feature_rank"
            )
        )


        shap_evidence = []


        for _, row in transaction_shap.iterrows():

            shap_evidence.append({

                "rank":
                    safe_int(
                        row["feature_rank"]
                    ),

                "feature":
                    str(
                        row["feature"]
                    ),

                "feature_value":
                    safe_float(
                        row["feature_value"]
                    ),

                "shap_value":
                    safe_float(
                        row["shap_value"]
                    ),

                "direction":
                    str(
                        row["direction"]
                    )
            })


        result[
            "xgboost_explanation"
        ] = {

            "top_features":
                shap_evidence
        }


        return result


    # ========================================================
    # GET RAW FEATURES
    # ========================================================

    def get_raw_features(
        self,
        tx_id
    ):

        tx_id = int(tx_id)


        # ----------------------------------------------------
        # Check node exists
        # ----------------------------------------------------

        if tx_id not in self.node_lookup:

            return {

                "success": False,

                "error": (
                    f"Transaction {tx_id} "
                    "was not found in graph data."
                )
            }


        transaction = self.node_lookup[
            tx_id
        ]


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = {}


        for feature in self.feature_columns:

            value = transaction[
                feature
            ]


            if pd.isna(value):

                features[
                    feature
                ] = None

            else:

                try:

                    features[
                        feature
                    ] = float(value)

                except Exception:

                    features[
                        feature
                    ] = None


        return {

            "success":
                True,

            "txId":
                tx_id,

            "feature_count":
                len(features),

            "features":
                features
        }


# ============================================================
# HELPER — SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    try:

        if pd.isna(value):

            return 0.0

        return float(value)

    except Exception:

        return 0.0


# ============================================================
# HELPER — SAFE INT
# ============================================================

def safe_int(
    value
):

    try:

        if pd.isna(value):

            return 0

        return int(value)

    except Exception:

        return 0


# ============================================================
# HELPER — PARSE NEIGHBOR IDS
# ============================================================

def parse_neighbor_ids(
    value
):

    if pd.isna(value):

        return []


    value = str(
        value
    ).strip()


    if not value:

        return []


    neighbor_ids = []


    for item in value.split(","):

        item = item.strip()

        if not item:

            continue

        try:

            neighbor_ids.append(
                int(item)
            )

        except ValueError:

            continue


    return neighbor_ids


# ============================================================
# DEMO
# ============================================================

def main():

    print("=" * 70)
    print("TRANSACTION INVESTIGATION TOOL")
    print("=" * 70)


    tool = TransactionTool()


    # --------------------------------------------------------
    # Example transaction
    # --------------------------------------------------------

    tx_id = 71987809


    print(
        f"\nInvestigating transaction: {tx_id}"
    )


    result = tool.investigate(
        tx_id
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "TRANSACTION RESULT"
    )

    print(
        "=" * 70
    )


    print(
        json.dumps(
            result,
            indent=4
        )
    )


    # --------------------------------------------------------
    # Raw feature demonstration
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAW FEATURE TOOL"
    )

    print(
        "=" * 70
    )


    raw_features = (
        tool.get_raw_features(
            tx_id
        )
    )


    if raw_features[
        "success"
    ]:

        print(
            "Features retrieved:",
            raw_features[
                "feature_count"
            ]
        )

    else:

        print(
            raw_features
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()