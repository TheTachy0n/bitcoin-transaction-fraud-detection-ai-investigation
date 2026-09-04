# ============================================================
# STEP 13 — GRAPH INVESTIGATION TOOL
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import pickle
import json

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

RISK_ENGINE_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_risk_engine.csv"
)


# ============================================================
# GRAPH TOOL
# ============================================================

class GraphTool:

    def __init__(self):

        print("Initializing Graph Tool...")

        # ----------------------------------------------------
        # Load graph
        # ----------------------------------------------------

        print("Loading graph data...")

        with open(
            GRAPH_PATH,
            "rb"
        ) as f:

            graph = pickle.load(f)


        self.nodes = graph[
            "nodes"
        ].copy()

        self.edge_index = graph[
            "edge_index"
        ]


        self.nodes["txId"] = (
            self.nodes["txId"]
            .astype(int)
        )


        # ----------------------------------------------------
        # Build transaction → node lookup
        # ----------------------------------------------------

        self.tx_to_idx = {

            int(tx_id): idx

            for idx, tx_id in enumerate(
                self.nodes["txId"].values
            )
        }


        self.idx_to_tx = {

            idx: int(tx_id)

            for idx, tx_id in enumerate(
                self.nodes["txId"].values
            )
        }


        # ----------------------------------------------------
        # Load risk scores
        # ----------------------------------------------------

        print("Loading risk scores...")

        risk_df = pd.read_csv(
            RISK_ENGINE_PATH
        )


        risk_df["txId"] = (
            risk_df["txId"]
            .astype(int)
        )


        self.risk_lookup = {

            int(row["txId"]): float(
                row["risk_score"]
            )

            for _, row in risk_df.iterrows()
        }


        # ----------------------------------------------------
        # Build adjacency map
        # ----------------------------------------------------

        print("Building adjacency map...")

        self.neighbor_map = {}


        for source, target in zip(
            self.edge_index[0],
            self.edge_index[1]
        ):

            source = int(source)
            target = int(target)


            if source not in self.neighbor_map:

                self.neighbor_map[
                    source
                ] = []


            self.neighbor_map[
                source
            ].append(
                target
            )


        print(
            "Graph nodes:",
            len(self.nodes)
        )

        print(
            "Graph edges:",
            self.edge_index.shape[1]
        )

        print(
            "Risk scores:",
            len(self.risk_lookup)
        )

        print("Graph Tool ready.")


    # ========================================================
    # GET NEIGHBORS
    # ========================================================

    def get_neighbors(
        self,
        tx_id
    ):

        tx_id = int(tx_id)


        # ----------------------------------------------------
        # Check transaction
        # ----------------------------------------------------

        if tx_id not in self.tx_to_idx:

            return {

                "success": False,

                "error": (
                    f"Transaction {tx_id} "
                    "was not found in graph."
                )
            }


        node_idx = self.tx_to_idx[
            tx_id
        ]


        # ----------------------------------------------------
        # Get neighbors
        # ----------------------------------------------------

        neighbor_indices = (
            self.neighbor_map.get(
                node_idx,
                []
            )
        )


        # Remove duplicates

        neighbor_indices = list(
            dict.fromkeys(
                neighbor_indices
            )
        )


        neighbors = []


        for neighbor_idx in (
            neighbor_indices
        ):

            neighbor_tx_id = (
                self.idx_to_tx.get(
                    neighbor_idx
                )
            )


            if neighbor_tx_id is None:

                continue


            # -----------------------------------------------
            # Node information
            # -----------------------------------------------

            node = self.nodes.iloc[
                neighbor_idx
            ]


            timestep = int(
                node["timestep"]
            )


            label = int(
                node["label"]
            )


            risk_score = (
                self.risk_lookup.get(
                    neighbor_tx_id
                )
            )


            neighbors.append({

                "txId":
                    neighbor_tx_id,

                "timestep":
                    timestep,

                "label":
                    label,

                "risk_score":
                    risk_score
            })


        return {

            "success": True,

            "txId":
                tx_id,

            "neighbor_count":
                len(neighbors),

            "neighbors":
                neighbors
        }


    # ========================================================
    # ANALYZE NEIGHBORHOOD
    # ========================================================

    def analyze_neighborhood(
        self,
        tx_id
    ):

        result = self.get_neighbors(
            tx_id
        )


        if not result[
            "success"
        ]:

            return result


        neighbors = result[
            "neighbors"
        ]


        # ----------------------------------------------------
        # Extract available risk scores
        # ----------------------------------------------------

        risk_scores = [

            neighbor["risk_score"]

            for neighbor in neighbors

            if neighbor["risk_score"]
            is not None
        ]


        # ----------------------------------------------------
        # Basic statistics
        # ----------------------------------------------------

        neighbor_count = len(
            neighbors
        )


        if risk_scores:

            average_risk = (
                sum(risk_scores)
                / len(risk_scores)
            )

        else:

            average_risk = None


        high_risk_neighbors = [

            neighbor

            for neighbor in neighbors

            if (
                neighbor["risk_score"]
                is not None
                and neighbor["risk_score"]
                >= 0.79
            )
        ]


        high_risk_count = len(
            high_risk_neighbors
        )


        if neighbor_count > 0:

            high_risk_rate = (
                high_risk_count
                / neighbor_count
            )

        else:

            high_risk_rate = 0.0


        # ----------------------------------------------------
        # Fraud-labeled neighbors
        # ----------------------------------------------------

        labeled_neighbors = [

            neighbor

            for neighbor in neighbors

            if neighbor["label"] in [0, 1]
        ]


        fraud_labeled_neighbors = [

            neighbor

            for neighbor
            in labeled_neighbors

            if neighbor["label"] == 1
        ]


        if labeled_neighbors:

            labeled_fraud_rate = (

                len(
                    fraud_labeled_neighbors
                )

                /

                len(
                    labeled_neighbors
                )
            )

        else:

            labeled_fraud_rate = 0.0


        # ----------------------------------------------------
        # Return analysis
        # ----------------------------------------------------

        return {

            "success":
                True,

            "txId":
                int(tx_id),

            "neighbor_count":
                neighbor_count,

            "average_neighbor_risk":
                average_risk,

            "high_risk_neighbor_count":
                high_risk_count,

            "high_risk_neighbor_rate":
                high_risk_rate,

            "labeled_neighbor_count":
                len(
                    labeled_neighbors
                ),

            "labeled_neighbor_fraud_rate":
                labeled_fraud_rate,

            "high_risk_neighbors":
                high_risk_neighbors,

            "neighbors":
                neighbors
        }
        # ========================================================
    # ANALYZE
    # Compatibility interface for AI Investigator
    # ========================================================

    def analyze(self, tx_id):
        return self.analyze_neighborhood(
            tx_id
        )

# ============================================================
# DEMO
# ============================================================

def main():

    print("=" * 70)
    print("GRAPH INVESTIGATION TOOL")
    print("=" * 70)


    tool = GraphTool()


    tx_id = 71987809


    print(
        f"\nAnalyzing transaction: {tx_id}"
    )


    result = tool.analyze_neighborhood(
        tx_id
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "GRAPH ANALYSIS RESULT"
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


if __name__ == "__main__":

    main()