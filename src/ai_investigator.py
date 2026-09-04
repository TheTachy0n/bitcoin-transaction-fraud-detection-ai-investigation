# ============================================================
# STEP 15 — AI INVESTIGATOR
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import json
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT INVESTIGATION TOOLS
# ============================================================

from transaction_tool import TransactionTool
from graph_tool import GraphTool
from rag_retriever import RAGRetriever
from evidence_package import EvidencePackage


# ============================================================
# AI INVESTIGATOR
# ============================================================

class AIInvestigator:

    def __init__(self):

        print("=" * 70)
        print("AI INVESTIGATOR")
        print("=" * 70)

        print("\nInitializing investigation tools...")

        # ----------------------------------------------------
        # Transaction Tool
        # ----------------------------------------------------

        self.transaction_tool = TransactionTool()

        # ----------------------------------------------------
        # Graph Tool
        # ----------------------------------------------------

        self.graph_tool = GraphTool()

        # ----------------------------------------------------
        # RAG Retriever
        # ----------------------------------------------------

        self.rag = RAGRetriever()

        print("\nAI Investigator ready.")


    # ========================================================
    # INVESTIGATE TRANSACTION
    # ========================================================

    def investigate(self, tx_id):

        print(
            "\n" + "=" * 70
        )

        print(
            f"INVESTIGATING TRANSACTION: {tx_id}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # 1. TRANSACTION TOOL
        # ----------------------------------------------------

        print(
            "\n[1/3] Retrieving transaction evidence..."
        )

        transaction_result = (
            self.transaction_tool.investigate(
                tx_id
            )
        )

        if not transaction_result.get(
            "success",
            False
        ):

            return {
                "success": False,
                "error": transaction_result.get(
                    "error",
                    "Transaction not found"
                )
            }


        # ----------------------------------------------------
        # 2. GRAPH TOOL
        # ----------------------------------------------------

        print(
            "[2/3] Retrieving graph evidence..."
        )

        graph_result = (
            self.graph_tool.analyze(
                tx_id
            )
        )

        if not graph_result.get(
            "success",
            False
        ):

            graph_result = {
                "success": False,
                "error": graph_result.get(
                    "error",
                    "Graph analysis unavailable"
                )
            }


        # ----------------------------------------------------
        # 3. RAG
        # ----------------------------------------------------

        print(
            "[3/3] Retrieving investigation guidance..."
        )

        risk = transaction_result.get(
            "risk",
            {}
        )

        model_evidence = transaction_result.get(
            "model_evidence",
            {}
        )

        graph_evidence = transaction_result.get(
            "graph_evidence",
            {}
        )

        query = self._build_rag_query(
            risk,
            model_evidence,
            graph_evidence
        )

        rag_results = self.rag.retrieve(
            query,
            top_k=3
        )


        # ========================================================
        # BUILD STANDARDIZED EVIDENCE PACKAGE
        # ========================================================

        evidence_package = EvidencePackage(

            transaction=
                transaction_result.get(
                    "transaction",
                    {}
                ),

            risk_assessment=
                risk,

            model_predictions=
                model_evidence,

            model_agreement=
                model_evidence.get(
                    "agreement_category",
                    "UNKNOWN"
                ),

            shap_evidence=
                transaction_result.get(
                    "xgboost_explanation",
                    {}
                ),

            graph_evidence={

                "transaction_graph_evidence":
                    graph_evidence,

                "graph_analysis":
                    graph_result
            },

            rag_evidence=
                rag_results,

            metadata={

                "tools_called": [

                    "transaction_tool",

                    "graph_tool",

                    "rag_retriever"
                ]
            }
        )

        evidence = evidence_package.to_dict()


        # ----------------------------------------------------
        # GENERATE REPORT
        # ----------------------------------------------------

        report = self._generate_report(
            evidence
        )


        # ----------------------------------------------------
        # RETURN INVESTIGATION
        # ----------------------------------------------------

        return {

            "success": True,

            "transaction_id":
                int(tx_id),

            "evidence":
                evidence,

            "report":
                report
        }


    # ========================================================
    # BUILD RAG QUERY
    # ========================================================

    def _build_rag_query(

        self,
        risk,
        model_evidence,
        graph_evidence

    ):

        risk_level = risk.get(
            "risk_level",
            "UNKNOWN"
        )

        agreement = model_evidence.get(
            "agreement_category",
            "UNKNOWN"
        )

        high_risk_neighbors = (
            graph_evidence.get(
                "high_risk_neighbor_count",
                0
            )
        )

        return (

            f"Investigation guidance for a "
            f"{risk_level} risk transaction. "

            f"Model agreement category: "
            f"{agreement}. "

            f"High-risk graph neighbors: "
            f"{high_risk_neighbors}. "

            f"Explain appropriate investigation "
            f"procedures, model evidence, graph "
            f"evidence, SHAP interpretation, and "
            f"recommended actions."
        )


    # ========================================================
    # GENERATE REPORT
    # ========================================================

    def _generate_report(
        self,
        evidence
    ):

        # ====================================================
        # READ STANDARDIZED EVIDENCE PACKAGE
        # ====================================================

        transaction = evidence.get(
            "transaction",
            {}
        )

        risk = evidence.get(
            "risk_assessment",
            {}
        )

        models = evidence.get(
            "model_predictions",
            {}
        )

        agreement = evidence.get(
            "model_agreement",
            "UNKNOWN"
        )

        shap = evidence.get(
            "shap_evidence",
            {}
        )

        graph_package = evidence.get(
            "graph_evidence",
            {}
        )

        graph_evidence = graph_package.get(
            "transaction_graph_evidence",
            {}
        )

        graph = graph_package.get(
            "graph_analysis",
            {}
        )

        rag = evidence.get(
            "rag_evidence",
            []
        )


        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        tx_id = transaction.get(
            "txId"
        )

        timestep = transaction.get(
            "timestep"
        )


        risk_score = risk.get(
            "risk_score",
            0.0
        )

        risk_level = risk.get(
            "risk_level",
            "UNKNOWN"
        )

        alert_priority = risk.get(
            "alert_priority",
            "UNKNOWN"
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        xgb = models.get(
            "xgboost_probability",
            0.0
        )

        gnn = models.get(
            "graphsage_probability",
            0.0
        )

        # Use standardized agreement field
        # but fall back to model predictions if necessary.

        agreement = models.get(
            "agreement_category",
            agreement
        )

        disagreement = models.get(
            "model_disagreement",
            0.0
        )


        # ====================================================
        # GRAPH INFORMATION
        # ====================================================

        # Prefer Graph Tool analysis.

        neighbor_count = graph.get(
            "neighbor_count",
            graph_evidence.get(
                "neighbor_count",
                0
            )
        )

        high_risk_neighbors = graph.get(
            "high_risk_neighbor_count",
            graph_evidence.get(
                "high_risk_neighbor_count",
                0
            )
        )

        high_risk_rate = graph.get(
            "high_risk_neighbor_rate",
            graph_evidence.get(
                "neighbor_high_risk_rate",
                0.0
            )
        )

        labeled_count = graph.get(
            "labeled_neighbor_count",
            0
        )


        # ====================================================
        # SHAP
        # ====================================================

        top_features = shap.get(
            "top_features",
            []
        )


        # ====================================================
        # DETERMINE MODEL ASSESSMENT
        # ====================================================

        if agreement == "BOTH_HIGH":

            model_assessment = (

                "Both XGBoost and GraphSAGE assign "
                "high fraud risk. This provides "
                "strong corroborating model evidence."
            )

        elif agreement == "XGB_HIGH_GNN_LOW":

            model_assessment = (

                "XGBoost provides the stronger "
                "transaction-level fraud signal, "
                "while GraphSAGE provides weaker "
                "graph-based evidence."
            )

        elif agreement == "XGB_LOW_GNN_HIGH":

            model_assessment = (

                "GraphSAGE provides the stronger "
                "graph-based fraud signal, while "
                "XGBoost provides weaker "
                "transaction-level evidence."
            )

        else:

            model_assessment = (

                "The models do not provide strong "
                "corroborating evidence."
            )


        # ====================================================
        # GRAPH ASSESSMENT
        # ====================================================

        if high_risk_neighbors > 0:

            graph_assessment = (

                f"The transaction has "
                f"{high_risk_neighbors} high-risk "
                f"neighbor(s), with a high-risk "
                f"neighbor rate of "
                f"{high_risk_rate:.2%}. "
                f"This provides supporting graph evidence."
            )

        elif neighbor_count > 0 and labeled_count == 0:

            graph_assessment = (

                f"The transaction has "
                f"{neighbor_count} connected "
                f"neighbor(s), but no labeled neighbors "
                f"are available. Therefore the graph "
                f"provides limited corroborating evidence."
            )

        elif neighbor_count > 0:

            graph_assessment = (

                f"The transaction has "
                f"{neighbor_count} connected neighbor(s), "
                f"but no high-risk neighbors were identified."
            )

        else:

            graph_assessment = (

                "No graph neighbors were identified, "
                "so graph-based evidence is limited."
            )


        # ====================================================
        # SHAP ASSESSMENT
        # ====================================================

        shap_lines = []

        for feature in top_features[:5]:

            feature_name = feature.get(
                "feature",
                "unknown"
            )

            shap_value = feature.get(
                "shap_value",
                0.0
            )

            direction = feature.get(
                "direction",
                "UNKNOWN"
            )

            shap_lines.append(

                f"{feature_name}: "
                f"SHAP {shap_value:+.4f} "
                f"({direction})"
            )


        if shap_lines:

            shap_assessment = "\n".join(
                shap_lines
            )

        else:

            shap_assessment = (
                "No SHAP explanation available."
            )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        if risk_score >= 0.79:

            recommendation = (

                "ESCALATE FOR MANUAL REVIEW. "
                "Review the model evidence, inspect "
                "graph relationships, examine the "
                "highest-impact SHAP features, and "
                "retrieve additional transaction context "
                "before making a final decision."
            )

        else:

            recommendation = (

                "CONTINUE MONITORING. "
                "The transaction does not currently "
                "meet the configured high-risk threshold."
            )


        # ====================================================
        # RAG EVIDENCE
        # ====================================================

        rag_lines = []

        for result in rag[:3]:

            source = result.get(
                "source",
                "unknown"
            )

            score = result.get(
                "score",
                0.0
            )

            rag_lines.append(

                f"{source} "
                f"(relevance {score:.4f})"
            )


        if rag_lines:

            rag_summary = "\n".join(
                rag_lines
            )

        else:

            rag_summary = (
                "No investigation guidance retrieved."
            )


        # ====================================================
        # FINAL REPORT
        # ====================================================

        report = f"""

============================================================
AI FRAUD INVESTIGATION REPORT
============================================================

TRANSACTION
------------------------------------------------------------
Transaction ID : {tx_id}
Timestep       : {timestep}


RISK ASSESSMENT
------------------------------------------------------------
Risk Score     : {risk_score:.6f}
Risk Level     : {risk_level}
Alert Priority : {alert_priority}


MODEL EVIDENCE
------------------------------------------------------------
XGBoost        : {xgb:.6f}
GraphSAGE      : {gnn:.6f}
Agreement      : {agreement}
Disagreement   : {disagreement:.6f}

Assessment:
{model_assessment}


GRAPH EVIDENCE
------------------------------------------------------------
Neighbors               : {neighbor_count}
High-risk neighbors     : {high_risk_neighbors}
High-risk neighbor rate : {high_risk_rate:.2%}
Labeled neighbors       : {labeled_count}

Assessment:
{graph_assessment}


XGBOOST / SHAP EVIDENCE
------------------------------------------------------------
Top contributing features:

{shap_assessment}


RETRIEVED INVESTIGATION GUIDANCE
------------------------------------------------------------
{rag_summary}


RECOMMENDED ACTION
------------------------------------------------------------
{recommendation}


CONFIDENCE AND LIMITATIONS
------------------------------------------------------------
The investigation is based on machine-learning model
outputs, graph relationships, SHAP explanations, and
retrieved investigation guidance.

The model prediction is not proof of fraud.

SHAP values explain model behavior and do not establish
causality.

Unknown graph neighbors cannot be assumed legitimate.

Feature identifiers should not be assigned real-world
meaning unless their semantic meaning is known.

============================================================
END OF INVESTIGATION REPORT
============================================================
"""

        return report


# ============================================================
# MAIN
# ============================================================

def main():

    investigator = AIInvestigator()


    # --------------------------------------------------------
    # Demo transaction
    # --------------------------------------------------------

    tx_id = 71987809


    result = investigator.investigate(
        tx_id
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL INVESTIGATION"
    )

    print(
        "=" * 70
    )


    if result["success"]:

        print(
            result["report"]
        )

    else:

        print(
            "\nInvestigation failed:"
        )

        print(
            result["error"]
        )


    # --------------------------------------------------------
    # Save JSON evidence package
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "results"
        / "investigation_report.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )


    print(
        "\nEvidence package saved:"
    )

    print(
        output_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()