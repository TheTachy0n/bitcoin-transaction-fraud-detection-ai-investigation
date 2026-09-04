# ============================================================
# STEP 16 — POLICY ENGINE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVESTIGATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "investigation_report.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "policy_decision.json"
)


# ============================================================
# POLICY ENGINE
# ============================================================

class PolicyEngine:

    def __init__(self):

        print("=" * 70)
        print("STEP 16 — POLICY ENGINE")
        print("=" * 70)

        print("\nPolicy Engine ready.")


    # ========================================================
    # LOAD INVESTIGATION
    # ========================================================

    def load_investigation(self):

        print("\nLoading investigation evidence...")

        with open(
            INVESTIGATION_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            investigation = json.load(f)

        print("Investigation loaded.")

        return investigation


    # ========================================================
    # APPLY POLICY
    # ========================================================

    def evaluate(self, investigation):

        if not investigation.get(
            "success",
            False
        ):

            return {

                "success": False,

                "error":
                    "Investigation was unsuccessful."
            }


        # ====================================================
        # STANDARDIZED EVIDENCE PACKAGE
        # ====================================================

        evidence = investigation.get(
            "evidence",
            {}
        )


        # ----------------------------------------------------
        # Risk assessment
        # ----------------------------------------------------

        risk = evidence.get(
            "risk_assessment",
            {}
        )


        # ----------------------------------------------------
        # Model predictions
        # ----------------------------------------------------

        models = evidence.get(
            "model_predictions",
            {}
        )


        # ----------------------------------------------------
        # Graph evidence
        # ----------------------------------------------------

        graph_package = evidence.get(
            "graph_evidence",
            {}
        )


        graph = graph_package.get(
            "graph_analysis",
            {}
        )


        transaction_graph_evidence = (
            graph_package.get(
                "transaction_graph_evidence",
                {}
            )
        )


        # ====================================================
        # EXTRACT VALUES
        # ====================================================

        risk_score = float(
            risk.get(
                "risk_score",
                0.0
            )
        )


        risk_level = str(
            risk.get(
                "risk_level",
                "UNKNOWN"
            )
        )


        alert_priority = str(
            risk.get(
                "alert_priority",
                "UNKNOWN"
            )
        )


        agreement = str(
            models.get(
                "agreement_category",
                evidence.get(
                    "model_agreement",
                    "UNKNOWN"
                )
            )
        )


        xgb_probability = float(
            models.get(
                "xgboost_probability",
                0.0
            )
        )


        gnn_probability = float(
            models.get(
                "graphsage_probability",
                0.0
            )
        )


        disagreement = float(
            models.get(
                "model_disagreement",
                1.0
            )
        )


        # ----------------------------------------------------
        # Graph values
        # ----------------------------------------------------

        neighbor_count = int(
            graph.get(
                "neighbor_count",
                transaction_graph_evidence.get(
                    "neighbor_count",
                    0
                )
            )
        )


        high_risk_neighbors = int(
            graph.get(
                "high_risk_neighbor_count",
                transaction_graph_evidence.get(
                    "high_risk_neighbor_count",
                    0
                )
            )
        )


        labeled_neighbors = int(
            graph.get(
                "labeled_neighbor_count",
                0
            )
        )


        # ====================================================
        # POLICY RULES
        # ====================================================

        reasons = []


        # ----------------------------------------------------
        # RULE 1 — Very high risk
        # ----------------------------------------------------

        if risk_score >= 0.90:

            reasons.append(
                "Risk score is extremely high."
            )


        elif risk_score >= 0.79:

            reasons.append(
                "Risk score exceeds the high-risk threshold."
            )


        # ----------------------------------------------------
        # RULE 2 — Model corroboration
        # ----------------------------------------------------

        if agreement == "BOTH_HIGH":

            reasons.append(
                "XGBoost and GraphSAGE independently "
                "assign high fraud probability."
            )


        elif agreement == "XGB_HIGH_GNN_LOW":

            reasons.append(
                "XGBoost provides strong transaction-level "
                "fraud evidence."
            )


        elif agreement == "XGB_LOW_GNN_HIGH":

            reasons.append(
                "GraphSAGE provides strong graph-based "
                "fraud evidence."
            )


        else:

            reasons.append(
                "The two models do not strongly corroborate "
                "the fraud prediction."
            )


        # ----------------------------------------------------
        # RULE 3 — Model disagreement
        # ----------------------------------------------------

        if disagreement <= 0.10:

            reasons.append(
                "Model disagreement is low."
            )

        elif disagreement <= 0.25:

            reasons.append(
                "Model disagreement is moderate."
            )

        else:

            reasons.append(
                "Model disagreement is high; additional "
                "manual investigation is recommended."
            )


        # ----------------------------------------------------
        # RULE 4 — Graph evidence
        # ----------------------------------------------------

        if high_risk_neighbors > 0:

            reasons.append(
                f"{high_risk_neighbors} high-risk graph "
                "neighbor(s) provide supporting evidence."
            )

        elif neighbor_count > 0:

            reasons.append(
                f"{neighbor_count} graph neighbor(s) were "
                "identified, but no high-risk neighbors "
                "were available."
            )

        else:

            reasons.append(
                "No graph neighbors were available."
            )


        # ----------------------------------------------------
        # RULE 5 — Unknown graph evidence
        # ----------------------------------------------------

        if (
            neighbor_count > 0
            and labeled_neighbors == 0
        ):

            reasons.append(
                "Graph neighbors are unlabeled; they cannot "
                "be assumed legitimate."
            )


        # ====================================================
        # FINAL DECISION
        # ====================================================

        requires_manual_review = False


        # ----------------------------------------------------
        # CRITICAL ESCALATION
        # ----------------------------------------------------

        if (

            risk_score >= 0.90

            and agreement == "BOTH_HIGH"

        ):

            decision = "ESCALATE"

            priority = "CRITICAL"

            requires_manual_review = True


        # ----------------------------------------------------
        # HIGH-RISK ESCALATION
        # ----------------------------------------------------

        elif risk_score >= 0.79:

            decision = "ESCALATE"

            priority = "HIGH"

            requires_manual_review = True


        # ----------------------------------------------------
        # MODERATE RISK
        # ----------------------------------------------------

        elif risk_score >= 0.50:

            decision = "MONITOR"

            priority = "MEDIUM"


        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        else:

            decision = "ALLOW"

            priority = "LOW"


        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {

            "success": True,

            "transaction_id":
                int(
                    investigation[
                        "transaction_id"
                    ]
                ),

            "decision":
                decision,

            "priority":
                priority,

            "requires_manual_review":
                requires_manual_review,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "alert_priority":
                alert_priority,

            "model_evidence": {

                "xgboost_probability":
                    xgb_probability,

                "graphsage_probability":
                    gnn_probability,

                "agreement_category":
                    agreement,

                "model_disagreement":
                    disagreement
            },

            "graph_evidence": {

                "neighbor_count":
                    neighbor_count,

                "high_risk_neighbor_count":
                    high_risk_neighbors,

                "labeled_neighbor_count":
                    labeled_neighbors
            },

            "reasons":
                reasons,

            "policy_statement":
                (
                    "This decision is a risk-based "
                    "operational recommendation and "
                    "does not constitute proof of fraud."
                )
        }


        return result


    # ========================================================
    # SAVE RESULT
    # ========================================================

    def save(self, result):

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )


        print(
            "\nPolicy decision saved:"
        )

        print(
            OUTPUT_PATH
        )


# ============================================================
# MAIN
# ============================================================

def main():

    engine = PolicyEngine()


    investigation = (
        engine.load_investigation()
    )


    print(
        "\nEvaluating investigation..."
    )


    result = engine.evaluate(
        investigation
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "POLICY DECISION"
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


    engine.save(
        result
    )


    print(
        "\nSTEP 16 COMPLETE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()