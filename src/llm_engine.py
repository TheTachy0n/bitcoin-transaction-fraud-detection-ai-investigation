# ============================================================
# STEP 17 — GROQ LLM ENGINE
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

import os
import json


# ============================================================
# GROQ IMPORT
# ============================================================

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# LLM ENGINE
# ============================================================

class LLMEngine:

    def __init__(self):

        print("=" * 70)
        print("GROQ LLM ENGINE")
        print("=" * 70)

        self.client = None
        self.llm_available = False

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = "openai/gpt-oss-120b"

        # ----------------------------------------------------
        # API key
        # ----------------------------------------------------

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            print(
                "\nWARNING: GROQ_API_KEY not found."
            )

            print(
                "Deterministic fallback will be used."
            )

            return


        # ----------------------------------------------------
        # Groq package
        # ----------------------------------------------------

        if Groq is None:

            print(
                "\nWARNING: Groq package is not installed."
            )

            print(
                "Deterministic fallback will be used."
            )

            return


        # ----------------------------------------------------
        # Initialize client
        # ----------------------------------------------------

        try:

            print(
                "\nInitializing Groq client..."
            )

            self.client = Groq(
                api_key=api_key
            )

            self.llm_available = True

            print(
                f"Model: {self.model}"
            )

            print(
                "Groq LLM Engine ready."
            )

        except Exception as e:

            print(
                "\nWARNING: Could not initialize Groq."
            )

            print(
                f"Reason: {e}"
            )

            print(
                "Deterministic fallback will be used."
            )

            self.client = None
            self.llm_available = False


    # ========================================================
    # MAIN GENERATION FUNCTION
    # ========================================================

    def generate_investigation(self, evidence):

        """
        Generate an investigation report.

        Groq is used when available.

        If Groq is unavailable or the request fails,
        a deterministic fallback report is returned.
        """

        # ----------------------------------------------------
        # Try Groq
        # ----------------------------------------------------

        if self.llm_available:

            try:

                return self._generate_with_groq(
                    evidence
                )

            except Exception as e:

                print(
                    "\nWARNING: Groq request failed."
                )

                print(
                    f"Reason: {e}"
                )

                print(
                    "Using deterministic fallback."
                )


        # ----------------------------------------------------
        # Deterministic fallback
        # ----------------------------------------------------

        return self._generate_fallback(
            evidence
        )


    # ========================================================
    # GROQ GENERATION
    # ========================================================

    def _generate_with_groq(self, evidence):

        print(
            "\nSending investigation evidence to Groq..."
        )

        # ----------------------------------------------------
        # Extract investigation and policy
        # ----------------------------------------------------

        investigation = evidence.get(
            "investigation",
            {}
        )

        policy = evidence.get(
            "policy_decision",
            {}
        )


        investigation_evidence = (
            investigation.get(
                "evidence",
                {}
            )
        )


        # ----------------------------------------------------
        # Build compact evidence payload
        # ----------------------------------------------------

        payload = {

            "transaction":
                investigation_evidence.get(
                    "transaction",
                    {}
                ),

            "risk_assessment":
                investigation_evidence.get(
                    "risk_assessment",
                    {}
                ),

            "model_predictions":
                investigation_evidence.get(
                    "model_predictions",
                    {}
                ),

            "model_agreement":
                investigation_evidence.get(
                    "model_agreement",
                    "UNKNOWN"
                ),

            "graph_evidence":
                investigation_evidence.get(
                    "graph_evidence",
                    {}
                ),

            "shap_evidence":
                investigation_evidence.get(
                    "shap_evidence",
                    {}
                ),

            "rag_evidence":
                investigation_evidence.get(
                    "rag_evidence",
                    []
                ),

            "policy_decision":
                policy
        }


        # ----------------------------------------------------
        # System prompt
        # ----------------------------------------------------

        system_prompt = """
You are an AI fraud investigation analyst.

Your job is to analyze the evidence produced by a fraud
detection system and write a concise, evidence-grounded
investigation report.

IMPORTANT RULES:

1. Do not invent transaction facts.
2. Do not claim that an investigation step was performed
   unless the evidence explicitly shows it.
3. Clearly distinguish model predictions from confirmed facts.
4. Treat unknown or unlabeled graph neighbors as unknown.
5. Do not override the Policy Engine decision.
6. Preserve the policy decision and priority exactly.
7. Explain why the transaction received its risk level.
8. Explain model agreement or disagreement.
9. Explain relevant graph evidence.
10. Explain the most important SHAP features.
11. Use the RAG evidence as investigation guidance.
12. Clearly state important limitations.
13. Recommend appropriate next investigative steps when
    evidence is incomplete.

Structure the response as:

INVESTIGATION SUMMARY

RISK ASSESSMENT

MODEL EVIDENCE

GRAPH EVIDENCE

KEY RISK DRIVERS

INVESTIGATION GUIDANCE

POLICY DECISION

LIMITATIONS

RECOMMENDED NEXT STEPS
"""


        # ----------------------------------------------------
        # User prompt
        # ----------------------------------------------------

        user_prompt = f"""
Analyze the following fraud investigation evidence.

Do not introduce facts that are not present in the evidence.

Evidence:

{json.dumps(payload, indent=2, default=str)}

Produce the investigation report using the required structure.
"""


        # ----------------------------------------------------
        # Groq API call
        # ----------------------------------------------------

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.2,

            max_completion_tokens=3000
        )


        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        report = response.choices[0].message.content


        if not report:

            raise RuntimeError(
                "Groq returned an empty response."
            )


        print(
            "Groq investigation report generated."
        )


        return report


    # ========================================================
    # DETERMINISTIC FALLBACK
    # ========================================================

    def _generate_fallback(self, evidence):

        print(
            "\nGenerating deterministic investigation report..."
        )

        investigation = evidence.get(
            "investigation",
            {}
        )

        policy = evidence.get(
            "policy_decision",
            {}
        )

        investigation_evidence = (
            investigation.get(
                "evidence",
                {}
            )
        )


        # ----------------------------------------------------
        # Extract evidence
        # ----------------------------------------------------

        transaction = (
            investigation_evidence.get(
                "transaction",
                {}
            )
        )

        risk = (
            investigation_evidence.get(
                "risk_assessment",
                {}
            )
        )

        models = (
            investigation_evidence.get(
                "model_predictions",
                {}
            )
        )

        agreement = (
            investigation_evidence.get(
                "model_agreement",
                "UNKNOWN"
            )
        )

        graph_package = (
            investigation_evidence.get(
                "graph_evidence",
                {}
            )
        )

        graph = (
            graph_package.get(
                "graph_analysis",
                {}
            )
        )

        shap = (
            investigation_evidence.get(
                "shap_evidence",
                {}
            )
        )

        rag = (
            investigation_evidence.get(
                "rag_evidence",
                []
            )
        )


        # ----------------------------------------------------
        # Transaction
        # ----------------------------------------------------

        tx_id = transaction.get(
            "txId",
            "UNKNOWN"
        )

        timestep = transaction.get(
            "timestep",
            "UNKNOWN"
        )


        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk_score = risk.get(
            "risk_score",
            "UNKNOWN"
        )

        risk_level = risk.get(
            "risk_level",
            "UNKNOWN"
        )

        priority = risk.get(
            "alert_priority",
            "UNKNOWN"
        )


        # ----------------------------------------------------
        # Model evidence
        # ----------------------------------------------------

        xgb = models.get(
            "xgboost_probability",
            "UNKNOWN"
        )

        gnn = models.get(
            "graphsage_probability",
            "UNKNOWN"
        )

        disagreement = models.get(
            "model_disagreement",
            "UNKNOWN"
        )


        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        neighbor_count = graph.get(
            "neighbor_count",
            0
        )

        high_risk_neighbors = graph.get(
            "high_risk_neighbor_count",
            0
        )

        labeled_neighbors = graph.get(
            "labeled_neighbor_count",
            0
        )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        top_features = shap.get(
            "top_features",
            []
        )


        feature_lines = []

        for feature in top_features[:5]:

            feature_lines.append(
                f"- {feature.get('feature')}: "
                f"SHAP {feature.get('shap_value')}, "
                f"direction {feature.get('direction')}"
            )


        if not feature_lines:

            feature_lines.append(
                "- No SHAP explanations available."
            )


        # ----------------------------------------------------
        # RAG sources
        # ----------------------------------------------------

        rag_sources = []

        for item in rag:

            source = item.get(
                "source"
            )

            if source:
                rag_sources.append(
                    source
                )


        # Remove duplicates
        rag_sources = list(
            dict.fromkeys(
                rag_sources
            )
        )


        # ----------------------------------------------------
        # Policy
        # ----------------------------------------------------

        decision = policy.get(
            "decision",
            "UNKNOWN"
        )

        policy_priority = policy.get(
            "priority",
            "UNKNOWN"
        )


        # ----------------------------------------------------
        # Build report
        # ----------------------------------------------------

        report = f"""
INVESTIGATION SUMMARY

Transaction {tx_id} received a risk score of {risk_score}
and was classified as {risk_level} risk.

The transaction is associated with timestep {timestep}.


RISK ASSESSMENT

Risk score: {risk_score}
Risk level: {risk_level}
Alert priority: {priority}


MODEL EVIDENCE

XGBoost fraud probability: {xgb}
GraphSAGE fraud probability: {gnn}
Model agreement: {agreement}
Model disagreement: {disagreement}

The model outputs provide predictive evidence and should
not be interpreted as definitive proof of fraudulent activity.


GRAPH EVIDENCE

Graph neighbors identified: {neighbor_count}
High-risk neighbors: {high_risk_neighbors}
Labeled neighbors: {labeled_neighbors}

Unlabeled or unavailable graph evidence should be treated
as unknown rather than assumed legitimate.


KEY RISK DRIVERS

The available SHAP explanation identifies the following
features as the strongest contributors:

{chr(10).join(feature_lines)}


INVESTIGATION GUIDANCE

Relevant knowledge-base sources retrieved:

{chr(10).join(f"- {source}" for source in rag_sources)
 if rag_sources else "- No RAG sources available."}

These sources provide investigation guidance and contextual
information rather than transaction-specific proof.


POLICY DECISION

Decision: {decision}
Priority: {policy_priority}

The Policy Engine decision is the authoritative operational
recommendation of this pipeline.


LIMITATIONS

The available evidence does not by itself establish that the
transaction is fraudulent. Model predictions, graph structure,
and retrieved knowledge should be interpreted together with
additional investigative evidence.


RECOMMENDED NEXT STEPS

Review relevant transaction history, counterparties, graph
relationships, and other available compliance signals.
Additional verification should be performed before treating
the prediction as confirmed fraud.
""".strip()


        print(
            "Deterministic fallback report generated."
        )


        return report


    # ========================================================
    # POLICY CONSISTENCY VALIDATION
    # ========================================================

    def validate_policy_consistency(
        self,
        report,
        policy
    ):

        """
        Verify that the generated report contains the
        authoritative Policy Engine decision and priority.
        """

        if report is None:

            report = ""


        report_text = str(
            report
        ).upper()


        decision = str(
            policy.get(
                "decision",
                ""
            )
        ).upper()


        priority = str(
            policy.get(
                "priority",
                ""
            )
        ).upper()


        decision_present = (
            bool(decision)
            and decision in report_text
        )

        priority_present = (
            bool(priority)
            and priority in report_text
        )


        return {

            "policy_decision":
                policy.get(
                    "decision"
                ),

            "policy_priority":
                policy.get(
                    "priority"
                ),

            "decision_present":
                decision_present,

            "priority_present":
                priority_present,

            "consistent":
                decision_present
                and priority_present
        }


    # ========================================================
    # SAVE REPORT
    # ========================================================

    def save_report(
        self,
        report,
        output_path,
        policy=None,
        llm_used=None,
        fallback_used=None
    ):

        output = {

            "report":
                report,

            "policy":
                policy or {},

            "llm": {

                "used":
                    llm_used,

                "fallback":
                    fallback_used
            }
        }


        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=4,
                default=str
            )


        print(
            f"\nReport saved to: {output_path}"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    project_root = (
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(
                    __file__
                )
            )
        )
    )


    investigation_path = os.path.join(
        project_root,
        "results",
        "investigation_report.json"
    )

    policy_path = os.path.join(
        project_root,
        "results",
        "policy_decision.json"
    )


    if not os.path.exists(
        investigation_path
    ):

        print(
            "Investigation report not found:"
        )

        print(
            investigation_path
        )

        return


    if not os.path.exists(
        policy_path
    ):

        print(
            "Policy decision not found:"
        )

        print(
            policy_path
        )

        return


    with open(
        investigation_path,
        "r",
        encoding="utf-8"
    ) as f:

        investigation = json.load(f)


    with open(
        policy_path,
        "r",
        encoding="utf-8"
    ) as f:

        policy = json.load(f)


    evidence = {

        "investigation":
            investigation,

        "policy_decision":
            policy
    }


    engine = LLMEngine()


    report = (
        engine.generate_investigation(
            evidence
        )
    )


    validation = (
        engine.validate_policy_consistency(
            report,
            policy
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "LLM INVESTIGATION REPORT"
    )

    print(
        "=" * 70
    )

    print(
        report
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "POLICY VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            validation,
            indent=4
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()