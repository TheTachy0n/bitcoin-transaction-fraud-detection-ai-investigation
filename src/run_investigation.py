# ============================================================
# STEP 18 — FINAL INVESTIGATION RUNNER
# ELLIPTIC BITCOIN FRAUD DETECTION
# ============================================================

from pathlib import Path
import sys
import json


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT PIPELINE COMPONENTS
# ============================================================

from ai_investigator import AIInvestigator
from policy_engine import PolicyEngine
from llm_engine import LLMEngine
from audit_logger import AuditLogger


# ============================================================
# FINAL INVESTIGATION RUNNER
# ============================================================

class InvestigationRunner:

    def __init__(self):

        print("=" * 70)
        print("FINAL FRAUD INVESTIGATION PIPELINE")
        print("=" * 70)

        print("\nInitializing components...")

        # ----------------------------------------------------
        # AI Investigator
        # ----------------------------------------------------

        self.investigator = AIInvestigator()

        # ----------------------------------------------------
        # Policy Engine
        # ----------------------------------------------------

        self.policy_engine = PolicyEngine()

        # ----------------------------------------------------
        # LLM Engine
        # ----------------------------------------------------

        self.llm_engine = LLMEngine()

        # ----------------------------------------------------
        # Audit Logger
        # ----------------------------------------------------

        self.audit_logger = AuditLogger(
            PROJECT_ROOT
            / "results"
            / "audit_log.json"
        )

        print(
            "\nFinal investigation pipeline ready."
        )


    # ========================================================
    # RUN INVESTIGATION
    # ========================================================

    def run(self, tx_id):

        print(
            "\n" + "=" * 70
        )

        print(
            f"STARTING INVESTIGATION: {tx_id}"
        )

        print(
            "=" * 70
        )


        # ====================================================
        # STEP 1 — AI INVESTIGATOR
        # ====================================================

        print(
            "\n[1/4] AI INVESTIGATOR"
        )

        investigation = (
            self.investigator.investigate(
                tx_id
            )
        )


        if not investigation.get(
            "success",
            False
        ):

            print(
                "\nInvestigation failed."
            )

            print(
                investigation.get(
                    "error",
                    "Unknown error"
                )
            )

            return investigation


        # ====================================================
        # STEP 2 — POLICY ENGINE
        # ====================================================

        print(
            "\n[2/4] POLICY ENGINE"
        )

        policy = (
            self.policy_engine.evaluate(
                investigation
            )
        )


        if not policy.get(
            "success",
            False
        ):

            print(
                "\nPolicy evaluation failed."
            )

            return {

                "success": False,

                "error":
                    policy.get(
                        "error",
                        "Policy evaluation failed."
                    ),

                "investigation":
                    investigation
            }


        # ====================================================
        # STEP 3 — LLM INVESTIGATION
        # ====================================================

        print(
            "\n[3/4] LLM INVESTIGATION"
        )


        llm_input = {

            "investigation":
                investigation,

            "policy_decision":
                policy
        }


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Let LLMEngine handle Groq + fallback internally.
        #
        # We do NOT access llm_available here.
        # ----------------------------------------------------

        try:

            llm_report = (
                self.llm_engine
                .generate_investigation(
                    llm_input
                )
            )

        except Exception as e:

            print(
                "\nWARNING: LLM engine failed."
            )

            print(
                f"Reason: {e}"
            )

            print(
                "Attempting deterministic fallback..."
            )

            try:

                llm_report = (
                    self.llm_engine
                    ._generate_fallback(
                        llm_input
                    )
                )

            except Exception as fallback_error:

                return {

                    "success": False,

                    "error":
                        (
                            "Both LLM generation and "
                            "fallback generation failed."
                        ),

                    "llm_error":
                        str(e),

                    "fallback_error":
                        str(fallback_error),

                    "investigation":
                        investigation,

                    "policy_decision":
                        policy
                }


        # ====================================================
        # DETERMINE LLM / FALLBACK STATUS
        # ====================================================

        # ----------------------------------------------------
        # We determine this from the LLM Engine state when
        # available, but never require the attribute to exist.
        # ----------------------------------------------------

        llm_available = getattr(
            self.llm_engine,
            "llm_available",
            None
        )


        if llm_available is True:

            llm_used = True
            fallback_used = False

        elif llm_available is False:

            llm_used = False
            fallback_used = True

        else:

            # Older LLMEngine versions may not expose
            # llm_available. In that case, determine status
            # from the presence of the Groq client.

            client = getattr(
                self.llm_engine,
                "client",
                None
            )

            if client is not None:

                llm_used = True
                fallback_used = False

            else:

                llm_used = False
                fallback_used = True


        # ====================================================
        # POLICY CONSISTENCY CHECK
        # ====================================================

        validation = (
            self.llm_engine
            .validate_policy_consistency(
                llm_report,
                policy
            )
        )


        # ====================================================
        # STEP 4 — AUDIT LOGGER
        # ====================================================

        print(
            "\n[4/4] AUDIT LOGGER"
        )


        evidence = investigation.get(
            "evidence",
            {}
        )


        # ----------------------------------------------------
        # Transaction evidence
        # ----------------------------------------------------

        transaction_evidence = (
            evidence.get(
                "transaction",
                {}
            )
        )


        # ----------------------------------------------------
        # Risk assessment
        # ----------------------------------------------------

        risk_assessment = (
            evidence.get(
                "risk_assessment",
                {}
            )
        )


        # ----------------------------------------------------
        # Model predictions
        # ----------------------------------------------------

        model_predictions = (
            evidence.get(
                "model_predictions",
                {}
            )
        )


        # ----------------------------------------------------
        # Model agreement
        # ----------------------------------------------------

        model_agreement = (
            evidence.get(
                "model_agreement",
                "UNKNOWN"
            )
        )


        # ----------------------------------------------------
        # Graph evidence
        # ----------------------------------------------------

        graph_package = (
            evidence.get(
                "graph_evidence",
                {}
            )
        )


        graph_evidence = (
            graph_package.get(
                "graph_analysis",
                {}
            )
        )


        # ----------------------------------------------------
        # SHAP evidence
        # ----------------------------------------------------

        shap_evidence = (
            evidence.get(
                "shap_evidence",
                {}
            )
        )


        # ----------------------------------------------------
        # RAG evidence
        # ----------------------------------------------------

        rag_evidence = (
            evidence.get(
                "rag_evidence",
                []
            )
        )


        # ----------------------------------------------------
        # Tools called
        # ----------------------------------------------------

        tools_called = (
            evidence
            .get(
                "metadata",
                {}
            )
            .get(
                "tools_called",
                []
            )
        )


        # ====================================================
        # LLM STATUS
        # ====================================================

        llm_status = {

            "provider":
                "groq"
                if llm_used
                else "deterministic_fallback",

            "model":
                getattr(
                    self.llm_engine,
                    "model",
                    None
                )
                if llm_used
                else None,

            "used":
                llm_used,

            "fallback":
                fallback_used,

            "policy_validation":
                validation
        }


        # ====================================================
        # FINAL RECOMMENDATION
        # ====================================================

        final_recommendation = (
            policy.get(
                "decision",
                "UNKNOWN"
            )
        )


        # ====================================================
        # METADATA
        # ====================================================

        metadata = {

            "pipeline_version":
                "1.0",

            "runner":
                "InvestigationRunner",

            "transaction_id":
                str(tx_id)
        }


        # ====================================================
        # WRITE AUDIT RECORD
        # ====================================================

        audit_record = (
            self.audit_logger
            .log_investigation(

                transaction_id=tx_id,

                risk_assessment=
                    risk_assessment,

                model_predictions=
                    model_predictions,

                model_agreement=
                    model_agreement,

                tools_called=
                    tools_called,

                transaction_evidence=
                    transaction_evidence,

                graph_evidence=
                    graph_evidence,

                shap_evidence=
                    shap_evidence,

                rag_evidence=
                    rag_evidence,

                llm_status=
                    llm_status,

                policy_decision=
                    policy,

                final_recommendation=
                    final_recommendation,

                metadata=
                    metadata
            )
        )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {

            "success":
                True,

            "transaction_id":
                int(tx_id),

            "investigation":
                investigation,

            "policy_decision":
                policy,

            "llm": {

                "report":
                    llm_report,

                "used":
                    llm_used,

                "fallback":
                    fallback_used,

                "policy_validation":
                    validation
            },

            "audit_record":
                audit_record
        }


        return result


# ============================================================
# PRINT FINAL RESULT
# ============================================================

def print_final_result(result):

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL INVESTIGATION RESULT"
    )

    print(
        "=" * 70
    )


    if not result.get(
        "success",
        False
    ):

        print(
            "\nINVESTIGATION FAILED"
        )

        print(
            result.get(
                "error",
                "Unknown error"
            )
        )

        return


    policy = result.get(
        "policy_decision",
        {}
    )

    llm = result.get(
        "llm",
        {}
    )


    print(
        f"\nTransaction ID : "
        f"{result.get('transaction_id')}"
    )

    print(
        f"Decision       : "
        f"{policy.get('decision')}"
    )

    print(
        f"Priority       : "
        f"{policy.get('priority')}"
    )

    print(
        f"Manual Review  : "
        f"{policy.get('requires_manual_review')}"
    )

    print(
        f"LLM Used       : "
        f"{llm.get('used')}"
    )

    print(
        f"Fallback Used  : "
        f"{llm.get('fallback')}"
    )


    print(
        "\nPolicy validation:"
    )

    print(
        json.dumps(
            llm.get(
                "policy_validation",
                {}
            ),
            indent=4
        )
    )


    print(
        "\n" + "-" * 70
    )

    print(
        "AI INVESTIGATION REPORT"
    )

    print(
        "-" * 70
    )

    print(
        llm.get(
            "report",
            "No report generated."
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "AUDIT LOG UPDATED"
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate transaction ID argument
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/run_investigation.py <TRANSACTION_ID>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src/run_investigation.py 71987809"
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Parse transaction ID
    # --------------------------------------------------------

    try:

        tx_id = int(
            sys.argv[1]
        )

    except ValueError:

        print(
            "\nERROR: Transaction ID must be an integer."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    runner = InvestigationRunner()

    result = runner.run(
        tx_id
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_final_result(
        result
    )


    # --------------------------------------------------------
    # Save final result
    # --------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "results"
        / "final_investigation.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            default=str
        )


    print(
        "\nFinal result saved:"
    )

    print(
        output_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()