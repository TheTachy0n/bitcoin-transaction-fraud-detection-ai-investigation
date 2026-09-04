# ============================================================
# END-TO-END INVESTIGATION SUITE
# AI INVESTIGATOR + POLICY ENGINE
# ============================================================

from pathlib import Path
import sys
import json
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT COMPONENTS
# ============================================================

from ai_investigator import AIInvestigator
from policy_engine import PolicyEngine


# ============================================================
# PATHS
# ============================================================

TEST_CASE_PATH = (
    PROJECT_ROOT
    / "results"
    / "representative_test_cases.json"
)

RISK_ENGINE_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_risk_engine.csv"
)


# ============================================================
# LOAD TEST CASES
# ============================================================

with open(
    TEST_CASE_PATH,
    "r",
    encoding="utf-8"
) as f:

    test_cases = json.load(f)


risk_df = pd.read_csv(
    RISK_ENGINE_PATH
)

risk_lookup = (
    risk_df
    .set_index("txId")
)


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

print("=" * 70)
print("INVESTIGATION TEST SUITE")
print("=" * 70)

print("\nInitializing AI Investigator...")

investigator = AIInvestigator()

print("\nInitializing Policy Engine...")

policy_engine = PolicyEngine()


# ============================================================
# RUN TESTS
# ============================================================

results = []


for index, case in enumerate(
    test_cases,
    start=1
):

    tx_id = int(
        case["transaction_id"]
    )

    category = case["category"]


    print(
        "\n" + "-" * 70
    )

    print(
        f"TEST {index}/{len(test_cases)}"
    )

    print(
        f"Transaction: {tx_id}"
    )

    print(
        f"Expected category: {category}"
    )


    # --------------------------------------------------------
    # Get model/risk information
    # --------------------------------------------------------

    row = risk_lookup.loc[tx_id]


    expected_risk = str(
        row["risk_level"]
    )

    expected_agreement = str(
        row["agreement_category"]
    )

    expected_risk_score = float(
        row["risk_score"]
    )


    # --------------------------------------------------------
    # AI INVESTIGATOR
    # --------------------------------------------------------

    try:

        investigation = (
            investigator.investigate(
                tx_id
            )
        )

    except Exception as e:

        print(
            f"INVESTIGATOR ERROR: {e}"
        )

        results.append(
            {
                "transaction_id": tx_id,
                "category": category,
                "status": "FAIL",
                "error": str(e)
            }
        )

        continue


    if not investigation.get(
        "success",
        False
    ):

        print(
            "INVESTIGATION FAILED"
        )

        results.append(
            {
                "transaction_id": tx_id,
                "category": category,
                "status": "FAIL",
                "error":
                    investigation.get(
                        "error",
                        "Unknown error"
                    )
            }
        )

        continue


    # --------------------------------------------------------
    # POLICY ENGINE
    # --------------------------------------------------------

    try:

        policy = (
            policy_engine.evaluate(
                investigation
            )
        )

    except Exception as e:

        print(
            f"POLICY ERROR: {e}"
        )

        results.append(
            {
                "transaction_id": tx_id,
                "category": category,
                "status": "FAIL",
                "error": str(e)
            }
        )

        continue


    if not policy.get(
        "success",
        False
    ):

        print(
            "POLICY EVALUATION FAILED"
        )

        results.append(
            {
                "transaction_id": tx_id,
                "category": category,
                "status": "FAIL",
                "error":
                    policy.get(
                        "error",
                        "Unknown policy error"
                    )
            }
        )

        continue


    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    actual_decision = policy.get(
        "decision",
        "UNKNOWN"
    )

    actual_priority = policy.get(
        "priority",
        "UNKNOWN"
    )

    manual_review = policy.get(
        "requires_manual_review",
        False
    )

    actual_risk = policy.get(
        "risk_score",
        None
    )


    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    risk_match = (
        str(
            policy.get(
                "risk_level",
                ""
            )
        ).upper()
        ==
        expected_risk.upper()
    )


    score_match = False

    if actual_risk is not None:

        score_match = (
            abs(
                float(actual_risk)
                -
                expected_risk_score
            )
            < 1e-6
        )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Risk: {expected_risk}"
    )

    print(
        f"Risk score: {expected_risk_score:.6f}"
    )

    print(
        f"Agreement: {expected_agreement}"
    )

    print(
        f"Decision: {actual_decision}"
    )

    print(
        f"Priority: {actual_priority}"
    )

    print(
        f"Manual review: {manual_review}"
    )

    print(
        f"Risk level match: {risk_match}"
    )

    print(
        f"Risk score match: {score_match}"
    )


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "transaction_id": tx_id,
            "category": category,
            "risk_level": expected_risk,
            "risk_score": expected_risk_score,
            "agreement_category":
                expected_agreement,
            "decision":
                actual_decision,
            "priority":
                actual_priority,
            "manual_review":
                manual_review,
            "risk_level_match":
                risk_match,
            "risk_score_match":
                score_match,
            "status": "PASS"
                if risk_match and score_match
                else "CHECK"
        }
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    PROJECT_ROOT
    / "results"
    / "investigation_test_results.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n\n" + "=" * 70
)

print(
    "TEST SUITE SUMMARY"
)

print(
    "=" * 70
)


if not results_df.empty:

    print(
        f"\nTotal tests: "
        f"{len(results_df)}"
    )

    print(
        f"Passed: "
        f"{(results_df['status'] == 'PASS').sum()}"
    )

    print(
        f"Need review: "
        f"{(results_df['status'] == 'CHECK').sum()}"
    )

    print(
        f"Failed: "
        f"{(results_df['status'] == 'FAIL').sum()}"
    )


    print(
        "\nDecision distribution:"
    )

    print(
        results_df[
            "decision"
        ].value_counts()
    )


    print(
        "\nRisk-level distribution:"
    )

    print(
        results_df[
            "risk_level"
        ].value_counts()
    )


    print(
        "\nAgreement distribution:"
    )

    print(
        results_df[
            "agreement_category"
        ].value_counts()
    )


print(
    f"\nResults saved to:"
)

print(
    output_path
)

print(
    "\n" + "=" * 70
)