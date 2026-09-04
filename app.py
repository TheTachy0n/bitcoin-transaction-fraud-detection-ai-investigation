
import streamlit as st
import sys
import os
import re

# =========================================================
# PATH SETUP
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fraud Investigation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #777;
        margin-bottom: 2rem;
    }

    .risk-box {
        border: 1px solid #d9d9d9;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin: 10px 0 25px 0;
    }

    .risk-score {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .feature-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid #eee;
    }

    .feature-name {
        font-weight: 600;
    }

    .feature-value {
        font-family: monospace;
        font-weight: 600;
    }

    .rag-source {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .rag-score {
        font-size: 0.8rem;
        color: #777;
        margin-bottom: 8px;
    }

    .rag-excerpt {
        padding: 12px;
        border-left: 3px solid #888;
        background: rgba(128, 128, 128, 0.06);
        border-radius: 4px;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .footer {
        color: #888;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def format_probability(value):
    try:
        value = float(value)
        return f"{value * 100:.3f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_number(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def extract_shap_features(shap):
    """Extract standardized SHAP evidence."""

    if not isinstance(shap, dict):
        return []

    features = shap.get("top_features", [])

    if not isinstance(features, list):
        return []

    return features


def normalize_rag_item(item):
    """Normalize standardized RAG evidence."""

    if not isinstance(item, dict):
        return None

    return {
        "source": item.get(
            "source",
            "Knowledge Base",
        ),
        "score": item.get("score"),
        "text": item.get("text", ""),
    }


def get_rag_excerpt(text, max_chars=650):
    """
    Convert the full retrieved document into a concise
    evidence excerpt for the dashboard.
    """

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    keywords = [
        "HIGH RISK",
        "MODEL CORROBORATION",
        "GRAPH EVIDENCE POLICY",
        "INVESTIGATION REQUIREMENT",
        "STEP 2 — REVIEW MODEL EVIDENCE",
        "STEP 3 — REVIEW SHAP EVIDENCE",
        "STEP 4 — REVIEW GRAPH EVIDENCE",
        "MODEL-BASED RISK",
        "SHAP INDICATORS",
    ]

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    selected = []

    for paragraph in paragraphs:

        upper = paragraph.upper()

        if any(
            keyword in upper
            for keyword in keywords
        ):
            selected.append(paragraph)

    if selected:

        excerpt = "\n\n".join(
            selected[:2]
        )

    else:

        excerpt = "\n\n".join(
            paragraphs[:2]
        )

    if len(excerpt) > max_chars:

        excerpt = (
            excerpt[:max_chars]
            .rsplit(" ", 1)[0]
            + "..."
        )

    return excerpt


def clean_report_text(report_text):
    """
    Remove common formatting artifacts from generated
    investigation reports.
    """

    if not report_text:
        return ""

    clean_report = str(report_text)

    # Remove localhost SVG anchor artifacts.
    clean_report = re.sub(
        r"\[svg\]\(http://localhost:[^)]+\)",
        "",
        clean_report,
    )

    # Remove excessive blank lines.
    clean_report = re.sub(
        r"\n{3,}",
        "\n\n",
        clean_report,
    )

    return clean_report.strip()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🔍 AI Fraud Investigation System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
    Multi-model fraud detection with graph intelligence,
    explainability, retrieval-augmented evidence,
    policy reasoning, and AI investigation.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# INPUT
# =========================================================

input_col, button_col = st.columns([5, 1])

with input_col:

    transaction_id = st.text_input(
        "Transaction ID",
        placeholder="Enter transaction ID, e.g. 71987809",
    )

with button_col:

    st.write("")
    st.write("")

    investigate = st.button(
        "🔎 Investigate",
        type="primary",
        use_container_width=True,
    )

# =========================================================
# INVESTIGATION
# =========================================================

if investigate:

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not transaction_id.strip():

        st.warning(
            "Please enter a transaction ID."
        )

        st.stop()

    try:

        tx_id = int(
            transaction_id.strip()
        )

    except ValueError:

        st.error(
            "Transaction ID must be numeric."
        )

        st.stop()

    # -----------------------------------------------------
    # Run investigation
    # -----------------------------------------------------

    with st.spinner(
        "Running AI investigation..."
    ):

        try:

            # =================================================
            # IMPORT COMPONENTS
            # =================================================

            from ai_investigator import (
                AIInvestigator
            )

            from policy_engine import (
                PolicyEngine
            )

            from llm_engine import (
                LLMEngine
            )

            from audit_logger import (
                AuditLogger
            )

            # =================================================
            # INITIALIZE
            # =================================================

            investigator = AIInvestigator()

            policy_engine = PolicyEngine()

            llm_engine = LLMEngine()

            audit_logger = AuditLogger(
                os.path.join(
                    PROJECT_ROOT,
                    "results",
                    "audit_log.json",
                )
            )

            # =================================================
            # STEP 1 — AI INVESTIGATOR
            # =================================================

            investigation = (
                investigator.investigate(
                    tx_id
                )
            )

            if not investigation.get(
                "success",
                False,
            ):

                st.error(
                    investigation.get(
                        "error",
                        f"Transaction {tx_id} "
                        "was not found.",
                    )
                )

                st.stop()

            # =================================================
            # STEP 2 — POLICY ENGINE
            # =================================================

            policy = (
                policy_engine.evaluate(
                    investigation
                )
            )

            if not policy.get(
                "success",
                True,
            ):

                st.error(
                    policy.get(
                        "error",
                        "Policy evaluation failed.",
                    )
                )

                st.stop()

            # =================================================
            # STEP 3 — LLM INVESTIGATION
            # =================================================

            llm_input = {
                "investigation": investigation,
                "policy_decision": policy,
            }

            try:

                report = (
                    llm_engine.generate_investigation(
                        llm_input
                    )
                )

            except Exception:

                report = (
                    llm_engine._generate_fallback(
                        llm_input
                    )
                )

            # =================================================
            # DETERMINE LLM / FALLBACK STATUS
            # =================================================

            llm_available = getattr(
                llm_engine,
                "llm_available",
                None,
            )

            if llm_available is True:

                llm_used = True
                fallback_used = False

            elif llm_available is False:

                llm_used = False
                fallback_used = True

            else:

                client = getattr(
                    llm_engine,
                    "client",
                    None,
                )

                if client is not None:

                    llm_used = True
                    fallback_used = False

                else:

                    llm_used = False
                    fallback_used = True

            # =================================================
            # STEP 4 — AUDIT LOGGER
            # =================================================

            audit_evidence = investigation.get(
                "evidence",
                {},
            )

            audit_record = (
                audit_logger.log_investigation(

                    transaction_id=tx_id,

                    risk_assessment=(
                        audit_evidence.get(
                            "risk_assessment",
                            {},
                        )
                    ),

                    model_predictions=(
                        audit_evidence.get(
                            "model_predictions",
                            {},
                        )
                    ),

                    model_agreement=(
                        audit_evidence.get(
                            "model_agreement",
                            "UNKNOWN",
                        )
                    ),

                    tools_called=(
                        audit_evidence
                        .get(
                            "metadata",
                            {},
                        )
                        .get(
                            "tools_called",
                            [],
                        )
                    ),

                    transaction_evidence=(
                        audit_evidence.get(
                            "transaction",
                            {},
                        )
                    ),

                    graph_evidence=(
                        audit_evidence
                        .get(
                            "graph_evidence",
                            {},
                        )
                        .get(
                            "graph_analysis",
                            {},
                        )
                    ),

                    shap_evidence=(
                        audit_evidence.get(
                            "shap_evidence",
                            {},
                        )
                    ),

                    rag_evidence=(
                        audit_evidence.get(
                            "rag_evidence",
                            [],
                        )
                    ),

                    llm_status={
                        "provider": (
                            "groq"
                            if llm_used
                            else "deterministic_fallback"
                        ),

                        "model": getattr(
                            llm_engine,
                            "model",
                            None,
                        ),

                        "used": llm_used,

                        "fallback": fallback_used,
                    },

                    policy_decision=policy,

                    final_recommendation=(
                        policy.get(
                            "decision",
                            "UNKNOWN",
                        )
                    ),

                    metadata={
                        "pipeline_version": "1.0",
                        "runner": "Streamlit",
                        "transaction_id": str(tx_id),
                    },
                )
            )

            # =================================================
            # EVIDENCE
            # =================================================

            evidence = investigation.get(
                "evidence",
                {},
            )

            transaction = evidence.get(
                "transaction",
                {},
            )

            risk = evidence.get(
                "risk_assessment",
                {},
            )

            models = evidence.get(
                "model_predictions",
                {},
            )

            graph_package = evidence.get(
                "graph_evidence",
                {},
            )

            graph = graph_package.get(
                "graph_analysis",
                {},
            )

            transaction_graph = (
                graph_package.get(
                    "transaction_graph_evidence",
                    {},
                )
            )

            shap = evidence.get(
                "shap_evidence",
                {},
            )

            rag = evidence.get(
                "rag_evidence",
                [],
            )

            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "Investigation completed successfully."
            )

            # =================================================
            # RISK HERO
            # =================================================

            risk_score = risk.get(
                "risk_score",
                0,
            )

            risk_level = risk.get(
                "risk_level",
                "UNKNOWN",
            )

            priority = policy.get(
                "priority",
                risk_level,
            )

            decision = policy.get(
                "decision",
                "UNKNOWN",
            )

            st.markdown(
                '<div class="section-title">'
                'Risk Assessment'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="risk-box">
                    <div class="risk-score">
                        {format_probability(risk_score)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"### {risk_level} RISK · {priority} PRIORITY"
            )

            st.markdown(
                f"Policy Decision: **{decision}**"
            )

            # =================================================
            # MODEL EVIDENCE
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Model Evidence'
                '</div>',
                unsafe_allow_html=True,
            )

            xgb = models.get(
                "xgboost_probability",
                0,
            )

            gnn = models.get(
                "graphsage_probability",
                0,
            )

            agreement = models.get(
                "agreement_category",
                "UNKNOWN",
            )

            disagreement = models.get(
                "model_disagreement",
                0,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "XGBoost",
                    format_probability(xgb),
                )

            with c2:

                st.metric(
                    "GraphSAGE",
                    format_probability(gnn),
                )

            with c3:

                st.metric(
                    "Model Agreement",
                    str(agreement),
                )

            with c4:

                st.metric(
                    "Disagreement",
                    format_number(
                        disagreement,
                        5,
                    ),
                )

            # =================================================
            # TRANSACTION DETAILS
            # =================================================

            with st.expander(
                "📄 Transaction Details",
                expanded=True,
            ):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.write(
                        "**Transaction ID**"
                    )

                    st.code(
                        str(
                            transaction.get(
                                "txId",
                                tx_id,
                            )
                        )
                    )

                with c2:

                    st.write(
                        "**Timestep**"
                    )

                    st.write(
                        transaction.get(
                            "timestep",
                            "Unknown",
                        )
                    )

                with c3:

                    st.write(
                        "**Dataset Label**"
                    )

                    st.write(
                        transaction.get(
                            "label",
                            "Unknown",
                        )
                    )

            # =================================================
            # GRAPH + SHAP
            # =================================================

            graph_col, shap_col = st.columns(2)

            # =================================================
            # GRAPH EVIDENCE
            # =================================================

            with graph_col:

                with st.expander(
                    "🕸️ Graph Evidence",
                    expanded=True,
                ):

                    neighbor_count = graph.get(
                        "neighbor_count",
                        transaction_graph.get(
                            "neighbor_count",
                            0,
                        ),
                    )

                    high_risk_neighbors = graph.get(
                        "high_risk_neighbor_count",
                        transaction_graph.get(
                            "high_risk_neighbor_count",
                            0,
                        ),
                    )

                    labeled_neighbors = graph.get(
                        "labeled_neighbor_count",
                        transaction_graph.get(
                            "labeled_neighbor_count",
                            0,
                        ),
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Neighbors",
                            neighbor_count,
                        )

                    with c2:

                        st.metric(
                            "High-Risk",
                            high_risk_neighbors,
                        )

                    with c3:

                        st.metric(
                            "Labeled",
                            labeled_neighbors,
                        )

                    neighbor_ids = graph.get(
                        "neighbors",
                        graph.get(
                            "neighbor_tx_ids",
                            transaction_graph.get(
                                "neighbor_tx_ids",
                                [],
                            ),
                        ),
                    )

                    if neighbor_ids:

                        st.write(
                            "**Connected Transactions**"
                        )

                        for neighbor in neighbor_ids:

                            if isinstance(
                                neighbor,
                                dict,
                            ):

                                neighbor_tx = (
                                    neighbor.get(
                                        "txId",
                                        "Unknown",
                                    )
                                )

                                neighbor_timestep = (
                                    neighbor.get(
                                        "timestep",
                                        "Unknown",
                                    )
                                )

                                neighbor_label = (
                                    neighbor.get(
                                        "label",
                                    )
                                )

                                neighbor_risk = (
                                    neighbor.get(
                                        "risk_score",
                                    )
                                )

                                if (
                                    neighbor_label == -1
                                    or neighbor_label is None
                                ):

                                    label_text = "Unknown"

                                else:

                                    label_text = str(
                                        neighbor_label
                                    )

                                if neighbor_risk is None:

                                    risk_text = (
                                        "Unavailable"
                                    )

                                else:

                                    risk_text = (
                                        format_probability(
                                            neighbor_risk
                                        )
                                    )

                                st.markdown(
                                    f"""
                                    **Transaction {neighbor_tx}**

                                    - **Timestep:** {neighbor_timestep}
                                    - **Label:** {label_text}
                                    - **Risk Score:** {risk_text}
                                    """
                                )

                            elif isinstance(
                                neighbor,
                                (int, str),
                            ):

                                st.markdown(
                                    f"""
                                    **Transaction {neighbor}**

                                    - **Details:** Unavailable
                                    """
                                )

                            else:

                                st.markdown(
                                    "**Transaction details unavailable**"
                                )

                    else:

                        st.info(
                            "No connected transactions "
                            "were identified."
                        )

                    st.caption(
                        "Unknown or unlabeled neighbors "
                        "are not treated as evidence "
                        "of legitimacy."
                    )

            # =================================================
            # SHAP EXPLAINABILITY
            # =================================================

            with shap_col:

                with st.expander(
                    "🔬 XGBoost Explainability",
                    expanded=True,
                ):

                    features = (
                        extract_shap_features(
                            shap
                        )
                    )

                    if features:

                        st.write(
                            "**Top fraud-risk "
                            "contributing features**"
                        )

                        for item in features[:5]:

                            if isinstance(
                                item,
                                dict,
                            ):

                                feature = (
                                    item.get(
                                        "feature"
                                    )
                                    or item.get(
                                        "feature_name"
                                    )
                                    or item.get(
                                        "name"
                                    )
                                    or "Unknown feature"
                                )

                                value = item.get(
                                    "shap_value"
                                )

                                direction = item.get(
                                    "direction",
                                    "",
                                )

                            else:

                                feature = str(
                                    item
                                )

                                value = None
                                direction = ""

                            if value is not None:

                                try:

                                    numeric_value = (
                                        float(value)
                                    )

                                    st.markdown(
                                        f"**{feature}** — "
                                        f"`{numeric_value:+.4f}`"
                                    )

                                    if direction:

                                        st.caption(
                                            direction.replace(
                                                "_",
                                                " "
                                            ).title()
                                        )

                                except (
                                    TypeError,
                                    ValueError,
                                ):

                                    st.write(
                                        f"**{feature}** — "
                                        f"{value}"
                                    )

                            else:

                                st.write(
                                    f"**{feature}**"
                                )

                    else:

                        st.info(
                            "No SHAP explanation "
                            "is available for "
                            "this transaction."
                        )

            # =================================================
            # RAG KNOWLEDGE BASE
            # =================================================

            with st.expander(
                "📚 Knowledge Base Evidence",
                expanded=True,
            ):

                rag_items = [
                    normalize_rag_item(item)
                    for item in rag
                ]

                valid_rag = [
                    item
                    for item in rag_items
                    if item is not None
                    and item["text"]
                ]

                if valid_rag:

                    for i, item in enumerate(
                        valid_rag,
                        1,
                    ):

                        source = item["source"]
                        score = item["score"]
                        text = item["text"]

                        if score is not None:

                            try:

                                score_text = (
                                    f"Retrieval relevance: "
                                    f"{float(score):.3f}"
                                )

                            except (
                                TypeError,
                                ValueError,
                            ):

                                score_text = (
                                    "Retrieval relevance: N/A"
                                )

                        else:

                            score_text = (
                                "Retrieval relevance: N/A"
                            )

                        st.markdown(
                            f"""
                            <div class="rag-source">
                                {i}. {source}
                            </div>

                            <div class="rag-score">
                                {score_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        excerpt = get_rag_excerpt(
                            text
                        )

                        if excerpt:

                            st.markdown(
                                f"""
                                <div class="rag-excerpt">
                                    {excerpt}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        else:

                            st.info(
                                "No relevant excerpt "
                                "available."
                            )

                        if i < len(valid_rag):

                            st.divider()

                else:

                    st.info(
                        "No knowledge-base evidence "
                        "was retrieved."
                    )

            # =================================================
            # POLICY DECISION
            # =================================================

            with st.expander(
                "⚖️ Policy Decision",
                expanded=True,
            ):

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Decision",
                        decision,
                    )

                with c2:

                    st.metric(
                        "Priority",
                        priority,
                    )

                with c3:

                    manual_review = policy.get(
                        "requires_manual_review",
                        False,
                    )

                    st.metric(
                        "Manual Review",
                        "YES"
                        if manual_review
                        else "NO",
                    )

                reasons = policy.get(
                    "reasons",
                    [],
                )

                if reasons:

                    st.write(
                        "**Decision Rationale**"
                    )

                    for reason in reasons:

                        st.write(
                            f"• {reason}"
                        )

            # =================================================
            # AI INVESTIGATION REPORT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '🤖 AI Investigation Report'
                '</div>',
                unsafe_allow_html=True,
            )

            if isinstance(
                report,
                dict,
            ):

                report_text = (
                    report.get(
                        "report"
                    )
                    or report.get(
                        "investigation_report"
                    )
                    or report.get(
                        "text"
                    )
                )

                if report_text:

                    clean_report = clean_report_text(
                        report_text
                    )

                    st.markdown(
                        clean_report
                    )

                else:

                    st.json(
                        report
                    )

            else:

                clean_report = clean_report_text(
                    report
                )

                st.markdown(
                    clean_report
                )

            # =================================================
            # TECHNICAL EVIDENCE
            # =================================================

            with st.expander(
                "🔎 Technical Evidence Package"
            ):

                st.json(
                    investigation
                )

            # =================================================
            # FOOTER
            # =================================================

            st.divider()

            st.markdown(
                """
                <div class="footer">
                Risk-based investigation system.
                Model predictions and graph evidence support
                investigation and do not by themselves constitute
                proof of fraud.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:

            st.error(
                f"Investigation failed: {str(e)}"
            )

