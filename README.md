# AI-Powered Fraud Detection and Investigation System

An end-to-end Bitcoin transaction fraud detection and investigation system that combines transaction-level machine learning, graph-based analysis, explainable AI, retrieval-augmented generation, and policy-based decision making.

Traditional fraud detection systems typically produce a prediction or risk score. This project goes a step further by investigating the reasons behind a suspicious transaction and combining multiple sources of evidence into an auditable investigation workflow.

The system uses XGBoost for transaction-level fraud prediction and GraphSAGE for graph-based analysis. Their predictions are combined through a hybrid risk engine, which is then supported by SHAP explanations, connected transaction analysis, transaction history, and a RAG-based knowledge base. An AI Investigator synthesizes this evidence before a policy engine produces an operational recommendation.

---

## Overview

The system is designed around the following workflow:

```text
Bitcoin Transaction
        |
        v
Feature Processing
        |
   +----+----+
   |         |
   v         v
XGBoost   GraphSAGE
   |         |
   +----+----+
        |
        v
   Hybrid Fusion
        |
        v
    Risk Engine
        |
        v
AI Investigator
        |
   +----+----+----+
   |         |    |
   v         v    v
 SHAP     Graph   RAG
   |         |    |
   +----+----+----+
        |
        v
 Evidence Package
        |
        v
 Policy Engine
        |
        v
 Investigation Report
        |
        v
    Audit Log
```

The main objective is to move from simple fraud classification toward an investigation-oriented system that can provide evidence and explain the reasoning behind a risk assessment.

---

## Problem Statement

A fraud detection model can identify transactions that appear suspicious, but an investigator typically needs more information before taking action.

Important questions include:

* Why was the transaction flagged?
* Which transaction features contributed to the prediction?
* Is the transaction connected to other suspicious transactions?
* Do transaction-level and graph-level models agree?
* What additional domain knowledge supports the assessment?
* What action should be taken?
* Can the investigation be reviewed later?

This project addresses these questions by combining machine learning, graph analysis, explainability, retrieval, and rule-based decision making into a single pipeline.

---

## Dataset

The project uses the Elliptic Bitcoin transaction dataset.

The dataset contains approximately:

* 203,769 transactions
* 234,355 transaction relationships
* 166 transaction features
* Licit, illicit, and unknown transaction labels

The transactions form a graph where connected transactions represent relationships between entities in the Bitcoin transaction network.

### Temporal Evaluation

The dataset is organized into temporal timesteps. The model evaluation uses a temporal split rather than randomly mixing transactions across training and testing.

This reduces the risk of temporal leakage and provides a more realistic evaluation setting for fraud detection on future transactions.

---

## Machine Learning Models

### XGBoost

XGBoost provides the transaction-level fraud prediction.

It analyzes engineered transaction features and identifies patterns associated with illicit activity.

Final held-out test performance:

| Metric    | XGBoost |
| --------- | ------: |
| Accuracy  |  97.27% |
| Precision |  87.78% |
| Recall    |  47.55% |
| F1-score  |  61.69% |
| ROC-AUC   |  84.82% |
| PR-AUC    |  55.50% |

### GraphSAGE

GraphSAGE provides graph-based fraud predictions by incorporating information from connected transactions.

Final held-out test performance:

| Metric    | GraphSAGE |
| --------- | --------: |
| Accuracy  |    94.59% |
| Precision |    41.59% |
| Recall    |    42.40% |
| F1-score  |    41.99% |
| ROC-AUC   |    80.69% |
| PR-AUC    |    42.69% |

---

## Hybrid Model Fusion

The system combines the predictions from XGBoost and GraphSAGE rather than relying on a single model.

The final fusion configuration uses a 90/10 weighting between the model outputs.

Final held-out test performance:

| Metric    | Hybrid Fusion |
| --------- | ------------: |
| Accuracy  |        97.30% |
| Precision |        88.24% |
| Recall    |        47.79% |
| F1-score  |        62.00% |
| ROC-AUC   |        83.16% |
| PR-AUC    |        55.26% |

The individual model outputs remain available to the downstream investigation system, allowing model agreement and disagreement to be considered separately.

---

## Model Agreement

Model disagreement is explicitly retained rather than discarded.

On the validation set, the two models agreed on approximately 82.91% of transactions.

The validation set contained 7,829 transactions:

| Model Behaviour               | Transactions |
| ----------------------------- | -----------: |
| XGBoost LOW + GraphSAGE LOW   |        6,020 |
| XGBoost LOW + GraphSAGE HIGH  |        1,159 |
| XGBoost HIGH + GraphSAGE HIGH |          574 |
| XGBoost HIGH + GraphSAGE LOW  |           76 |

This allows the system to distinguish between strong corroboration, transaction-level evidence, graph-level evidence, and model disagreement.

---

## Risk Engine

The Risk Engine combines model outputs and supporting evidence into operational risk categories.

The final risk-engine evaluation processed 8,841 transactions.

| Risk Level | Transactions | Percentage |
| ---------- | -----------: | ---------: |
| LOW        |        8,553 |     96.74% |
| MEDIUM     |           67 |      0.76% |
| HIGH       |          221 |      2.50% |

The system also categorizes the underlying evidence associated with each transaction.

Examples include:

* Low signal
* Graph-only signal
* Strong transaction signal
* Strong corroboration

This provides more context than a binary fraud/non-fraud prediction.

---

## Explainability

The system uses SHAP (SHapley Additive exPlanations) to identify the transaction features that contribute to the XGBoost prediction.

Instead of providing only a fraud probability, the investigation pipeline can identify which features contributed positively or negatively to the prediction.

SHAP is treated as supporting evidence rather than proof that a transaction is fraudulent.

---

## Graph Investigation

The Graph Tool allows the system to inspect connected transactions and graph-level relationships.

This provides additional context when a transaction appears suspicious because of its relationship with other transactions rather than solely because of its individual features.

Graph-based evidence is passed to the investigation layer alongside the model predictions and SHAP explanations.

---

## Retrieval-Augmented Generation

The system includes a RAG component that retrieves relevant information from a project knowledge base during an investigation.

The retrieval process follows:

```text
Investigation Query
        |
        v
Knowledge Base Retrieval
        |
        v
Relevant Evidence
        |
        v
AI Investigator
```

RAG provides additional domain context to the investigation and helps ground generated reports in retrieved information.

---

## AI Investigator

The AI Investigator acts as the orchestration layer between the predictive models and the final investigation.

For a suspicious transaction, it can combine:

```text
Model Predictions
        +
SHAP Explanations
        +
Connected Transactions
        +
Transaction History
        +
RAG Evidence
        +
Risk Assessment
```

The resulting evidence package is used to generate a structured investigation report.

The LLM is used primarily for evidence synthesis and reporting rather than as the underlying fraud classifier.

---

## Policy Engine

The Policy Engine converts the risk assessment and investigation evidence into an operational recommendation.

A simplified policy flow is:

```text
LOW     -> Monitor / Allow
MEDIUM  -> Review
HIGH    -> Escalate
```

Separating the policy layer from the LLM ensures that operational decisions are based on explicit rules rather than being determined solely by generated text.

---

## Audit Logging

Each investigation can produce a structured audit record containing information such as:

* Transaction ID
* XGBoost prediction
* GraphSAGE prediction
* Fusion result
* Risk level
* Evidence category
* Investigation findings
* Retrieved evidence
* Policy decision
* Investigation output

This provides traceability for later review.

---

## Application

The project includes an interactive Streamlit application that exposes the investigation workflow.

The application provides access to:

* Risk assessment
* XGBoost prediction
* GraphSAGE prediction
* Hybrid result
* Model agreement
* SHAP explanations
* Connected transactions
* Retrieved knowledge
* AI investigation
* Policy recommendation
* Audit information

The project also includes a CLI investigation pipeline for running investigations without the Streamlit interface.

---

## Example Investigation

A representative high-risk investigation uses transaction `71987809`.

The investigation follows:

```text
Transaction 71987809
        |
        v
XGBoost + GraphSAGE
        |
        v
Hybrid Fusion
        |
        v
Risk Engine
        |
        v
HIGH Risk
        |
        +------------------+
        |                  |
        v                  v
      SHAP               Graph
        |                  |
        +--------+---------+
                 |
                 v
                RAG
                 |
                 v
         AI Investigator
                 |
                 v
          Policy Engine
                 |
                 v
             ESCALATE
                 |
                 v
            Audit Log
```

Additional transactions are used to demonstrate medium-risk behaviour, low-risk behaviour, and model disagreement.

---

## Results

### Final Held-Out Test Results

| Model     | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| --------- | -------: | --------: | -----: | -----: | ------: |
| XGBoost   |   97.27% |    87.78% | 47.55% | 61.69% |  84.82% |
| GraphSAGE |   94.59% |    41.59% | 42.40% | 41.99% |  80.69% |
| Hybrid    |   97.30% |    88.24% | 47.79% | 62.00% |  83.16% |

### System-Level Results

* 8,841 transactions processed by the final Risk Engine
* 221 transactions classified as HIGH risk
* 82.91% model agreement on the validation set
* 17.09% model disagreement retained for investigation
* 1,328 graph-signal-only cases
* 246 strong-corroboration cases

These results represent the final project pipeline and are separate from results obtained in earlier versions of the GNN project.

---

## Project Structure

```text
bitcoin-transaction-fraud-detection-ai-investigation/
│
├── app.py
├── README.md
├── .gitignore
│
├── knowledge_base/
│   ├── fraud_indicators.txt
│   ├── investigation_procedures.txt
│   ├── model_interpretation.txt
│   └── risk_policy.txt
│
├── models/
│   ├── graphsage_best.pt
│   ├── hybrid_fusion.pkl
│   └── xgboost_final.pkl
│
├── notebooks/
│   └── compare_gnn_models.ipynb
│
├── results/
│   ├── audit_log.json
│   ├── final_risk_engine.csv
│   ├── final_test_metrics.csv
│   ├── hybrid_fusion_metrics.csv
│   ├── investigation_evidence.csv
│   ├── model_agreement_validation.csv
│   └── representative_test_cases.json
│
└── src/
    ├── models/
    ├── ai_investigator.py
    ├── align_prediction.py
    ├── analyze_fusion.py
    ├── analyze_graph_features.py
    ├── analyze_graph_temporal.py
    ├── analyze_model_agreement.py
    ├── analyze_xgboost_features.py
    ├── audit_logger.py
    ├── build_graph.py
    ├── build_historical_graph_features.py
    ├── calibrate_models.py
    ├── data_loader.py
    ├── evaluate_fusion.py
    ├── evaluate_fusion_weights.py
    ├── evaluate_graph_features.py
    ├── evaluate_hybrid_test.py
    ├── evidence_package.py
    ├── explain_xgboost.py
    ├── final_model_selection.py
    ├── final_risk_engine.py
    ├── final_test_evaluation.py
    ├── generate_investigation_evidence.py
    ├── graph_tool.py
    ├── hybrid_risk_fusion.py
    ├── inspect_data.py
    ├── llm_engine.py
    ├── optimize_thresholds.py
    ├── optimize_xgboost.py
    ├── policy_engine.py
    ├── rag_retriever.py
    ├── risk_engine.py
    ├── risk_engine_v2.py
    ├── run_investigation.py
    ├── select_test_transactions.py
    ├── split.py
    ├── test_investigation_suite.py
    ├── train_baseline.py
    ├── train_final_xgboost.py
    ├── train_graphsage.py
    ├── train_xgboost.py
    ├── transaction_tool.py
    ├── tune_xgboost.py
    └── verify_graph.py
```


---

## Installation

Clone the repository:

```bash
git clone https://github.com/TheTachy0n/bitcoin-transaction-fraud-detection-ai-investigation.git
cd bitcoin-transaction-fraud-detection-ai-investigation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Configure any required API keys using environment variables. API keys and other secrets should not be committed to the repository.

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

Select a supported transaction ID and run the investigation pipeline.

---

## Demo Transactions

| Transaction ID | Demonstration                      |
| -------------- | ---------------------------------- |
| `71987809`     | Strong corroborated high-risk case |
| `3295818`      | Medium / conflicting behaviour     |
| `73020281`     | Low-risk case                      |
| `72689707`     | Model disagreement                 |

---

## Limitations

* The system is evaluated using the Elliptic Bitcoin transaction dataset and does not represent the full complexity of a production cryptocurrency monitoring environment.
* The dataset contains a substantial number of transactions without known labels.
* Model performance may change when applied to different transaction distributions.
* GraphSAGE performance depends on the structure and quality of the available transaction graph.
* RAG quality depends on the coverage and quality of the knowledge base.
* LLM-generated reports should be treated as decision-support outputs rather than definitive ground truth.
* A production deployment would require additional monitoring, security controls, latency optimization, and model governance.

---

## Future Improvements

Potential future improvements include:

* Real-time transaction-stream processing
* Larger and continuously updated transaction graphs
* Temporal graph neural networks
* Improved class-imbalance handling
* Model drift detection
* Human-in-the-loop analyst feedback
* Expanded fraud knowledge bases
* Automated case management integration
* Production-scale inference
* Continuous model and policy evaluation

---

## Conclusion

This project combines machine learning, graph analysis, explainability, retrieval, and AI-assisted investigation into a single fraud detection workflow.

Rather than stopping at a fraud prediction, the system attempts to provide an explanation of **why a transaction is suspicious, what evidence supports the assessment, and what action should be taken next**.

The result is an investigation-oriented fraud detection system with explicit risk assessment, evidence generation, policy decisions, and auditability.

---

## Disclaimer

This project is a research prototype developed using the Elliptic Bitcoin transaction dataset. Risk assessments and investigation reports are intended as decision-support outputs and should not be interpreted as definitive proof of criminal activity.
