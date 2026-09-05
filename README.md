#  AI-Powered Fraud Detection & Investigation System

An end-to-end **Bitcoin transaction fraud detection and investigation system** that combines transaction-level machine learning, graph-based intelligence, explainable AI, retrieval-augmented generation, and policy-based decision making.

Unlike a conventional fraud classifier that only produces a prediction, the system is designed to answer a more useful question:

> **Why is this transaction suspicious, what evidence supports the decision, and what action should be taken?**

The system combines **XGBoost** for transaction-level fraud prediction and **GraphSAGE** for graph-based relationship analysis. Their outputs are fused by a risk engine, which then provides evidence to an AI Investigator using **SHAP explanations, transaction history, graph relationships, and RAG-based domain knowledge**. A policy engine converts the resulting risk assessment into an auditable action such as escalation or review.

---

##  Key Features

* **XGBoost transaction-level fraud detection**
* **GraphSAGE graph-based fraud detection**
* **Hybrid model fusion**
* **Risk scoring and risk categorization**
* **SHAP-based explainability**
* **Connected transaction / graph investigation**
* **Transaction history analysis**
* **Retrieval-Augmented Generation (RAG)**
* **AI Investigator for evidence synthesis**
* **Policy-based decision engine**
* **LLM with fallback handling**
* **Structured audit logging**
* **CLI investigation pipeline**
* **Interactive Streamlit application**

---

##  Problem Statement

Traditional transaction-monitoring systems often stop after identifying a suspicious transaction.

A fraud analyst, however, needs additional information:

* Why was the transaction flagged?
* Which transaction features contributed to the prediction?
* Is the transaction connected to other suspicious activity?
* Do transaction-level and graph-level models agree?
* What historical or domain evidence supports the assessment?
* What action should be taken?
* Can the entire decision be audited later?

This project addresses these requirements by combining **machine learning, graph intelligence, explainability, retrieval, and automated investigation** into a single pipeline.

---

#  System Architecture

```text
                    Bitcoin Transaction
                            │
                            ▼
                  ┌───────────────────┐
                  │ Feature Processing │
                  └─────────┬─────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          ┌─────────────┐       ┌─────────────┐
          │  XGBoost    │       │  GraphSAGE  │
          │ Transaction │       │    Graph    │
          │    Model    │       │    Model    │
          └──────┬──────┘       └──────┬──────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                  ┌───────────────────┐
                  │   Hybrid Fusion   │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │    Risk Engine    │
                  └─────────┬─────────┘
                            │
                            ▼
                   Suspicious Alert
                            │
                            ▼
                  ┌───────────────────┐
                  │  AI Investigator  │
                  └─────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          ┌──────┐      ┌────────┐      ┌─────┐
          │ SHAP │      │ Graph  │      │ RAG │
          └──────┘      └────────┘      └─────┘
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                   Evidence Package
                            │
                            ▼
                  ┌───────────────────┐
                  │   Policy Engine   │
                  └─────────┬─────────┘
                            ▼
                   Investigation Report
                            │
                            ▼
                     Audit Logger
```

---

#  Dataset

The project uses the **Elliptic Bitcoin transaction dataset**, a graph-based cryptocurrency transaction dataset containing transaction features, transaction relationships, and illicit/lic it transaction labels.

The dataset contains approximately:

* **203,769 transactions**
* **234,355 transaction relationships**
* **166 transaction features**
* Licit, illicit, and unknown transaction labels

The graph structure represents relationships between connected Bitcoin transactions.

### Temporal consideration

The dataset is organized across temporal timesteps. Model evaluation uses a **temporal split** rather than randomly mixing transactions across train and test sets.

This helps reduce temporal leakage and better reflects how a fraud detection system would operate on future transactions.

---

#  Machine Learning Models

## XGBoost

XGBoost provides the **transaction-level prediction**.

It analyzes engineered transaction features and identifies patterns associated with illicit activity.

### Final held-out test performance

| Metric    |    XGBoost |
| --------- | ---------: |
| Accuracy  | **97.27%** |
| Precision | **87.78%** |
| Recall    | **47.55%** |
| F1-score  | **61.69%** |
| ROC-AUC   | **84.82%** |
| PR-AUC    | **55.50%** |

---

## GraphSAGE

GraphSAGE provides a complementary **graph-level perspective**.

Instead of considering a transaction independently, it incorporates information from connected nodes in the transaction graph.

### Final held-out test performance

| Metric    |  GraphSAGE |
| --------- | ---------: |
| Accuracy  | **94.59%** |
| Precision | **41.59%** |
| Recall    | **42.40%** |
| F1-score  | **41.99%** |
| ROC-AUC   | **80.69%** |
| PR-AUC    | **42.69%** |

---

#  Hybrid Model Fusion

The system combines the transaction-level and graph-level predictions rather than relying on a single model.

The final fusion configuration uses a **90/10 weighting** between the model outputs.

### Held-out test performance

| Metric    | Hybrid Fusion |
| --------- | ------------: |
| Accuracy  |    **97.30%** |
| Precision |    **88.24%** |
| Recall    |    **47.79%** |
| F1-score  |    **62.00%** |
| ROC-AUC   |    **83.16%** |
| PR-AUC    |    **55.26%** |

The hybrid model provides a single risk signal while retaining the ability to inspect the individual model predictions.

---

#  Model Agreement

An important part of the system is that **model disagreement is not discarded**.

During validation, the two models agreed on approximately **82.91%** of transactions.

The validation set contained **7,829 transactions**:

| Model Behaviour               | Transactions |
| ----------------------------- | -----------: |
| XGBoost LOW + GraphSAGE LOW   |        6,020 |
| XGBoost LOW + GraphSAGE HIGH  |        1,159 |
| XGBoost HIGH + GraphSAGE HIGH |          574 |
| XGBoost HIGH + GraphSAGE LOW  |           76 |

This allows the investigation layer to distinguish between:

* Strong corroboration
* Transaction-level evidence
* Graph-level evidence
* Model disagreement

Rather than treating every prediction as equally certain, the system preserves this evidence for downstream investigation.

---

#  Risk Engine

The Risk Engine converts model outputs and supporting evidence into operational risk categories.

The final risk-engine evaluation processed **8,841 transactions**.

### Risk distribution

| Risk Level | Transactions | Percentage |
| ---------- | -----------: | ---------: |
| LOW        |        8,553 |     96.74% |
| MEDIUM     |           67 |      0.76% |
| HIGH       |          221 |      2.50% |

The system therefore identifies a relatively small subset of transactions for higher-priority investigation.

The risk engine also categorizes the underlying evidence, including:

* Low signal
* Graph-only signal
* Strong transaction signal
* Strong corroboration

This provides more context than a binary fraud/non-fraud output.

---

#  Explainability with SHAP

The system uses **SHAP (SHapley Additive exPlanations)** to identify which transaction features contributed to the model prediction.

Instead of displaying only:

```text
Fraud probability: 0.91
```

the investigation layer can provide information about the features that pushed the prediction higher or lower.

SHAP is treated as **supporting evidence**, not proof that a transaction is fraudulent.

---

#  Graph Investigation

Graph-based investigation allows the system to inspect the transaction's connected entities and relationships.

The Graph Tool can retrieve relevant connected transaction information and expose graph-level evidence to the investigation pipeline.

This is particularly useful when a transaction appears suspicious because of its **relationship with other transactions**, rather than solely because of its individual features.

---

#  Retrieval-Augmented Generation

The system incorporates a RAG layer to provide domain knowledge during investigation.

The retrieval component:

1. Receives an investigation query.
2. Searches the project knowledge base.
3. Retrieves relevant supporting information.
4. Passes the retrieved evidence to the investigation layer.
5. Uses the evidence to support the generated investigation report.

RAG is used to **ground the investigation in retrieved information** rather than relying exclusively on an LLM's pretrained knowledge.

---

#  AI Investigator

The AI Investigator acts as the orchestration layer between the predictive models and the final decision.

For a suspicious transaction, it can combine:

```text
Model predictions
      +
SHAP explanations
      +
Connected transactions
      +
Transaction history
      +
RAG evidence
      +
Risk-engine assessment
```

The result is an **evidence package** that can be interpreted by the downstream policy and reporting layers.

The LLM is therefore used primarily for **evidence synthesis and investigation reporting**, rather than as the underlying fraud classifier.

---

#  Policy Engine

The Policy Engine converts the risk assessment and evidence into an operational recommendation.

Examples include:

```text
LOW     → Monitor / Allow
MEDIUM  → Review
HIGH    → Escalate
```

The policy layer is intentionally separated from the LLM so that operational decisions can be governed by explicit rules rather than generated solely by an AI model.

---

#  Audit Logging

Every investigation can produce a structured audit record containing information such as:

* Transaction ID
* Model predictions
* Fusion result
* Risk level
* Evidence category
* Investigation findings
* Retrieved evidence
* Policy decision
* Investigation output

This provides traceability for later review.

---

#  Application

The project includes an interactive **Streamlit application** that exposes the complete investigation workflow.

The application allows a user to select a transaction and inspect:

* Risk assessment
* XGBoost prediction
* GraphSAGE prediction
* Hybrid result
* Model agreement
* SHAP evidence
* Connected transactions
* Retrieved knowledge
* AI investigation
* Policy recommendation
* Audit information

---

#  Example Investigation

A representative high-risk investigation follows the workflow:

```text
Transaction: 71987809
        │
        ▼
XGBoost prediction
        │
        ├──────────────┐
        ▼              ▼
GraphSAGE prediction  SHAP
        │              │
        └──────┬───────┘
               ▼
         Model Fusion
               │
               ▼
          HIGH RISK
               │
               ▼
       Graph Investigation
               │
               ▼
          RAG Retrieval
               │
               ▼
        AI Investigator
               │
               ▼
         Policy Engine
               │
               ▼
            ESCALATE
               │
               ▼
          Audit Log
```

Additional representative transactions are used to demonstrate:

* Medium/conflicting behaviour
* Low-risk behaviour
* Model disagreement

---

#  Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

Create and activate a Python environment:

```bash
python -m venv venv
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required API key through an environment variable rather than committing it to the repository.

---

#  Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

Select a supported transaction ID and run the investigation pipeline.

---

#  CLI Investigation

The project also provides a command-line investigation pipeline for running investigations without the Streamlit interface.

The CLI produces structured investigation output and can be used for testing and reproducibility.

---

#  Results Summary

The final system combines predictive performance with downstream investigation capabilities.

### Final held-out model results

| Model      |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ---------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| XGBoost    |     97.27% |     87.78% |     47.55% |     61.69% |     84.82% |
| GraphSAGE  |     94.59% |     41.59% |     42.40% |     41.99% |     80.69% |
| **Hybrid** | **97.30%** | **88.24%** | **47.79%** | **62.00%** | **83.16%** |

### System-level results

* **8,841 transactions** processed by the final risk engine
* **221 HIGH-risk transactions**
* **82.91% model agreement** on validation data
* **17.09% model disagreement** retained for investigation
* **1,328 graph-signal-only cases**
* **246 strong-corroboration cases**

These results demonstrate the combination of **transaction-level prediction, graph intelligence, evidence generation, and automated investigation** rather than relying on a single fraud score.

---

#  Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── ...
│
├── models/
│   ├── xgboost/
│   └── graphsage/
│
├── src/
│   ├── models/
│   ├── risk_engine/
│   ├── investigation/
│   ├── graph/
│   ├── rag/
│   ├── policy/
│   └── audit/
│
├── results/
│   ├── final_test_metrics.csv
│   ├── hybrid_fusion_metrics.csv
│   ├── model_agreement_validation.csv
│   └── final_risk_engine.csv
│
└── ...
```

---

#  Limitations

* The system is evaluated using the available Elliptic dataset and therefore does not represent the full complexity of a production cryptocurrency monitoring environment.
* The dataset contains a substantial number of unknown transactions.
* Model performance can change when deployed on new transaction distributions.
* GraphSAGE performance is dependent on the quality and structure of the available transaction graph.
* RAG quality depends on the coverage and quality of the knowledge base.
* LLM-generated investigation reports should be treated as decision support rather than unquestionable ground truth.
* Production deployment would require additional monitoring, security controls, latency optimization, and model governance.

---

#  Future Improvements

Potential future extensions include:

* Real-time transaction-stream processing
* Larger and continuously updated transaction graphs
* Temporal graph neural networks
* Improved class-imbalance handling
* Online model monitoring and drift detection
* More sophisticated graph anomaly detection
* Human-in-the-loop analyst feedback
* Expanded fraud knowledge bases
* Automated case management integration
* Production-scale distributed inference
* Continuous policy and model evaluation

---

#  Key Takeaway

Most fraud detection systems answer:

> **"Is this transaction suspicious?"**

This project aims to answer a broader question:

> **"Is this transaction suspicious, why is it suspicious, what evidence supports that assessment, and what should happen next?"**

By combining **XGBoost, GraphSAGE, hybrid risk fusion, SHAP, graph investigation, RAG, AI-assisted investigation, policy rules, and audit logging**, the project turns a fraud prediction into a more complete **automated investigation workflow**.

---

##  Demo Transactions

| Transaction ID | Purpose                                     |
| -------------- | ------------------------------------------- |
| **71987809**   | Strong corroborated high-risk investigation |
| **3295818**    | Medium / conflicting behaviour              |
| **73020281**   | Low-risk investigation                      |
| **72689707**   | Model disagreement                          |

---

##  Disclaimer

This project is a research/prototype system developed for fraud detection and investigation using the Elliptic Bitcoin transaction dataset. The generated risk assessments and investigation reports are intended as decision-support outputs and should not be interpreted as definitive proof of criminal activity.
