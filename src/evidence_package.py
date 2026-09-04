from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvidencePackage:
    """
    Standardized evidence container passed between
    the investigation pipeline components.
    """

    transaction: Dict[str, Any] = field(default_factory=dict)

    risk_assessment: Dict[str, Any] = field(default_factory=dict)

    model_predictions: Dict[str, Any] = field(default_factory=dict)

    model_agreement: str = "UNKNOWN"

    shap_evidence: Dict[str, Any] = field(default_factory=dict)

    graph_evidence: Dict[str, Any] = field(default_factory=dict)

    rag_evidence: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """
        Convert the evidence package into a JSON-serializable
        dictionary.
        """

        return {
            "transaction": self.transaction,

            "risk_assessment": self.risk_assessment,

            "model_predictions": self.model_predictions,

            "model_agreement": self.model_agreement,

            "shap_evidence": self.shap_evidence,

            "graph_evidence": self.graph_evidence,

            "rag_evidence": self.rag_evidence,

            "metadata": self.metadata
        }