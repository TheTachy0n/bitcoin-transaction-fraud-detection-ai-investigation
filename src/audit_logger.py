import json
from datetime import datetime
from pathlib import Path


class AuditLogger:
    """
    Records every investigation in a persistent JSON audit log.
    """

    def __init__(self, log_path="results/audit_log.json"):
        self.log_path = Path(log_path)

        # Make sure results/ exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty audit log if it doesn't exist
        if not self.log_path.exists():
            self._write([])

    def _read(self):
        """Read existing audit records."""
        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            return []

        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, data):
        """Write audit records to disk."""
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=4, default=str)

    def log_investigation(
        self,
        transaction_id,
        risk_assessment=None,
        model_predictions=None,
        model_agreement=None,
        tools_called=None,
        transaction_evidence=None,
        graph_evidence=None,
        shap_evidence=None,
        rag_evidence=None,
        llm_status=None,
        policy_decision=None,
        final_recommendation=None,
        metadata=None,
    ):
        """
        Append one complete investigation record to the audit log.
        """

        record = {
            "timestamp": datetime.now().isoformat(),

            "transaction_id": str(transaction_id),

            "risk_assessment": risk_assessment or {},

            "model_predictions": model_predictions or {},

            "model_agreement": model_agreement,

            "tools_called": tools_called or [],

            "evidence": {
                "transaction": transaction_evidence or {},
                "graph": graph_evidence or {},
                "shap": shap_evidence or {},
                "rag": rag_evidence or [],
            },

            "llm": llm_status or {},

            "policy_decision": policy_decision or {},

            "final_recommendation": final_recommendation,

            "metadata": metadata or {},
        }

        logs = self._read()
        logs.append(record)
        self._write(logs)

        return record