from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_csv, read_json


class ReasoningEvaluator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def evaluate(self, root_causes: pd.DataFrame, total_fault_rows: int) -> Dict[str, Any]:


        if root_causes.empty:
            return {
                "coverage": 0.0,
                "n_distinct_causes": 0,
                "n_distinct_subsystems": 0,
                "hard_causal_claims": 0,
                "memory_match_rate": 0.0,
                "acceptance": {"no_hard_causal_claims": True, "reasoned_something": False},
            }

        coverage = len(root_causes) / max(total_fault_rows, 1)
        n_causes = int(root_causes["root_cause"].nunique()) if "root_cause" in root_causes.columns else 0
        n_subsystems = int(root_causes["subsystem"].nunique()) if "subsystem" in root_causes.columns else 0

        hard_claims = 0
        if "hard_causal_claim" in root_causes.columns:
            hard_claims = int(
                root_causes["hard_causal_claim"].astype(str).str.lower().eq("true").sum()
            )

        memory_matches = (
            root_causes["similar_past_anomaly_id"].notna().sum()
            if "similar_past_anomaly_id" in root_causes.columns
            else 0
        )
        memory_match_rate = float(memory_matches / max(len(root_causes), 1))

        report: Dict[str, Any] = {
            "rows_reasoned": int(len(root_causes)),
            "total_fault_rows": int(total_fault_rows),
            "coverage": round(coverage, 4),
            "n_distinct_causes": n_causes,
            "n_distinct_subsystems": n_subsystems,
            "hard_causal_claims": hard_claims,
            "memory_match_rate": round(memory_match_rate, 4),
            "acceptance": {
                "no_hard_causal_claims": hard_claims == 0,
                "reasoned_something": len(root_causes) > 0,
                "full_coverage": coverage >= 0.99,
            },
        }
        self.logger.info("Reasoning evaluation: %s", report["acceptance"])
        return report

    def run(
        self,
        root_causes: Optional[pd.DataFrame] = None,
        total_fault_rows: Optional[int] = None,
    ) -> Dict[str, Any]:


        self.config.ensure_directories()
        source = root_causes
        if source is None:
            path = self.config.paths.processed_path(self.config.paths.root_cause_analysis_csv)
            source = read_csv(path, stage="evaluate_reasoning") if path.is_file() else pd.DataFrame()

        if total_fault_rows is None:
            anomaly_report = read_json(self.config.paths.metrics / "anomaly_detection.json", default={})
            total_fault_rows = int(anomaly_report.get("fault_anomalies", len(source)))

        report = self.evaluate(source, total_fault_rows)
        atomic_write_json(report, self.config.paths.reports / "reasoning_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_reasoning.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = ReasoningEvaluator(config)
    report = evaluator.run()
    logger.info("Reasoning evaluation report: %s", report)


if __name__ == "__main__":
    main()
