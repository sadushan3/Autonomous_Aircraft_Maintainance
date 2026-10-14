"""
Explainability evaluation: are the generated explanations actually useful?

Checks two things that are easy to get subtly wrong when generating text
programmatically:

* **Completeness** -- every explanation record should have non-empty why /
  which_sensors / what_pattern / how_severe fields; a blank field means the
  narrative builder silently failed to find something to say.
* **Consistency** -- the sensors named in ``which_sensors`` should actually be
  present in the underlying root-cause record's ``top_sensors`` for the same
  row, i.e. the narrative is not drifting from the numbers that produced it.

Also reports whether SHAP was actually computed (versus silently skipped),
since that is easy to miss when SHAP fails softly by design.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_csv, read_json


class ExplainabilityEvaluator:
    """Evaluates explanation completeness and numeric consistency."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying report output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def _completeness(self, explanations: pd.DataFrame) -> Dict[str, Any]:
        """Measure how many explanation records have all four narrative fields.

        Args:
            explanations: Output of :class:`ExplanationGenerator`.

        Returns:
            A mapping of completeness fraction and any incomplete-field counts.
        """
        required_fields = ["why", "which_sensors", "what_pattern", "how_severe"]
        present = [f for f in required_fields if f in explanations.columns]
        if not present or explanations.empty:
            return {"completeness_fraction": 0.0, "missing_by_field": {}}

        missing_by_field = {
            field: int(explanations[field].astype(str).str.strip().eq("").sum())
            for field in present
        }
        complete_rows = (
            explanations[present].astype(str).apply(lambda col: col.str.strip().ne("")).all(axis=1)
        )
        return {
            "completeness_fraction": round(float(complete_rows.mean()), 4),
            "missing_by_field": missing_by_field,
        }

    def _consistency(
        self, explanations: pd.DataFrame, root_causes: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check that narrated sensors match the underlying root-cause record.

        Args:
            explanations: Output of :class:`ExplanationGenerator`.
            root_causes: Output of :class:`RootCauseAnalyzer`.

        Returns:
            A mapping of the consistency fraction across matched rows.
        """
        if explanations.empty or root_causes.empty:
            return {"consistency_fraction": None, "rows_checked": 0}
        if "which_sensors" not in explanations.columns or "top_sensors" not in root_causes.columns:
            return {"consistency_fraction": None, "rows_checked": 0}

        merged = explanations.merge(
            root_causes[["unit", "cycle", "top_sensors"]], on=["unit", "cycle"], how="inner"
        )
        if merged.empty:
            return {"consistency_fraction": None, "rows_checked": 0}

        def _safe_parse(value: Any) -> set:
            try:
                parsed = ast.literal_eval(str(value))
                return set(parsed) if isinstance(parsed, (list, tuple)) else set()
            except (ValueError, SyntaxError):
                return set()

        matches = merged.apply(
            lambda row: _safe_parse(row["which_sensors"]) == _safe_parse(row["top_sensors"]),
            axis=1,
        )
        return {
            "consistency_fraction": round(float(matches.mean()), 4),
            "rows_checked": int(len(merged)),
        }

    def evaluate(
        self, explanations: pd.DataFrame, root_causes: pd.DataFrame
    ) -> Dict[str, Any]:
        """Combine completeness and consistency checks into one report.

        Args:
            explanations: Output of :class:`ExplanationGenerator`.
            root_causes: Output of :class:`RootCauseAnalyzer`.

        Returns:
            The combined evaluation report.
        """
        explainability_summary = read_json(self.config.paths.metrics / "explainability.json", default={})
        completeness = self._completeness(explanations)
        consistency = self._consistency(explanations, root_causes)

        report: Dict[str, Any] = {
            "rows_explained": int(len(explanations)),
            "completeness": completeness,
            "consistency": consistency,
            "shap_computed": explainability_summary.get("rows_explained", 0) > 0,
            "acceptance": {
                "mostly_complete": completeness["completeness_fraction"] >= 0.95,
            },
        }
        self.logger.info("Explainability evaluation: %s", report["acceptance"])
        return report

    def run(
        self,
        explanations: Optional[pd.DataFrame] = None,
        root_causes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute the explainability evaluation stage.

        Args:
            explanations: Optional in-memory input; otherwise
                ``explanation_reports.csv`` is read from disk.
            root_causes: Optional in-memory input; otherwise
                ``root_cause_analysis.csv`` is read from disk.

        Returns:
            The evaluation report.
        """
        self.config.ensure_directories()
        explanation_source = explanations
        if explanation_source is None:
            path = self.config.paths.processed_path(self.config.paths.explanation_reports_csv)
            explanation_source = read_csv(path, stage="evaluate_explainability") if path.is_file() else pd.DataFrame()

        root_cause_source = root_causes
        if root_cause_source is None:
            path = self.config.paths.processed_path(self.config.paths.root_cause_analysis_csv)
            root_cause_source = read_csv(path, stage="evaluate_explainability") if path.is_file() else pd.DataFrame()

        report = self.evaluate(explanation_source, root_cause_source)
        atomic_write_json(report, self.config.paths.reports / "explainability_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_explainability.json")
        return report


def main() -> None:
    """Run explainability evaluation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = ExplainabilityEvaluator(config)
    report = evaluator.run()
    logger.info("Explainability evaluation report: %s", report)


if __name__ == "__main__":
    main()
