"""
Context-modelling evaluation: did operating-mode discovery find the right structure?

Compares the discovered regime count against the documented C-MAPSS
expectation for the loaded subsets (1 for FD001/FD003, 6 for FD002/FD004),
reports the clustering quality (silhouette or BIC, whichever the configured
algorithm produced), and reports the fleet's transition-row fraction --
too high suggests the guard window is too wide; zero on a multi-regime subset
would suggest the guard is not firing at all.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_json



class ContextEvaluator:
    """Produces the acceptance report for the context-modelling layer."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying report output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def evaluate(self) -> Dict[str, Any]:
        """Assemble the context-modelling evaluation report.

        Returns:
            A report of regime discovery quality and drift status.
        """
        context_report = read_json(self.config.paths.metrics / "context_modeling.json", default={})
        drift_report = read_json(self.config.paths.metrics / "context_drift.json", default={})

        report: Dict[str, Any] = {
            "modes_found": context_report.get("n_modes"),
            "modes_expected": context_report.get("expected_modes"),
            "matches_expectation": context_report.get("matches_expectation"),
            "algorithm": context_report.get("algorithm"),
            "silhouette_score": context_report.get("silhouette_score"),
            "mean_mode_confidence": context_report.get("mean_mode_confidence"),
            "transition_row_fraction": (
                context_report.get("transition_rows", 0) / max(context_report.get("rows", 1), 1)
            ),
            "drift_detected": drift_report.get("drift_detected"),
            "drift_psi": drift_report.get("psi"),
            "acceptance": {
                "regime_count_matches_documentation": bool(context_report.get("matches_expectation", False)),
                "context_stable": not bool(drift_report.get("drift_detected", False)),
            },
        }
        self.logger.info("Context evaluation: %s", report["acceptance"])
        return report

    def run(self) -> Dict[str, Any]:
        """Execute the evaluation and persist the report.

        Returns:
            The evaluation report.
        """
        self.config.ensure_directories()
        report = self.evaluate()
        atomic_write_json(report, self.config.paths.reports / "context_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_context.json")
        return report


def main() -> None:
    """Run context evaluation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = ContextEvaluator(config)
    report = evaluator.run()
    logger.info("Acceptance: %s", report["acceptance"])


if __name__ == "__main__":
    main()
