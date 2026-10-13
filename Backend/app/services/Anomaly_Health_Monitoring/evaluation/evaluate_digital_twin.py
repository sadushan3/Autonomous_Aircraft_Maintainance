"""
Digital Twin evaluation: does the ensemble actually predict healthy behaviour?

Reports, per twin and for the fused ensemble:

* RMSE / MAE / R-squared on the healthy region (already computed at training
  time, re-surfaced here for a single evaluation entry point).
* The residual-validation verdict from :mod:`validate_residuals` (centring,
  spread, coverage, growth).
* Twin agreement: how much the four twins disagree, fleet-wide.

This is the acceptance report for the whole Digital Twin layer, written to
``reports/`` for a human reader and ``metrics/`` for programmatic consumption.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.validate_residuals import (
    ResidualValidator,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)


class DigitalTwinEvaluator:
    """Produces the acceptance report for the Digital Twin layer."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying report output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.validator = ResidualValidator(self.config)

    def evaluate(self) -> Dict[str, Any]:
        """Assemble the full Digital Twin evaluation report.

        Returns:
            A report combining per-twin metrics, ensemble metrics and residual
            validation.
        """
        twin_report = read_json(self.config.paths.metrics / "digital_twin.json", default={})
        residuals_path = self.config.paths.processed_path(self.config.paths.residuals_csv)

        validation: Dict[str, Any] = {}
        if residuals_path.is_file():
            residuals = read_csv(residuals_path, required_columns=get_identity_columns(), stage="evaluate_digital_twin")
            validation = self.validator.validate(residuals)

        report: Dict[str, Any] = {
            "per_twin_metrics": twin_report.get("per_twin_metrics", {}),
            "ensemble_weights": twin_report.get("weights", {}),
            "weighted_mean_rmse": twin_report.get("weighted_mean_rmse"),
            "mean_twin_spread": twin_report.get("mean_twin_spread"),
            "startup_profile": twin_report.get("startup_profile", {"active": False}),
            "residual_validation": validation,
            "acceptance": {
                "twins_trained": len(twin_report.get("twins", [])),
                "residuals_pass_validation": bool(validation.get("passed", False)),
            },
        }
        self.logger.info("Digital twin evaluation: %s", report["acceptance"])
        return report

    def run(self) -> Dict[str, Any]:
        """Execute the evaluation and persist the report.

        Returns:
            The evaluation report.
        """
        self.config.ensure_directories()
        report = self.evaluate()
        atomic_write_json(report, self.config.paths.reports / "digital_twin_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_digital_twin.json")
        return report


def main() -> None:
    """Run Digital Twin evaluation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = DigitalTwinEvaluator(config)
    report = evaluator.run()
    logger.info("Acceptance: %s", report["acceptance"])


if __name__ == "__main__":
    main()
