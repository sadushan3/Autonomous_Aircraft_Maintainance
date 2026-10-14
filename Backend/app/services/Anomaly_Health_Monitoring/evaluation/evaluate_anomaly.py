"""
Anomaly-detection evaluation: are we finding faults without crying wolf?

Without ground-truth fault labels (classic C-MAPSS carries none), this module
evaluates the anomaly layer against two proxies that *are* verifiable from the
data:

* **Alignment with RUL** -- fault-type anomalies should concentrate in the
  low-RUL (near end-of-life) region rather than being scattered uniformly
  across a trajectory, since real degradation is what the layer is supposed to
  catch.
* **Operational-suppression rate** -- on multi-regime subsets (FD002/FD004),
  the fraction of raw detector flags that get reclassified as ``operational``
  rather than escalated as ``fault``.  A healthy value here is direct evidence
  the operational-vs-fault gate is doing its job; zero on a multi-regime
  subset is a red flag that regime changes are leaking through as false alarms.

Once labelled data or operator feedback accumulates, :mod:`feedback` closes
the loop that a proper precision/recall evaluation would need; this module
reports what can be assessed from the unlabelled signal alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import RUL_COLUMN, get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_csv


class AnomalyEvaluator:
    """Evaluates the anomaly-detection layer against RUL alignment and gate behaviour."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying report output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def rul_alignment(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Compare mean RUL at fault-flagged rows against the fleet mean.

        Args:
            frame: Frame carrying ``anomaly_type`` and RUL (directly or via a
                merge upstream).

        Returns:
            A mapping of mean RUL at fault rows vs. the fleet mean, and whether
            faults concentrate at low RUL as expected.
        """
        if RUL_COLUMN not in frame.columns or "anomaly_type" not in frame.columns:
            return {"available": False}

        fleet_mean_rul = float(frame[RUL_COLUMN].mean())
        fault_rows = frame[frame["anomaly_type"] == "fault"]
        fault_mean_rul = float(fault_rows[RUL_COLUMN].mean()) if len(fault_rows) else None

        return {
            "available": True,
            "fleet_mean_rul": round(fleet_mean_rul, 2),
            "fault_mean_rul": round(fault_mean_rul, 2) if fault_mean_rul is not None else None,
            "faults_concentrate_at_low_rul": bool(
                fault_mean_rul is not None and fault_mean_rul < fleet_mean_rul
            ),
        }

    def gate_behaviour(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Summarise how the operational-vs-fault gate behaved.

        Args:
            frame: Frame carrying ``anomaly_flag`` and ``anomaly_type``.

        Returns:
            A mapping of flagged/fault/operational counts and the suppression
            rate.
        """
        if "anomaly_type" not in frame.columns:
            return {"available": False}

        total_flagged = int(frame.get("anomaly_flag", pd.Series(dtype=bool)).sum())
        fault = int((frame["anomaly_type"] == "fault").sum())
        operational = int((frame["anomaly_type"] == "operational").sum())
        suppression_rate = operational / max(total_flagged, 1)

        return {
            "available": True,
            "total_flagged": total_flagged,
            "fault_anomalies": fault,
            "operational_anomalies": operational,
            "operational_suppression_rate": round(suppression_rate, 4),
        }

    def evaluate(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Combine RUL alignment and gate behaviour into one report.

        Args:
            frame: Fused-anomaly frame, optionally merged with RUL.

        Returns:
            The combined evaluation report.
        """
        rul = self.rul_alignment(frame)
        gate = self.gate_behaviour(frame)
        report: Dict[str, Any] = {
            "rul_alignment": rul,
            "gate_behaviour": gate,
            "acceptance": {
                "faults_align_with_low_rul": bool(rul.get("faults_concentrate_at_low_rul", False)),
            },
        }
        self.logger.info("Anomaly evaluation: %s", report["acceptance"])
        return report

    def run(self, frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the anomaly evaluation stage.

        Args:
            frame: Optional in-memory input; otherwise ``anomaly_fusion.csv``
                (merged with RUL from ``raw_data.csv`` when available) is used.

        Returns:
            The evaluation report.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.anomaly_fusion_csv),
                required_columns=get_identity_columns(),
                stage="evaluate_anomaly",
            )
            raw_path = self.config.paths.processed_path(self.config.paths.raw_data_csv)
            if raw_path.is_file() and RUL_COLUMN not in source.columns:
                raw = read_csv(raw_path, stage="evaluate_anomaly")
                if RUL_COLUMN in raw.columns:
                    from app.utils.Anomaly_Health_Monitoring.utils import merge_on_keys

                    source = merge_on_keys(
                        source,
                        raw[get_identity_columns() + [RUL_COLUMN]],
                        keys=get_identity_columns(),
                        how="left",
                    )

        report = self.evaluate(source)
        atomic_write_json(report, self.config.paths.reports / "anomaly_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_anomaly.json")
        return report


def main() -> None:
    """Run anomaly evaluation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = AnomalyEvaluator(config)
    report = evaluator.run()
    logger.info("Anomaly evaluation report: %s", report)


if __name__ == "__main__":
    main()
