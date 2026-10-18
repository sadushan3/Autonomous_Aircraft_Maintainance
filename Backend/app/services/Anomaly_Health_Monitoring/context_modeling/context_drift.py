from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_op_setting_columns,
    get_operating_mode_column,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    get_logger,
    atomic_write_json,
    read_csv,
    read_json,
)


_EPSILON = 1e-6


class ContextDriftDetector:


    REFERENCE_NAME = "context_reference.json"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.mode_column = get_operating_mode_column()
        self.reference_path = self.config.paths.model_path(self.REFERENCE_NAME)


    def fit_reference(self, frame: pd.DataFrame) -> Dict[str, Any]:


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
        frame = training_scope(frame, self.config)
        record_training_scope("context_drift_reference", frame, self.config)
        proportions = self._mode_proportions(frame)
        settings = [c for c in get_op_setting_columns() if c in frame.columns]

        reference: Dict[str, Any] = {
            "mode_proportions": proportions,
            "n_rows": int(len(frame)),
            "n_modes": len(proportions),
            "setting_means": (
                frame[settings].astype(float).mean().to_dict() if settings else {}
            ),
            "setting_stds": (
                frame[settings].astype(float).std().to_dict() if settings else {}
            ),
        }
        atomic_write_json(reference, self.reference_path)
        self.logger.info(
            "Captured context reference: %d mode(s) over %d rows",
            reference["n_modes"],
            reference["n_rows"],
        )
        return reference

    def load_reference(self) -> Optional[Dict[str, Any]]:


        return read_json(self.reference_path, default=None)

    def _mode_proportions(self, frame: pd.DataFrame) -> Dict[str, float]:


        if self.mode_column not in frame.columns or len(frame) == 0:
            return {}
        counts = frame[self.mode_column].value_counts()
        total = float(counts.sum())
        return {str(int(k)): float(v) / total for k, v in counts.items()}


    @staticmethod
    def population_stability_index(
        reference: Dict[str, float], current: Dict[str, float]
    ) -> float:


        labels = sorted(set(reference) | set(current))
        if not labels:
            return 0.0
        total = 0.0
        for label in labels:
            p_ref = max(float(reference.get(label, 0.0)), _EPSILON)
            p_now = max(float(current.get(label, 0.0)), _EPSILON)
            total += (p_now - p_ref) * float(np.log(p_now / p_ref))
        return float(abs(total))

    def detect(self, frame: pd.DataFrame) -> Dict[str, Any]:


        current = self._mode_proportions(frame)
        reference_doc = self.load_reference()

        if not reference_doc:
            from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError
            raise AnomalyHealthError("No frozen context drift reference; capture it from training engines before inference.")

        reference = reference_doc.get("mode_proportions", {})
        psi = self.population_stability_index(reference, current)
        threshold = float(self.config.context.drift_psi_threshold)

        per_mode = {
            label: round(
                float(current.get(label, 0.0)) - float(reference.get(label, 0.0)), 6
            )
            for label in sorted(set(reference) | set(current))
        }

        report: Dict[str, Any] = {
            "drift_detected": bool(psi > threshold),
            "psi": round(psi, 6),
            "threshold": threshold,
            "severity": self._severity(psi),
            "reference_proportions": reference,
            "current_proportions": current,
            "per_mode_shift": per_mode,
            "setting_drift": self._setting_drift(frame, reference_doc),
            "reason": "psi_above_threshold" if psi > threshold else "stable",
        }

        if report["drift_detected"]:
            self.logger.warning(
                "Operating-context drift detected: PSI=%.4f exceeds %.4f. "
                "Residual thresholds calibrated on the old regime mix may no "
                "longer hold; consider refitting the twin.",
                psi,
                threshold,
            )
        else:
            self.logger.info("Operating context stable: PSI=%.4f (threshold %.4f)", psi, threshold)
        return report

    def _setting_drift(
        self, frame: pd.DataFrame, reference_doc: Dict[str, Any]
    ) -> Dict[str, float]:


        reference_means = reference_doc.get("setting_means") or {}
        reference_stds = reference_doc.get("setting_stds") or {}
        drift: Dict[str, float] = {}
        for name in get_op_setting_columns():
            if name not in frame.columns or name not in reference_means:
                continue
            sigma = float(reference_stds.get(name, 0.0) or 0.0)
            shift = float(frame[name].astype(float).mean()) - float(reference_means[name])
            drift[name] = round(shift / sigma, 6) if sigma > _EPSILON else round(shift, 6)
        return drift

    @staticmethod
    def _severity(psi: float) -> str:


        if psi < 0.1:
            return "none"
        if psi < 0.2:
            return "moderate"
        return "major"


    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        capture_reference: bool = False,
    ) -> Dict[str, Any]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(
                    self.config.paths.context_clusters_csv
                ),
                required_columns=[self.mode_column],
                stage="context_drift",
            )

        if capture_reference:
            self.fit_reference(source)

        report = self.detect(source)
        atomic_write_json(report, self.config.paths.metrics / "context_drift.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    detector = ContextDriftDetector(config)
    report = detector.run()
    logger.info(
        "Drift: %s (PSI=%.4f, severity=%s)",
        report["drift_detected"],
        report["psi"],
        report["severity"],
    )
    logger.info("Per-mode shift: %s", report["per_mode_shift"])


if __name__ == "__main__":
    main()
