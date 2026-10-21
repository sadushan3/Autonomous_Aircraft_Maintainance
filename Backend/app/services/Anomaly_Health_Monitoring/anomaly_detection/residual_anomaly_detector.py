from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    empirical_percentile_normalize,
    get_logger,
    percentile_threshold,
    atomic_write_json,
    read_json,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    present_residual_z_columns,
)


SCORE_COLUMN = "residual_rule_score"


_SENSOR_Z_TRIGGER = 3.0


class ResidualAnomalyDetector:


    META_NAME = "residual_rule_meta.json"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.feature_columns: List[str] = []
        self.threshold: float = 1.0
        self._healthy_raw_sorted: np.ndarray = np.array([], dtype=float)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:


        settings = self.config.healthy_region
        if RUL_COLUMN in frame.columns:
            rul_ok = frame[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=frame.index)
        max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = frame["cycle"].astype(float) <= (float(settings.cycle_fraction) * max_cycle)
        return rul_ok & cycle_ok


    def _raw_score(self, frame: pd.DataFrame) -> np.ndarray:


        values = frame[self.feature_columns].astype(float).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean_abs_z = np.abs(values).mean(axis=1)
        violation_count = (np.abs(values) > _SENSOR_Z_TRIGGER).sum(axis=1)
        return mean_abs_z + 0.5 * violation_count


    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "ResidualAnomalyDetector":


        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for the rule detector.")
        self.feature_columns = present

        healthy = frame.loc[self._healthy_mask(frame)]
        if healthy.empty:
            healthy = frame

        raw = self._raw_score(healthy)
        self._healthy_raw_sorted = np.sort(raw)
        normalized_healthy = empirical_percentile_normalize(self._healthy_raw_sorted, raw)
        self.threshold = percentile_threshold(
            normalized_healthy, self.config.anomaly.residual_threshold_percentile
        )
        self.logger.info(
            "Residual rule detector calibrated on %d healthy row(s); threshold=%.4f (p%.0f)",
            len(healthy),
            self.threshold,
            self.config.anomaly.residual_threshold_percentile,
        )
        return self


    def score(self, frame: pd.DataFrame) -> np.ndarray:


        if not self.feature_columns:
            raise AnomalyHealthError("ResidualAnomalyDetector used before fit/load.")
        raw = self._raw_score(frame)
        return empirical_percentile_normalize(self._healthy_raw_sorted, raw)

    def predict_flags(self, frame: pd.DataFrame) -> np.ndarray:


        return self.score(frame) > self.threshold


    def save(self) -> None:

        atomic_write_json(
            {
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
                "healthy_raw_sorted": self._healthy_raw_sorted.tolist(),
            },
            self.config.paths.model_path(self.META_NAME),
        )

    def load(self) -> "ResidualAnomalyDetector":


        meta = read_json(self.config.paths.model_path(self.META_NAME), default=None)
        if not meta:
            raise AnomalyHealthError("No residual rule metadata found; calibrate it first.")
        self.feature_columns = list(meta["feature_columns"])
        self.threshold = float(meta["threshold"])
        self._healthy_raw_sorted = np.asarray(meta["healthy_raw_sorted"], dtype=float)
        return self


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="residual_anomaly_detector",
    )
    feature_columns = present_residual_z_columns(frame)
    detector = ResidualAnomalyDetector(config).fit(frame, feature_columns)
    detector.save()
    scores = detector.score(frame)
    logger.info(
        "Mean score=%.4f, flagged=%d/%d", float(scores.mean()),
        int((scores > detector.threshold).sum()), len(frame),
    )


if __name__ == "__main__":
    main()
