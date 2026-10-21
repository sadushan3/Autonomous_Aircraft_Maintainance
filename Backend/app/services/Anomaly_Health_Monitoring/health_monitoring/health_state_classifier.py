from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import (
    HEALTH_SCORE_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)


HEALTH_STATE_COLUMN = "health_state"


STATE_CHANGED_COLUMN = "state_changed"


_DEFAULT_ORDER = ["Healthy", "Degrading", "Warning", "Critical", "Failure Risk"]


class HealthStateClassifier:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.thresholds_path = self.config.paths.model_path(
            self.config.paths.health_state_thresholds_json
        )
        self.thresholds: Dict[str, float] = dict(self.config.health.state_thresholds)
        self.state_order: List[str] = list(self.config.health.state_order)


    def load_thresholds(self) -> Dict[str, float]:


        payload = read_json(self.thresholds_path, default=None)
        if payload and "thresholds" in payload:
            self.thresholds = {k: float(v) for k, v in payload["thresholds"].items()}
            self.logger.info("Loaded persisted health-state thresholds: %s", self.thresholds)
        else:
            self.save_thresholds()
        return self.thresholds

    def save_thresholds(self) -> Path:


        self._validate_thresholds(self.thresholds)
        payload = {"thresholds": self.thresholds, "state_order": self.state_order}
        return atomic_write_json(payload, self.thresholds_path)

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> Dict[str, float]:


        candidate = {**self.thresholds, **{k: float(v) for k, v in new_thresholds.items()}}
        self._validate_thresholds(candidate)
        self.thresholds = candidate
        self.save_thresholds()
        self.logger.info("Updated health-state thresholds: %s", self.thresholds)
        return self.thresholds


    @staticmethod
    def _validate_thresholds(thresholds: Dict[str, float]) -> None:
        values = [float(thresholds.get(key, np.nan)) for key in ("healthy", "degrading", "warning", "critical")]
        if not np.isfinite(values).all() or not (100 >= values[0] > values[1] > values[2] > values[3] >= 0):
            raise AnomalyHealthError("Health thresholds must be finite, within 0-100, and strictly decreasing.")

    def _classify_scalar(self, score: float) -> str:


        if not np.isfinite(score) or not 0 <= score <= 100:
            raise AnomalyHealthError("A health state requires a finite score between zero and 100.")
        if score >= self.thresholds.get("healthy", 80.0):
            return "Healthy"
        if score >= self.thresholds.get("degrading", 60.0):
            return "Degrading"
        if score >= self.thresholds.get("warning", 40.0):
            return "Warning"
        if score >= self.thresholds.get("critical", 20.0):
            return "Critical"
        return "Failure Risk"

    def classify(self, health_scores: pd.DataFrame) -> pd.DataFrame:


        self.load_thresholds()
        self._validate_thresholds(self.thresholds)
        result = health_scores.sort_values(get_identity_columns()).reset_index(drop=True)

        result[HEALTH_STATE_COLUMN] = result[HEALTH_SCORE_COLUMN].apply(self._classify_scalar)

        previous_state = result.groupby("unit", sort=False)[HEALTH_STATE_COLUMN].shift()
        first_cycle = result.groupby("unit", sort=False).cumcount() == 0
        result[STATE_CHANGED_COLUMN] = (
            (result[HEALTH_STATE_COLUMN] != previous_state) & ~first_cycle
        )

        distribution = result[HEALTH_STATE_COLUMN].value_counts().to_dict()
        self.logger.info("Health state distribution: %s", distribution)
        return result

    def state_rank(self, state: str) -> int:


        try:
            return self.state_order.index(state)
        except ValueError:
            return len(self.state_order) - 1


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.health_states_csv)
        verify_identity_columns(frame, context="health_states output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.health_states_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_state_classifier"
        )

    def run(self, health_scores: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = health_scores
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.health_scores_csv),
                required_columns=get_identity_columns() + [HEALTH_SCORE_COLUMN],
                stage="health_state_classifier",
            )

        classified = self.classify(source)
        output = self.save(classified)

        distribution = classified[HEALTH_STATE_COLUMN].value_counts().to_dict()
        report: Dict[str, Any] = {
            "rows": int(len(classified)),
            "engines": int(classified["unit"].nunique()),
            "state_distribution": {str(k): int(v) for k, v in distribution.items()},
            "state_changes": int(classified[STATE_CHANGED_COLUMN].sum()),
            "thresholds": self.thresholds,
            "output_path": str(output),
        }
        self.logger.info("Health state report: %s", report)
        return classified, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    classifier = HealthStateClassifier(config)
    classified, report = classifier.run()
    logger.info("health_states.csv written: %d rows", report["rows"])
    print(classified.head(10).to_string())


if __name__ == "__main__":
    main()
