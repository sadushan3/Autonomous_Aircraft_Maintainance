from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import (
    HEALTH_STATE_COLUMN,
    STATE_CHANGED_COLUMN,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_trend_tracker import (
    DETERIORATION_RATE_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_csv,
)


ALERT_LEVEL_COLUMN = "alert_level"


_ALERT_LADDER = ["none", "info", "warning", "high", "critical"]


_RAPID_DECLINE_THRESHOLD = 2.0


class HealthAlertEngine:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.mapping: Dict[str, str] = dict(self.config.health.alert_levels)


    def _escalate(self, level: str) -> str:


        try:
            index = _ALERT_LADDER.index(level)
        except ValueError:
            return level
        return _ALERT_LADDER[min(index + 1, len(_ALERT_LADDER) - 1)]

    def derive(self, frame: pd.DataFrame) -> pd.DataFrame:


        result = frame.copy()
        base_level = result[HEALTH_STATE_COLUMN].map(self.mapping).fillna("none")

        just_transitioned = (
            result[STATE_CHANGED_COLUMN]
            if STATE_CHANGED_COLUMN in result.columns
            else pd.Series(False, index=result.index)
        )
        rapid_decline = (
            result[DETERIORATION_RATE_COLUMN] > _RAPID_DECLINE_THRESHOLD
            if DETERIORATION_RATE_COLUMN in result.columns
            else pd.Series(False, index=result.index)
        )
        escalate_mask = (just_transitioned | rapid_decline) & (base_level != "none")

        result[ALERT_LEVEL_COLUMN] = base_level
        result.loc[escalate_mask, ALERT_LEVEL_COLUMN] = base_level.loc[escalate_mask].apply(
            self._escalate
        )

        counts = result[ALERT_LEVEL_COLUMN].value_counts().to_dict()
        self.logger.info(
            "Alert levels derived (%d escalated for rapid decline/transition): %s",
            int(escalate_mask.sum()),
            counts,
        )
        return result


    def run(self, frame: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.health_states_csv),
                required_columns=get_identity_columns() + [HEALTH_STATE_COLUMN],
                stage="health_alert_engine",
            )

        alerted = self.derive(source)
        distribution = alerted[ALERT_LEVEL_COLUMN].value_counts().to_dict()
        report: Dict[str, Any] = {
            "rows": int(len(alerted)),
            "engines": int(alerted["unit"].nunique()),
            "alert_distribution": {str(k): int(v) for k, v in distribution.items()},
        }
        atomic_write_json(report, self.config.paths.metrics / "health_alerts.json")
        self.logger.info("Health alert report: %s", report)
        return alerted, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    engine = HealthAlertEngine(config)
    alerted, report = engine.run()
    logger.info("Alert distribution: %s", report["alert_distribution"])
    print(alerted.head(10).to_string())


if __name__ == "__main__":
    main()
