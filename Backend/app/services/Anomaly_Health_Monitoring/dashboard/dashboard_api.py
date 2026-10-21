from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_service import (
    DashboardService,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    get_logger,
)


REQUIRED_EXCHANGE_COLUMNS: List[str] = [
    "engine_id",
    "cycle",
    "health_index",
    "health_state",
    "anomaly_score",
    "anomaly_flag",
    "severity",
    "confidence_score",
]


OPTIONAL_EXCHANGE_COLUMNS: List[str] = [
    "anomaly_type",
    "root_cause",
    "uncertainty_score",
    "health_trend",
    "remaining_health_pct",
    "early_warning_score",
    "recommended_action",
]


class DashboardApiService:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.dashboard_service = DashboardService(self.config)


    def dashboard_payload(self) -> Dict[str, Any]:


        return self.dashboard_service.build_all()

    def engine_payload(self, engine_id: int) -> Dict[str, Any]:


        frame = self.dashboard_service.load()
        subset = frame[frame["unit"] == engine_id]
        if subset.empty:
            raise AnomalyHealthError(f"No dashboard data found for engine_id={engine_id}.")

        summaries = self.dashboard_service.engine_summaries(subset)
        timeline = self.dashboard_service.engine_timeline(frame, engine_id)
        return {"summary": summaries[0] if summaries else {}, "timeline": timeline}


    def build_exchange_frame(self, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:


        source = frame if frame is not None else self.dashboard_service.load()

        result = source.copy()
        result["engine_id"] = result["unit"].astype(int)

        defaults: Dict[str, Any] = {
            "health_index": 100.0,
            "health_state": "Healthy",
            "anomaly_score": 0.0,
            "anomaly_flag": False,
            "severity": "none",
            "confidence_score": 100.0,
        }
        for column, default in defaults.items():
            if column not in result.columns:
                self.logger.warning(
                    "Exchange contract column '%s' missing from dashboard_data; "
                    "defaulting to %s for every row.",
                    column,
                    default,
                )
                result[column] = default
            else:
                result[column] = result[column].fillna(default)

        keep = ["engine_id", "cycle"] + [
            c for c in REQUIRED_EXCHANGE_COLUMNS if c not in ("engine_id", "cycle")
        ]
        keep += [c for c in OPTIONAL_EXCHANGE_COLUMNS if c in result.columns]

        exchange = result.loc[:, [c for c in keep if c in result.columns]].copy()
        exchange = exchange.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

        missing = [c for c in REQUIRED_EXCHANGE_COLUMNS if c not in exchange.columns]
        if missing:
            raise AnomalyHealthError(
                f"Exchange frame is missing required contract column(s): {missing}"
            )
        return exchange

    def write_exchange_file(self, frame: Optional[pd.DataFrame] = None) -> Path:


        self.config.ensure_directories()
        exchange = self.build_exchange_frame(frame)
        target = (
            self.config.paths.data_exchange / self.config.paths.exchange_predictions_csv
        )
        output = atomic_write_csv(exchange, target, config=self.config)
        self.logger.info(
            "Integration exchange file written: %s (%d rows, %d engines, %d columns)",
            output,
            len(exchange),
            exchange["engine_id"].nunique(),
            exchange.shape[1],
        )
        return output

    def run(self) -> Dict[str, Any]:


        frame = self.dashboard_service.load()
        exchange_path = self.write_exchange_file(frame)
        payload = self.dashboard_payload()

        report: Dict[str, Any] = {
            "rows": int(len(frame)),
            "engines": int(frame["unit"].nunique()),
            "exchange_path": str(exchange_path),
            "fleet_summary": payload["fleet"],
        }
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    service = DashboardApiService(config)
    report = service.run()
    logger.info("Dashboard/exchange report: %s", report)


if __name__ == "__main__":
    main()
