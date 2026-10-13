"""
Dashboard API service and the integration-handoff writer.

This module owns the single most important output-contract file in the whole
component: ``data/exchange/anomaly_health_predictions.csv``.  The Maintenance
Supervisor fuses this component's output with the RUL and Failure-Risk
components by joining on ``(engine_id, cycle)``, so getting this file's shape
exactly right is what makes the whole four-component system work together.

Rules enforced here:

* ``unit`` is aliased to ``engine_id`` *only* at this boundary -- every
  upstream stage keeps using ``unit`` internally.
* At minimum the columns the contract specifies are present:
  ``engine_id, cycle, health_index, health_state, anomaly_score, anomaly_flag,
  severity, confidence_score`` -- plus the extras this component additionally
  offers (``anomaly_type``, ``root_cause``, ``uncertainty_score``).
* The write is atomic (temp file + rename), matching the crash-safety
  contract observed everywhere else in this component.
* This component never imports from or hardcodes paths into the RUL /
  Failure-Risk / Supervisor components -- it only writes its own file.
"""

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

#: The mandatory column set the Maintenance Supervisor contract requires.
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

#: Extra columns this component offers beyond the minimum contract.
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
    """Serves dashboard JSON and writes the integration exchange file."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the service.

        Args:
            config: Configuration supplying the exchange-file path.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.dashboard_service = DashboardService(self.config)

    # ------------------------------------------------------------------ #
    # JSON payloads for the router
    # ------------------------------------------------------------------ #
    def dashboard_payload(self) -> Dict[str, Any]:
        """Build the full dashboard JSON payload.

        Returns:
            A mapping with ``fleet``, ``engines``, ``alerts`` and ``timeline``.
        """
        return self.dashboard_service.build_all()

    def engine_payload(self, engine_id: int) -> Dict[str, Any]:
        """Build the payload for a single engine.

        Args:
            engine_id: Engine identifier.

        Returns:
            A mapping with that engine's summary and full timeline.

        Raises:
            AnomalyHealthError: If the engine has no data.
        """
        frame = self.dashboard_service.load()
        subset = frame[frame["unit"] == engine_id]
        if subset.empty:
            raise AnomalyHealthError(f"No dashboard data found for engine_id={engine_id}.")

        summaries = self.dashboard_service.engine_summaries(subset)
        timeline = self.dashboard_service.engine_timeline(frame, engine_id)
        return {"summary": summaries[0] if summaries else {}, "timeline": timeline}

    # ------------------------------------------------------------------ #
    # Integration exchange file
    # ------------------------------------------------------------------ #
    def build_exchange_frame(self, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build the exchange-file frame from the dashboard data.

        Args:
            frame: Optional in-memory dashboard-data frame; otherwise
                ``dashboard_data.csv`` is loaded.

        Returns:
            A frame aliasing ``unit`` to ``engine_id`` and carrying every
            required and available optional exchange column.

        Raises:
            AnomalyHealthError: If a required contract column is missing and
                cannot be defaulted.
        """
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
        """Write the integration handoff CSV atomically.

        Args:
            frame: Optional in-memory dashboard-data frame.

        Returns:
            The path written: ``data/exchange/anomaly_health_predictions.csv``.
        """
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
        """Execute the dashboard + exchange stage end to end.

        Returns:
            A report of what was written.
        """
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
    """Run the dashboard/exchange stage standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    service = DashboardApiService(config)
    report = service.run()
    logger.info("Dashboard/exchange report: %s", report)


if __name__ == "__main__":
    main()
