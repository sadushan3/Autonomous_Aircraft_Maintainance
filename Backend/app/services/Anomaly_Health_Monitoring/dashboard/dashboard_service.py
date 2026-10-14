"""
Dashboard service: reshapes the wide dashboard table into API-ready views.

Three views are produced from ``dashboard_data.csv``:

* **Fleet statistics** -- one summary row across every engine.
* **Engine summaries** -- one card per engine, showing its latest cycle.
* **Alert feed** -- every row with a non-'none' alert level, most recent
  first, for an operator's triage queue.
* **Per-engine timelines** -- health/anomaly history for charting.

This module contains no I/O beyond reading the already-assembled wide table;
:mod:`dashboard_data_generator` owns producing that table, and
:mod:`dashboard_api` owns exposing these views plus the integration handoff
file over HTTP.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, read_csv
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

#: Columns pulled onto an engine card / timeline point when present.
_OPTIONAL_COLUMNS = [
    "health_index", "health_index_target", "health_score", "health_state", "health_trend",
    "remaining_health_pct", "anomaly_score", "anomaly_flag", "anomaly_type",
    "severity", "alert_level", "confidence_score", "root_cause",
    "top_sensors", "operating_mode", "explanation_text",
    "twin_spread_mean", "deterioration_rate", "reliability_score",
    "uncertainty_score", "model_agreement", "detector_agreement",
    "evolution_pattern",
    # Per-detector scores and the calibrated cut. Without these the UI can show
    # that a row was flagged but not which of the four detectors drove it, which
    # is the first question anyone asks of an anomaly.
    "isolation_forest_score", "mahalanobis_score", "lstm_ae_score",
    "residual_rule_score", "threshold_used", "anomaly_persistence_cycles",
    "is_anomalous", "early_warning_score",
    # Structured explainability. `subsystem_attribution` maps the deviation
    # onto physical engine sections with a contribution share each, which is a
    # far more interpretable answer to "why" than a flat sensor list -- it was
    # being computed for every explained anomaly and thrown away at the API.
    "subsystem_attribution", "why", "which_sensors", "what_pattern",
    "how_severe", "shap_top_features",
]


class DashboardService:
    """Reshapes the wide dashboard table into fleet/engine/alert/timeline views."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the service.

        Args:
            config: Configuration supplying the dashboard-data path.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load ``dashboard_data.csv``.

        Returns:
            The wide dashboard table.
        """
        return read_csv(
            self.config.paths.processed_path(self.config.paths.dashboard_data_csv),
            required_columns=get_identity_columns(),
            stage="dashboard_service",
        )

    @staticmethod
    def _latest_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
        """Reduce a frame to each engine's most recent cycle.

        Args:
            frame: The wide dashboard table.

        Returns:
            One row per engine, its latest cycle.
        """
        return (
            frame.sort_values("cycle")
            .groupby("unit", as_index=False, sort=True)
            .tail(1)
            .sort_values("unit")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    # Fleet statistics
    # ------------------------------------------------------------------ #
    def fleet_statistics(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Compute fleet-wide rollup statistics.

        Args:
            frame: The wide dashboard table.

        Returns:
            A mapping matching the ``FleetStatistics`` schema.
        """
        latest = self._latest_per_engine(frame)
        state_counts = latest.get("health_state", pd.Series(dtype=object)).value_counts().to_dict()
        type_counts = frame.get("anomaly_type", pd.Series(dtype=object)).value_counts().to_dict()
        cause_counts = (
            frame.get("root_cause", pd.Series(dtype=object)).dropna().value_counts().head(5).to_dict()
        )
        #: Health-state distribution across every recorded cycle of every
        #: engine's lifetime, not just each engine's latest/most-recent cycle.
        #: ``engines_*`` above answers "how is the fleet doing right now" (and
        #: is naturally skewed toward degraded states for run-to-failure
        #: trajectories, since every engine's *latest* row is near end-of-life
        #: by construction); this answers "how healthy is a lifetime overall".
        lifetime_state_counts = frame.get("health_state", pd.Series(dtype=object)).value_counts().to_dict()

        return {
            "total_engines": int(latest["unit"].nunique()),
            "total_rows": int(len(frame)),
            "mean_health_index": float(latest.get("health_index", pd.Series([0.0])).mean()),
            "engines_healthy": int(state_counts.get("Healthy", 0)),
            "engines_degrading": int(state_counts.get("Degrading", 0)),
            "engines_warning": int(state_counts.get("Warning", 0)),
            "engines_critical": int(state_counts.get("Critical", 0)),
            "engines_failure_risk": int(state_counts.get("Failure Risk", 0)),
            "total_anomalies": int(sum(v for k, v in type_counts.items() if k != "none")),
            "fault_anomalies": int(type_counts.get("fault", 0)),
            "operational_anomalies": int(type_counts.get("operational", 0)),
            "mean_confidence": float(latest.get("confidence_score", pd.Series([0.0])).mean()),
            "top_root_causes": {str(k): int(v) for k, v in cause_counts.items()},
            "lifetime_mean_health_index": float(frame.get("health_index", pd.Series([0.0])).mean()),
            "lifetime_rows_healthy": int(lifetime_state_counts.get("Healthy", 0)),
            "lifetime_rows_degrading": int(lifetime_state_counts.get("Degrading", 0)),
            "lifetime_rows_warning": int(lifetime_state_counts.get("Warning", 0)),
            "lifetime_rows_critical": int(lifetime_state_counts.get("Critical", 0)),
            "lifetime_rows_failure_risk": int(lifetime_state_counts.get("Failure Risk", 0)),
        }

    # ------------------------------------------------------------------ #
    # Engine cards
    # ------------------------------------------------------------------ #
    def engine_summaries(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build one summary card per engine.

        Args:
            frame: The wide dashboard table.

        Returns:
            A list of engine-summary dictionaries.
        """
        latest = self._latest_per_engine(frame)
        totals = frame.groupby("unit")["cycle"].max().to_dict()
        cards: List[Dict[str, Any]] = []

        for _, row in latest.iterrows():
            card: Dict[str, Any] = {
                "engine_id": int(row["unit"]),
                "latest_cycle": int(row["cycle"]),
                "total_cycles": int(totals.get(row["unit"], row["cycle"])),
            }
            for column in _OPTIONAL_COLUMNS:
                if column in row.index and pd.notna(row[column]):
                    card[column] = row[column]
            cards.append(self._sanitize(card))
        return cards

    # ------------------------------------------------------------------ #
    # Alert feed
    # ------------------------------------------------------------------ #
    def alert_feed(self, frame: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:
        """Build the operator alert feed, most severe/recent first.

        Args:
            frame: The wide dashboard table.
            limit: Maximum alerts to return.

        Returns:
            A list of alert-item dictionaries.
        """
        if "alert_level" not in frame.columns:
            candidate = frame[frame.get("anomaly_flag", False) == True]  # noqa: E712
        else:
            candidate = frame[frame["alert_level"].fillna("none") != "none"]

        if candidate.empty:
            return []

        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        candidate = candidate.copy()
        candidate["_rank"] = candidate.get("severity", "none").map(severity_rank).fillna(0)
        candidate = candidate.sort_values(["_rank", "cycle"], ascending=[False, False]).head(limit)

        items: List[Dict[str, Any]] = []
        for _, row in candidate.iterrows():
            item: Dict[str, Any] = {"engine_id": int(row["unit"]), "cycle": int(row["cycle"])}
            for column in [
                "alert_level", "severity", "health_state", "root_cause",
                "recommended_action", "confidence_score", "anomaly_score", "explanation_text",
                "twin_spread_mean", "deterioration_rate", "reliability_score",
                "uncertainty_score", "model_agreement", "detector_agreement",
                "evolution_pattern",
                "isolation_forest_score", "mahalanobis_score", "lstm_ae_score",
                "residual_rule_score", "threshold_used",
                "anomaly_persistence_cycles", "early_warning_score",
                "subsystem_attribution", "why", "which_sensors",
                "what_pattern", "how_severe", "shap_top_features",
            ]:
                if column in row.index and pd.notna(row[column]):
                    item[column] = row[column]
            items.append(self._sanitize(item))
        return items

    # ------------------------------------------------------------------ #
    # Timelines
    # ------------------------------------------------------------------ #
    def engine_timeline(self, frame: pd.DataFrame, engine_id: int) -> List[Dict[str, Any]]:
        """Build the full cycle-by-cycle timeline for one engine.

        Args:
            frame: The wide dashboard table.
            engine_id: Engine identifier.

        Returns:
            A list of timeline-point dictionaries, ordered by cycle.
        """
        engine_rows = frame[frame["unit"] == engine_id].sort_values("cycle")
        points: List[Dict[str, Any]] = []
        for _, row in engine_rows.iterrows():
            point: Dict[str, Any] = {"cycle": int(row["cycle"])}
            for column in [
                "health_index", "anomaly_score", "anomaly_flag",
                "health_state", "confidence_score", "operating_mode",
                "twin_spread_mean", "deterioration_rate", "reliability_score",
                "uncertainty_score", "model_agreement", "detector_agreement",
                "evolution_pattern",
                # Ground truth: 100 * clip(RUL, 0, rul_cap) / rul_cap. The health
                # index is computed without it, so plotting the two together
                # shows an unsupervised index against the true remaining-life
                # curve rather than against nothing.
                "health_index_target",
            ]:
                if column in row.index and pd.notna(row[column]):
                    point[column] = row[column]
            points.append(self._sanitize(point))
        return points

    def all_timelines(self, frame: pd.DataFrame, max_engines: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Build timelines for every engine, up to a cap.

        Args:
            frame: The wide dashboard table.
            max_engines: Maximum number of engines to include, to bound
                response size for large fleets.

        Returns:
            A mapping of engine id (as string) to its timeline.
        """
        engines = sorted(frame["unit"].unique())[:max_engines]
        return {str(engine): self.engine_timeline(frame, engine) for engine in engines}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    #: Columns that carry a stringified Python list in the wide CSV (several
    #: upstream stages stringify list-typed columns before writing CSVs) and
    #: must be parsed back into real lists before they reach a Pydantic model.
    #: Columns stored as a stringified Python literal in CSV that must be
    #: parsed back before serialising. `subsystem_attribution` is a list of
    #: dicts, not a flat list, so the parser has to accept both shapes.
    _LIST_COLUMNS = {
        "top_sensors",
        "which_sensors",
        "shap_top_features",
        "subsystem_attribution",
    }

    @classmethod
    def _sanitize(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NumPy scalars and stringified lists into native JSON types.

        Args:
            payload: A dictionary possibly containing NumPy scalar values and
                CSV-stringified list columns.

        Returns:
            The same dictionary with NumPy types converted and known list
            columns parsed back into actual lists.
        """
        clean: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in cls._LIST_COLUMNS and isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    clean[key] = list(parsed) if isinstance(parsed, (list, tuple)) else []
                except (ValueError, SyntaxError):
                    clean[key] = []
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                clean[key] = bool(value)
            else:
                clean[key] = value
        return clean

    def build_all(self) -> Dict[str, Any]:
        """Build every dashboard view in one call.

        Returns:
            A mapping with ``fleet``, ``engines``, ``alerts`` and ``timeline``.
        """
        frame = self.load()
        return {
            "fleet": self.fleet_statistics(frame),
            "engines": self.engine_summaries(frame),
            "alerts": self.alert_feed(frame),
            "timeline": self.all_timelines(frame),
        }


def main() -> None:
    """Build every dashboard view standalone and print a summary."""
    config = get_config()
    logger = get_logger(__name__, config)
    service = DashboardService(config)
    views = service.build_all()
    logger.info("Fleet stats: %s", views["fleet"])
    logger.info("Engines: %d cards, Alerts: %d items", len(views["engines"]), len(views["alerts"]))


if __name__ == "__main__":
    main()
