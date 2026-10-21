from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, read_csv
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

EXPLANATION_EVIDENCE_COLUMNS = [
    "shap_status", "shap_sensor", "shap_explained_output", "shap_base_value",
    "shap_prediction", "shap_commissioning_offset", "shap_additivity_error", "shap_reference_error",
    "sensor_evidence", "health_explanation_status", "health_explanation", "health_evidence",
    "cause_support_type", "cause_likelihood", "candidate_causes", "anomaly_margin", "hard_causal_claim",
]


_OPTIONAL_COLUMNS = [
    "health_index", "health_index_target", "health_score", "health_state", "health_trend",
    "remaining_health_pct", "anomaly_score", "anomaly_flag", "anomaly_type",
    "severity", "alert_level", "confidence_score", "root_cause",
    "top_sensors", "operating_mode", "explanation_text",
    "twin_spread_mean", "deterioration_rate", "reliability_score",
    "uncertainty_score", "model_agreement", "detector_agreement",
    "evolution_pattern",


    "isolation_forest_score", "mahalanobis_score", "lstm_ae_score",
    "residual_rule_score", "condition_consensus_score", "threshold_used", "anomaly_persistence_cycles",
    "is_anomalous", "early_warning_score",


    "subsystem_attribution", "why", "which_sensors", "what_pattern",
    "how_severe", "shap_top_features",
] + EXPLANATION_EVIDENCE_COLUMNS


class DashboardService:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def load(self) -> pd.DataFrame:


        return read_csv(
            self.config.paths.processed_path(self.config.paths.dashboard_data_csv),
            required_columns=get_identity_columns(),
            stage="dashboard_service",
        )

    @staticmethod
    def _latest_per_engine(frame: pd.DataFrame) -> pd.DataFrame:


        return (
            frame.sort_values("cycle")
            .groupby("unit", as_index=False, sort=True)
            .tail(1)
            .sort_values("unit")
            .reset_index(drop=True)
        )


    def fleet_statistics(self, frame: pd.DataFrame) -> Dict[str, Any]:


        latest = self._latest_per_engine(frame)
        state_counts = latest.get("health_state", pd.Series(dtype=object)).value_counts().to_dict()
        type_counts = frame.get("anomaly_type", pd.Series(dtype=object)).value_counts().to_dict()
        cause_counts = (
            frame.get("root_cause", pd.Series(dtype=object)).dropna().value_counts().head(5).to_dict()
        )


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


    def engine_summaries(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:


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


    def alert_feed(self, frame: pd.DataFrame, limit: int = 100) -> List[Dict[str, Any]]:


        if "alert_level" not in frame.columns:
            candidate = frame[frame.get("anomaly_flag", False) == True]
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
                "residual_rule_score", "condition_consensus_score", "threshold_used",
                "anomaly_persistence_cycles", "early_warning_score",
                "subsystem_attribution", "why", "which_sensors",
                "what_pattern", "how_severe", "shap_top_features",
            ] + EXPLANATION_EVIDENCE_COLUMNS:
                if column in row.index and pd.notna(row[column]):
                    item[column] = row[column]
            items.append(self._sanitize(item))
        return items


    def engine_timeline(self, frame: pd.DataFrame, engine_id: int) -> List[Dict[str, Any]]:


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


                "health_index_target",
            ]:
                if column in row.index and pd.notna(row[column]):
                    point[column] = row[column]
            points.append(self._sanitize(point))
        return points

    def all_timelines(self, frame: pd.DataFrame, max_engines: int = 50) -> Dict[str, List[Dict[str, Any]]]:


        engines = sorted(frame["unit"].unique())[:max_engines]
        return {str(engine): self.engine_timeline(frame, engine) for engine in engines}


    _LIST_COLUMNS = {
        "top_sensors",
        "which_sensors",
        "shap_top_features",
        "subsystem_attribution",
        "sensor_evidence",
        "candidate_causes",
    }

    @classmethod
    def _sanitize(cls, payload: Dict[str, Any]) -> Dict[str, Any]:


        clean: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "health_evidence" and isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    clean[key] = parsed if isinstance(parsed, dict) else {}
                except (ValueError, SyntaxError):
                    clean[key] = {}
            elif key in cls._LIST_COLUMNS and isinstance(value, str):
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


        frame = self.load()
        return {
            "fleet": self.fleet_statistics(frame),
            "engines": self.engine_summaries(frame),
            "alerts": self.alert_feed(frame),
            "timeline": self.all_timelines(frame),
        }


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    service = DashboardService(config)
    views = service.build_all()
    logger.info("Fleet stats: %s", views["fleet"])
    logger.info("Engines: %d cards, Alerts: %d items", len(views["engines"]), len(views["alerts"]))


if __name__ == "__main__":
    main()
