from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
    ResidualCalculator,
)
from app.services.Anomaly_Health_Monitoring.reasoning.root_cause_tracker import (
    RootCauseTracker,
)
from app.services.Anomaly_Health_Monitoring.reasoning.rule_engine import (
    ReasoningRuleEngine,
)
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import (
    SensorDependencyGraph,
)
from app.services.Anomaly_Health_Monitoring.reasoning.temporal_reasoning import (
    TemporalReasoner,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    get_residual_z_columns,
    sensor_from_residual,
    sensors_from_residual_frame,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    merge_on_keys,
    new_id,
    read_csv,
)


class RootCauseAnalyzer:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.graph = SensorDependencyGraph(self.config)
        self.rule_engine = ReasoningRuleEngine(self.config, graph=self.graph)
        self.temporal = TemporalReasoner(self.config)
        self.tracker = RootCauseTracker(self.config)
        self.top_k = int(self.config.reasoning.top_k_sensors)


    def _top_sensors_for_row(
        self, row: pd.Series, sensors: List[str]
    ) -> List[Tuple[str, float]]:


        z_columns = get_residual_z_columns(sensors)
        pairs = [
            (sensor_from_residual(col), float(row.get(col, 0.0)))
            for col in z_columns
            if col in row.index
        ]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return pairs[: self.top_k]

    def analyze_row(
        self,
        row: pd.Series,
        engine_history: pd.DataFrame,
        sensors: List[str],
        score_column: str = "anomaly_score",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:


        ranked = self._top_sensors_for_row(row, sensors)
        top_sensor_names = [name for name, _ in ranked]

        temporal = self.temporal.analyze(engine_history, score_column, threshold)
        trend_rising = temporal["trend_slope"] > self.config.reasoning.trend_slope_eps

        health_state = str(row.get("health_state", "Healthy"))
        best_rule = self.rule_engine.evaluate(top_sensor_names, trend_rising, health_state)
        candidates = self.rule_engine.rank_candidates(top_sensor_names, trend_rising, health_state)

        memory_match = self.tracker.match(top_sensor_names, subsystem=best_rule.subsystem)

        record: Dict[str, Any] = {
            "unit": int(row["unit"]),
            "cycle": int(row["cycle"]),
            "anomaly_type": str(row.get("anomaly_type", "fault")),
            "severity": str(row.get("severity", "none")),
            "top_sensors": top_sensor_names,
            "sensor_contributions": [
                {
                    "sensor": name,
                    "residual_z": value,
                    "subsystem": self.graph.subsystem_of(name),
                }
                for name, value in ranked
            ],
            "subsystem": best_rule.subsystem,
            "root_cause": best_rule.cause,
            "cause_likelihood": best_rule.likelihood,
            "candidate_causes": [
                {
                    "cause": c.cause,
                    "likelihood": c.likelihood,
                    "subsystem": c.subsystem,
                    "evidence": c.evidence,
                    "rule_id": c.rule_id,
                    "inspection_focus": c.inspection_focus,
                    "recommended_action": c.recommended_action,
                }
                for c in candidates[:3]
            ],
            "evolution_pattern": temporal["evolution_pattern"],
            "onset_cycle": temporal["onset_cycle"],
            "duration_cycles": temporal["duration_cycles"],
            "trend_slope": temporal["trend_slope"],
            "similar_past_anomaly_id": memory_match["anomaly_id"],
            "similarity_score": memory_match["similarity"],
            "times_seen_before": memory_match["times_seen_before"],
            "previously_false_alarm": memory_match["previously_false_alarm"],
            "recommended_action": best_rule.recommended_action,
            "inspection_focus": best_rule.inspection_focus,
            "hard_causal_claim": False,
            "anomaly_id": new_id("anom"),
        }
        return record


    def analyze(
        self,
        anomalies: pd.DataFrame,
        residuals: pd.DataFrame,
        only_anomalies: bool = True,
    ) -> pd.DataFrame:


        merged = merge_on_keys(
            anomalies, residuals, keys=get_identity_columns(), how="left"
        )
        merged = merged.sort_values(get_identity_columns()).reset_index(drop=True)

        sensors = sensors_from_residual_frame(residuals)

        target_rows = (
            merged[merged.get("anomaly_type", "none") == "fault"]
            if only_anomalies
            else merged
        )

        score_column = "anomaly_score" if "anomaly_score" in merged.columns else "residual_abs_mean"
        threshold = float(target_rows["threshold_used"].iloc[0]) if "threshold_used" in target_rows.columns and len(target_rows) else 0.5

        records: List[Dict[str, Any]] = []
        for unit, unit_rows in target_rows.groupby("unit", sort=False):
            engine_history = merged[merged["unit"] == unit].sort_values("cycle")
            for _, row in unit_rows.iterrows():
                history_up_to_now = engine_history[engine_history["cycle"] <= row["cycle"]]
                records.append(
                    self.analyze_row(row, history_up_to_now, sensors, score_column, threshold)
                )

        if not records:
            self.logger.info("No rows qualified for root-cause reasoning.")
            return pd.DataFrame(
                columns=get_identity_columns() + ["root_cause", "subsystem", "anomaly_type"]
            )

        result = pd.DataFrame(records)


        for column in ("top_sensors", "sensor_contributions", "candidate_causes"):
            result[column] = result[column].apply(lambda value: str(value))

        self.logger.info(
            "Reasoned about %d/%d row(s) (%.1f%% of the input)",
            len(result),
            len(merged),
            100.0 * len(result) / max(len(merged), 1),
        )
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.root_cause_analysis_csv
        )
        if not frame.empty:
            verify_identity_columns(frame, context="root_cause_analysis output")
        return atomic_write_csv(frame, target, config=self.config)

    def run(
        self,
        anomalies: Optional[pd.DataFrame] = None,
        residuals: Optional[pd.DataFrame] = None,
        only_anomalies: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        anomaly_source = anomalies
        if anomaly_source is None:
            anomaly_source = read_csv(
                self.config.paths.processed_path(self.config.paths.anomaly_fusion_csv),
                required_columns=get_identity_columns(),
                stage="root_cause_analyzer",
            )
        residual_source = residuals
        if residual_source is None:
            residual_source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="root_cause_analyzer",
            )

        analyzed = self.analyze(anomaly_source, residual_source, only_anomalies)
        output = self.save(analyzed)

        cause_distribution = (
            analyzed["root_cause"].value_counts().to_dict() if "root_cause" in analyzed.columns else {}
        )
        subsystem_distribution = (
            analyzed["subsystem"].value_counts().to_dict() if "subsystem" in analyzed.columns else {}
        )
        evolution_distribution = (
            analyzed["evolution_pattern"].value_counts().to_dict()
            if "evolution_pattern" in analyzed.columns
            else {}
        )

        report: Dict[str, Any] = {
            "rows_input": int(len(anomaly_source)),
            "rows_reasoned": int(len(analyzed)),
            "engines": int(analyzed["unit"].nunique()) if len(analyzed) else 0,
            "cause_distribution": {str(k): int(v) for k, v in cause_distribution.items()},
            "subsystem_distribution": {str(k): int(v) for k, v in subsystem_distribution.items()},
            "evolution_distribution": {str(k): int(v) for k, v in evolution_distribution.items()},
            "memory_matches": int(analyzed.get("similar_past_anomaly_id", pd.Series(dtype=object)).notna().sum())
            if len(analyzed)
            else 0,
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "reasoning.json")
        self.logger.info("Root-cause analysis report: %s", report)
        return analyzed, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    analyzer = RootCauseAnalyzer(config)
    analyzed, report = analyzer.run()
    logger.info("root_cause_analysis.csv written: %d row(s)", report["rows_reasoned"])
    if len(analyzed):
        print(analyzed[["unit", "cycle", "root_cause", "subsystem", "evolution_pattern"]].head(10).to_string())


if __name__ == "__main__":
    main()
