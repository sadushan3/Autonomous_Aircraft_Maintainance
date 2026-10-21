from __future__ import annotations

from pathlib import Path
import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.explainability.sensor_residual_ranking import (
    SensorResidualRanker,
)
from app.services.Anomaly_Health_Monitoring.explainability.subsystem_explainer import (
    SubsystemExplainer,
)
from app.services.Anomaly_Health_Monitoring.explainability.health_explainer import HealthExplainer
from app.services.Anomaly_Health_Monitoring.explainability.monitoring_reference_explainer import (
    MonitoringReferenceExplainer, representative_sample,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    sensors_from_residual_frame,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
)

_SEVERITY_PHRASES: Dict[str, str] = {
    "none": "no significant deviation",
    "low": "a mild deviation from expected behaviour",
    "medium": "a moderate deviation warranting attention",
    "high": "a serious deviation requiring prompt investigation",
    "critical": "a critical deviation requiring immediate action",
}


class ExplanationGenerator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.ranker = SensorResidualRanker(self.config)
        self.subsystem_explainer = SubsystemExplainer(self.config)
        self.last_report: Dict[str, Any] = {}

    @staticmethod
    def _number(row: pd.Series, name: str) -> Optional[float]:
        try:
            value = float(row.get(name))
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse(value: Any, default: Any) -> Any:
        if isinstance(value, (list, dict)):
            return value
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return default

    @staticmethod
    def _overlay(base: pd.DataFrame, evidence: Optional[pd.DataFrame]) -> pd.DataFrame:
        if evidence is None or evidence.empty:
            return base
        keys = get_identity_columns()
        if evidence.duplicated(keys).any():
            raise ValueError("Explanation evidence contains duplicate engine/cycle identities.")
        extra = [c for c in evidence if c not in base or c in keys]
        return base.merge(evidence[extra], on=keys, how="left", validate="one_to_one")


    def _build_narrative(
        self,
        row: pd.Series,
        top_sensors: List[Dict[str, Any]],
        subsystem_attribution: List[Dict[str, Any]],
    ) -> Dict[str, str]:


        severity = str(row.get("severity", "none"))
        root_cause = str(row.get("root_cause", "No fault indication"))
        subsystem = str(row.get("subsystem", "unknown"))
        evolution = str(row.get("evolution_pattern", "stable"))
        engine = int(row.get("unit", 0))
        cycle = int(row.get("cycle", 0))

        sensor_names = [s["sensor"] for s in top_sensors]
        sensor_phrase = ", ".join(sensor_names[:5]) if sensor_names else "no sensor in particular"

        dominant_subsystem_label = (
            subsystem_attribution[0]["subsystem_label"] if subsystem_attribution else subsystem
        )

        why = (
            f"Engine {engine} at cycle {cycle} deviated from its digital-twin "
            f"monitoring reference, with a pattern most consistent with {root_cause.lower()} "
            f"(subsystem: {dominant_subsystem_label})."
        )
        score, threshold = self._number(row, "anomaly_score"), self._number(row, "threshold_used")
        if score is not None and threshold is not None:
            relation = "above" if score >= threshold else "below"
            why += f" Anomaly score {score:.4f} is {relation} threshold {threshold:.4f} (margin {score - threshold:+.4f})."
        support = self._number(row, "cause_likelihood")
        if support is not None:
            why += f" Rule support is {support:.2f}; this is a suspected cause, not a calibrated fault probability."
        which_sensors = (
            f"The deviation is led by {sensor_phrase}, ranked by how far each "
            f"reading sits from its mode-normalised healthy expectation."
        )
        evidence = []
        for item in top_sensors[:3]:
            sensor = item["sensor"]
            observed = self._number(row, sensor)
            expected = self._number(row, f"reference_expected_{sensor}")
            if expected is None:
                expected = self._number(row, f"expected_{sensor}")
            if observed is not None and expected is not None:
                evidence.append(f"{sensor}: observed {observed:.3f}, reference {expected:.3f}, residual z {item['residual_z']:+.2f}")
        if evidence:
            which_sensors += " In mode-scaled units: " + "; ".join(evidence) + "."
        pattern_word = {
            "abrupt": "an abrupt, step-like change",
            "gradual": "a gradual, progressively worsening trend",
            "intermittent": "an intermittent pattern that repeatedly appears and recedes",
            "stable": "a stable deviation with no clear trend",
        }.get(evolution, "an unclassified pattern")
        what_pattern = (
            f"Across recent cycles this shows {pattern_word}, concentrated on "
            f"the {dominant_subsystem_label} subsystem."
        )
        onset, duration, slope = (self._number(row, c) for c in ("onset_cycle", "duration_cycles", "trend_slope"))
        if onset is not None and duration is not None:
            what_pattern += f" Detected onset: cycle {int(onset)}; duration: {int(duration)} cycles."
        if slope is not None:
            what_pattern += f" Recent anomaly-score slope: {slope:+.5f} per cycle."
        how_severe = (
            f"Severity is assessed as '{severity}' -- {_SEVERITY_PHRASES.get(severity, 'undetermined severity')}."
        )

        explanation_text = f"{why} {which_sensors} {what_pattern} {how_severe}"
        return {
            "why": why,
            "which_sensors": which_sensors,
            "what_pattern": what_pattern,
            "how_severe": how_severe,
            "explanation_text": explanation_text,
        }


    def generate(
        self,
        root_causes: pd.DataFrame,
        residuals: pd.DataFrame,
        include_shap: bool = True,
        engine_id: Optional[int] = None,
        max_rows: Optional[int] = None,
        scaled: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.DataFrame] = None,
        anomalies: Optional[pd.DataFrame] = None,
        health: Optional[pd.DataFrame] = None,
        ensemble: Any = None,
    ) -> pd.DataFrame:


        self.last_report = {"shap_requested": bool(include_shap), "shap_computed": False,
                            "shap_rows_computed": 0, "shap_rows_failed": 0,
                            "explainer_used": "none", "shap_errors": []}
        if root_causes.empty:
            self.logger.info("No reasoned anomalies to explain.")
            return pd.DataFrame(columns=get_identity_columns() + ["explanation_text"])

        keys = get_identity_columns()
        if root_causes.duplicated(keys).any():
            raise ValueError("Explanation input contains duplicate engine/cycle identities.")
        merged = self._overlay(root_causes, residuals)
        for data in (anomalies, health, scaled, predictions):
            merged = self._overlay(merged, data)
        if engine_id is not None:
            merged = merged[merged["unit"] == engine_id]
        if max_rows is not None:
            if max_rows < 1:
                raise ValueError("max_rows must be positive.")
            merged = merged.sort_values(keys).head(max_rows)

        sensors = sensors_from_residual_frame(residuals)

        local_shap: Dict[Tuple[int, int], Dict[str, Any]] = {}
        selected = representative_sample(merged, min(self.config.explainability.local_shap_max_rows,
                                                     self.config.explainability.max_explain_rows),
                                         self.config.data.random_state)
        selected_ids = set(zip(selected.unit, selected.cycle))
        if include_shap and not selected.empty:
            try:
                if scaled is None or predictions is None:
                    raise ValueError("Matching scaled context/history and predictions were not supplied.")
                selected = selected.copy()
                selected["shap_sensor"] = [self.ranker.rank_row(r, sensors)[0]["sensor"]
                                           for _, r in selected.iterrows()]
                if ensemble is None:
                    from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
                    ensemble = EnsembleTwin(self.config).load()
                local_shap, details = MonitoringReferenceExplainer(ensemble, self.config).explain(
                    scaled, selected[keys + ["shap_sensor"]], predictions)
                self.last_report.update(details)
            except Exception as exc:
                self.logger.warning("Local monitoring SHAP unavailable: %s", exc)
                self.last_report["shap_errors"].append(str(exc))
                self.last_report["shap_rows_failed"] = len(selected)
        self.last_report["shap_rows_computed"] = len(local_shap)
        self.last_report["shap_rows_attempted"] = len(selected) if include_shap else 0
        self.last_report["shap_computed"] = bool(local_shap)
        self.last_report["shap_coverage_fraction"] = len(local_shap) / len(merged) if len(merged) else 0.

        health_details = HealthExplainer(self.config).explain(residuals, health) if health is not None else {}

        records: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            top_sensors = self.ranker.rank_row(row, sensors)
            subsystem_attribution = self.subsystem_explainer.explain_row(row, sensors)
            narrative = self._build_narrative(row, top_sensors, subsystem_attribution)
            identity = (int(row.unit), int(row.cycle))
            shap_record = local_shap.get(identity, {
                "shap_status": ("disabled" if not include_shap else
                                "failed" if identity in selected_ids else "not_sampled"),
                "shap_top_features": [], "shap_explained_output": "monitoring_reference_expected_sensor",
            })
            health_record = health_details.get(identity, {"health_explanation_status": "unavailable", "health_evidence": {}})
            health_evidence = health_record["health_evidence"]
            health_text = "Health-index evidence is unavailable for this event."
            if health_record["health_explanation_status"] == "computed":
                names = ", ".join(s["sensor"] for s in health_evidence["top_sensors"][:3])
                health_text = (
                    f"Health index is {health_evidence['health_index']:.2f}/100; unsmoothed index is "
                    f"{health_evidence['raw_health_index']:.2f}/100. {names} contribute most to the weighted "
                    f"sensor deviation. Smoothing combines {health_evidence['current_cycle_term']:.2f} current "
                    f"health points and {health_evidence['history_term']:.2f} points retained from prior cycles."
                )
                if "health_score" in health_evidence:
                    lower, upper = health_evidence["state_lower_bound"], health_evidence["state_upper_bound"]
                    interval = f"at least {lower:.1f}" if upper is None else f"from {lower:.1f} to below {upper:.1f}"
                    health_text += (
                        f" Trend adjustment {health_evidence['health_score_trend_adjustment']:+.2f} gives a health "
                        f"score of {health_evidence['health_score']:.2f}, classified as {health_evidence['health_state']} "
                        f"using scores {interval}."
                    )
            elif health_record["health_explanation_status"] == "reference_mismatch":
                health_text = "Saved health index does not match current calibration/history; refresh inference outputs."
            sensor_evidence = []
            for item in top_sensors:
                name = item["sensor"]
                expected = self._number(row, f"reference_expected_{name}")
                if expected is None:
                    expected = self._number(row, f"expected_{name}")
                sensor_evidence.append({**item, "observed": self._number(row, name), "expected": expected,
                                        "residual": self._number(row, f"residual_{name}"), "units": "mode_scaled"})

            record = {
                "unit": int(row["unit"]),
                "cycle": int(row["cycle"]),
                "why": narrative["why"],
                "which_sensors": str([s["sensor"] for s in top_sensors]),
                "what_pattern": narrative["what_pattern"],
                "how_severe": narrative["how_severe"],
                "explanation_text": narrative["explanation_text"] + " " + health_text,
                "recommended_action": row.get("recommended_action", "Continue monitoring"),
                "anomaly_type": row.get("anomaly_type", "none"),
                "severity": row.get("severity", "none"),
                "health_state": row.get("health_state", "Healthy"),
                **shap_record,
                "shap_top_features": str(shap_record["shap_top_features"]),
                **health_record,
                "health_evidence": str(health_evidence),
                "health_explanation": health_text,
                "sensor_evidence": str(sensor_evidence),
                "cause_likelihood": self._number(row, "cause_likelihood"),
                "cause_support_type": "heuristic_rule_support",
                "candidate_causes": str(self._parse(row.get("candidate_causes"), [])),
                "anomaly_score": self._number(row, "anomaly_score"),
                "threshold_used": self._number(row, "threshold_used"),
                "anomaly_margin": (float(row.anomaly_score - row.threshold_used)
                                   if self._number(row, "anomaly_score") is not None and self._number(row, "threshold_used") is not None else None),
                "hard_causal_claim": False,
                "subsystem_attribution": str(subsystem_attribution),
            }
            records.append(record)

        result = pd.DataFrame(records)
        self.logger.info(
            "Generated %d explanation(s), %d with local SHAP", len(result), len(local_shap)
        )
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.explanation_reports_csv
        )
        if not frame.empty:
            verify_identity_columns(frame, context="explanation_reports output")
        return atomic_write_csv(frame, target, config=self.config)

    def run(
        self,
        root_causes: Optional[pd.DataFrame] = None,
        residuals: Optional[pd.DataFrame] = None,
        include_shap: bool = True,
        engine_id: Optional[int] = None,
        max_rows: Optional[int] = None,
        only_anomalies: bool = True,
        persist: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        root_cause_source = root_causes
        if root_cause_source is None:
            path = self.config.paths.processed_path(self.config.paths.root_cause_analysis_csv)
            root_cause_source = (
                read_csv(path, stage="explanation_generator") if path.is_file() else pd.DataFrame()
            )
        residual_source = residuals
        if residual_source is None:
            residual_source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="explanation_generator",
            )

        def optional(filename: str) -> Optional[pd.DataFrame]:
            path = self.config.paths.processed_path(filename)
            return read_csv(path, stage="explanation_generator") if path.is_file() else None

        if only_anomalies and "anomaly_type" in root_cause_source:
            root_cause_source = root_cause_source[root_cause_source.anomaly_type.eq("fault")]
        explanations = self.generate(
            root_cause_source, residual_source, include_shap, engine_id, max_rows,
            scaled=optional(self.config.paths.scaled_features_csv),
            predictions=optional(self.config.paths.twin_predictions_csv),
            anomalies=optional(self.config.paths.anomaly_fusion_csv),
            health=optional(self.config.paths.health_states_csv),
        )
        output = self.save(explanations) if persist else ""

        report: Dict[str, Any] = {
            "rows_input": int(len(root_cause_source)),
            "rows_explained": int(len(explanations)),
            "output_path": str(output),
            "engines": int(explanations.unit.nunique()) if len(explanations) else 0,
            **self.last_report,
        }
        if persist:
            atomic_write_json(report, self.config.paths.metrics / "explainability.json")
        self.logger.info("Explanation generation report: %s", report)
        return explanations, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    generator = ExplanationGenerator(config)
    explanations, report = generator.run()
    logger.info("explanation_reports.csv written: %d row(s)", report["rows_explained"])
    if len(explanations):
        print(explanations[["unit", "cycle", "explanation_text"]].head(5).to_string())


if __name__ == "__main__":
    main()
