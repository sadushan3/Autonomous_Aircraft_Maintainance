"""
Explanation generation: the explainability-layer orchestrator.

Turns residual rankings, subsystem attribution and root-cause records into the
four-question narrative the specification requires -- WHY, WHICH sensors, WHAT
pattern, HOW severe -- plus a single readable paragraph combining them, and
writes ``explanation_reports.csv``.

SHAP is deliberately not run for every row here: it is expensive, and the
sensor-residual ranking already answers "which sensors" cheaply and exactly
from the same numbers the anomaly was flagged on.  SHAP is computed on a
sample against the twin (via :mod:`shap_explainer`) and folded in as
``shap_top_features`` when requested, giving a global sanity check on what the
twin's expectations were keying off without paying its cost on every row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.explainability.sensor_residual_ranking import (
    SensorResidualRanker,
)
from app.services.Anomaly_Health_Monitoring.explainability.shap_explainer import (
    ShapExplainer,
)
from app.services.Anomaly_Health_Monitoring.explainability.subsystem_explainer import (
    SubsystemExplainer,
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
    merge_on_keys,
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
    """Builds the WHY/WHICH/WHAT/HOW narrative for reasoned anomalies."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the generator and its collaborators.

        Args:
            config: Configuration supplying explainability settings.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.ranker = SensorResidualRanker(self.config)
        self.subsystem_explainer = SubsystemExplainer(self.config)
        self.shap_explainer = ShapExplainer(self.config)

    # ------------------------------------------------------------------ #
    # Narrative construction
    # ------------------------------------------------------------------ #
    def _build_narrative(
        self,
        row: pd.Series,
        top_sensors: List[Dict[str, Any]],
        subsystem_attribution: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Compose the WHY / WHICH / WHAT / HOW text for one row.

        Args:
            row: A row from the merged reasoning + residual frame.
            top_sensors: Output of :meth:`SensorResidualRanker.rank_row`.
            subsystem_attribution: Output of
                :meth:`SubsystemExplainer.explain_row`.

        Returns:
            A mapping with ``why``, ``which_sensors``, ``what_pattern``,
            ``how_severe`` and the combined ``explanation_text``.
        """
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
            f"expectation, most consistent with {root_cause.lower()} "
            f"(subsystem: {dominant_subsystem_label})."
        )
        which_sensors = (
            f"The deviation is led by {sensor_phrase}, ranked by how far each "
            f"reading sits from its mode-normalised healthy expectation."
        )
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

    # ------------------------------------------------------------------ #
    # Batch orchestration
    # ------------------------------------------------------------------ #
    def generate(
        self,
        root_causes: pd.DataFrame,
        residuals: pd.DataFrame,
        include_shap: bool = True,
        engine_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate explanation records for every reasoned row.

        Args:
            root_causes: Output of :class:`RootCauseAnalyzer`
                (``root_cause_analysis.csv``).
            residuals: Residuals frame carrying ``residual_z_sensor_*`` columns.
            include_shap: Whether to also compute a global SHAP ranking from
                the trained twin ensemble and attach it to every row.
            engine_id: Optional restriction to a single engine.

        Returns:
            A frame of explanation records, one row per reasoned anomaly.
        """
        if root_causes.empty:
            self.logger.info("No reasoned anomalies to explain.")
            return pd.DataFrame(columns=get_identity_columns() + ["explanation_text"])

        merged = merge_on_keys(
            root_causes, residuals, keys=get_identity_columns(), how="left"
        )
        if engine_id is not None:
            merged = merged[merged["unit"] == engine_id]

        sensors = sensors_from_residual_frame(residuals)

        shap_ranking: List[Dict[str, Any]] = []
        explainer_used = "none"
        if include_shap:
            shap_ranking, explainer_used = self._compute_global_shap(residuals)

        records: List[Dict[str, Any]] = []
        for _, row in merged.iterrows():
            top_sensors = self.ranker.rank_row(row, sensors)
            subsystem_attribution = self.subsystem_explainer.explain_row(row, sensors)
            narrative = self._build_narrative(row, top_sensors, subsystem_attribution)

            record = {
                "unit": int(row["unit"]),
                "cycle": int(row["cycle"]),
                "why": narrative["why"],
                "which_sensors": str([s["sensor"] for s in top_sensors]),
                "what_pattern": narrative["what_pattern"],
                "how_severe": narrative["how_severe"],
                "explanation_text": narrative["explanation_text"],
                "recommended_action": row.get("recommended_action", "Continue monitoring"),
                "anomaly_type": row.get("anomaly_type", "none"),
                "severity": row.get("severity", "none"),
                "health_state": row.get("health_state", "Healthy"),
                "shap_top_features": str(shap_ranking[:5]),
                "subsystem_attribution": str(subsystem_attribution),
            }
            records.append(record)

        result = pd.DataFrame(records)
        self.logger.info(
            "Generated %d explanation(s) (explainer=%s)", len(result), explainer_used
        )
        return result

    @staticmethod
    def _most_informative_target(twin: Any, scaled: pd.DataFrame) -> Optional[str]:
        """Pick the twin target worth explaining: the one that actually varies.

        This used to be ``twin.target_names[0]``, i.e. ``sensor_1`` -- one of
        four channels that are constant within an operating mode and therefore
        have exactly zero variance once per-mode scaling is applied. The twin
        predicts them perfectly from the mode one-hots alone, so every SHAP
        value came back 0.0 with ``mode_1..mode_5`` as the "top" features: a
        ranking that was computed successfully and meant nothing.

        Selecting by variance separates the two groups cleanly -- the degenerate
        channels sit at 0.0 while informative ones sit near 1.0 -- and needs no
        evaluation artefact to be present.

        Args:
            twin: A trained twin exposing ``target_names``.
            scaled: The scaled feature frame the SHAP run will sample from.

        Returns:
            The highest-variance modelled sensor, or ``None`` when every target
            is constant (nothing meaningful could be attributed).
        """
        candidates = [c for c in twin.target_names if c in scaled.columns]
        if not candidates:
            return None
        variances = scaled[candidates].var(numeric_only=True)
        variances = variances[variances > 0]
        if variances.empty:
            return None
        return str(variances.idxmax())

    def _compute_global_shap(self, residuals: pd.DataFrame) -> Tuple[List[Dict[str, Any]], str]:
        """Compute a global SHAP ranking from the trained ensemble, tolerating failure.

        Args:
            residuals: Residuals frame (used only to size the request; the
                SHAP computation itself runs against the scaled-feature CSV).

        Returns:
            A ``(ranking, explainer_kind)`` tuple; ranking is empty and kind is
            ``'none'`` when SHAP could not be computed.
        """
        try:
            from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import (
                RandomForestTwin,
            )

            scaled_path = self.config.paths.processed_path(self.config.paths.scaled_features_csv)
            if not scaled_path.is_file():
                return [], "none"
            scaled = read_csv(scaled_path, required_columns=get_identity_columns(), stage="explanation_generator")

            twin = RandomForestTwin(self.config).load()
            sensor = self._most_informative_target(twin, scaled)
            if sensor is None:
                return [], "none"
            result = self.shap_explainer.explain_twin(
                twin, scaled, sensor=sensor,
                max_rows=self.config.explainability.max_explain_rows,
            )
            return result["global_ranking"], result["kind_used"]
        except Exception as exc:  # noqa: BLE001 - explanation must not crash the stage
            self.logger.warning("Global SHAP computation skipped: %s", exc)
            return [], "none"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``explanation_reports.csv`` atomically.

        Args:
            frame: Explanation records.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
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
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the explainability stage.

        Args:
            root_causes: Optional in-memory input; otherwise
                ``root_cause_analysis.csv`` is read from disk.
            residuals: Optional in-memory input; otherwise ``residuals.csv`` is
                read from disk.
            include_shap: Whether to compute the global SHAP ranking.

        Returns:
            A ``(explanations, report)`` tuple.
        """
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

        explanations = self.generate(root_cause_source, residual_source, include_shap)
        output = self.save(explanations)

        report: Dict[str, Any] = {
            "rows_input": int(len(root_cause_source)),
            "rows_explained": int(len(explanations)),
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "explainability.json")
        self.logger.info("Explanation generation report: %s", report)
        return explanations, report


def main() -> None:
    """Run explanation generation standalone against the reasoning output."""
    config = get_config()
    logger = get_logger(__name__, config)
    generator = ExplanationGenerator(config)
    explanations, report = generator.run()
    logger.info("explanation_reports.csv written: %d row(s)", report["rows_explained"])
    if len(explanations):
        print(explanations[["unit", "cycle", "explanation_text"]].head(5).to_string())


if __name__ == "__main__":
    main()
