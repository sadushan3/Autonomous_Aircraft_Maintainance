"""
Anomaly fusion and the operational-vs-fault gate.

This is the orchestrator of the anomaly-detection layer.  It trains (or
reloads) the enabled subset of four detector families, fuses their normalised scores into a single
``anomaly_score``, applies the adaptive threshold from the feedback loop, and
then runs the hard post-filter that is the core novelty requirement of this
component: distinguishing a genuine fault from an operating-condition change.

Fusion is a configured weighted sum:

    anomaly_score = sum(weight_d * score_d for d in detectors)

The operational-vs-fault gate then reclassifies any row that is:

* flagged by the fusion threshold, **and**
* within the regime-transition guard window (``is_mode_transition``), **and**
* not sustained -- fewer than ``persistence_min_cycles`` consecutive flagged
  cycles for the engine.

as ``anomaly_type='operational'`` rather than ``'fault'``.  Operational
detections are suppressed from escalating to the reasoning layer but are still
recorded, so a genuine fault that happens to coincide with a regime change and
*does* persist past the guard window is correctly still caught.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.isolation_forest_detector import (
    SCORE_COLUMN as IFOREST_SCORE_COLUMN,
    IsolationForestDetector,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.lstm_autoencorder import (
    SCORE_COLUMN as LSTM_SCORE_COLUMN,
    LSTMAutoencoderDetector,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.mahalanobis_detector import (
    SCORE_COLUMN as MAHAL_SCORE_COLUMN,
    MahalanobisDetector,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.residual_anomaly_detector import (
    SCORE_COLUMN as RULE_SCORE_COLUMN,
    ResidualAnomalyDetector,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.condition_consensus import (
    SCORE_COLUMN as CONSENSUS_SCORE_COLUMN,
    ConditionConsensusDetector,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.early_warning_score import (
    EarlyWarningScorer,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.severity_classifier import (
    SeverityClassifier,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import (
    MODE_TRANSITION_COLUMN,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.twin_comparator import (
    TwinComparator,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    present_residual_columns,
    present_residual_z_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    merge_on_keys,
    percentile_threshold,
    read_csv,
    read_json,
    engine_wise_split,
)

#: Fused, weighted anomaly score in [0, 1].
ANOMALY_SCORE_COLUMN = "anomaly_score"

#: Boolean flag from comparing the fused score to the adaptive threshold.
ANOMALY_FLAG_COLUMN = "anomaly_flag"

#: 'fault' | 'operational' | 'none' -- the outcome of the gate.
ANOMALY_TYPE_COLUMN = "anomaly_type"

#: The adaptive threshold actually applied to this row.
THRESHOLD_USED_COLUMN = "threshold_used"

#: How many consecutive cycles (inclusive) this row's flag has persisted.
PERSISTENCE_COLUMN = "anomaly_persistence_cycles"

_DETECTOR_REGISTRY: Dict[str, str] = {
    "isolation_forest": IFOREST_SCORE_COLUMN,
    "mahalanobis": MAHAL_SCORE_COLUMN,
    "lstm_autoencoder": LSTM_SCORE_COLUMN,
    "residual_rule": RULE_SCORE_COLUMN,
    "condition_consensus": CONSENSUS_SCORE_COLUMN,
}


class AnomalyFusion:
    """Trains all detectors, fuses scores, and classifies fault vs operational."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the fusion orchestrator.

        Args:
            config: Configuration supplying fusion weights and gate parameters.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.detectors: Dict[str, Any] = {
            "isolation_forest": IsolationForestDetector(self.config),
            "mahalanobis": MahalanobisDetector(self.config),
            "lstm_autoencoder": LSTMAutoencoderDetector(self.config),
            "residual_rule": ResidualAnomalyDetector(self.config),
            "condition_consensus": ConditionConsensusDetector(self.config),
        }
        self.detectors = {name: detector for name, detector in self.detectors.items()
                          if self.config.anomaly.fusion_weights.get(name, 0.0) > 0}
        self.feature_columns: List[str] = []
        self.threshold_provider: Any = None
        self.severity_classifier = SeverityClassifier(self.config)
        self.early_warning_scorer = EarlyWarningScorer(self.config)
        #: Fusion threshold calibrated from the healthy population's own
        #: fused score, the same way every individual detector calibrates its
        #: own threshold. None until train() (or load()) runs.
        self.calibrated_fusion_threshold: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Detector training / loading
    # ------------------------------------------------------------------ #
    def _feature_columns(self, frame: pd.DataFrame) -> List[str]:
        """Resolve the residual z-score columns used as the detector feature space.

        Args:
            frame: Residuals frame.

        Returns:
            The residual z-score columns present.
        """
        if self.config.anomaly.feature_space == "raw":
            return present_residual_columns(frame)
        return present_residual_z_columns(frame)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Select the same healthy region every individual detector trains on.

        Args:
            frame: Residuals frame.

        Returns:
            A boolean mask.
        """
        from app.utils.Anomaly_Health_Monitoring.model_utils import RUL_COLUMN

        settings = self.config.healthy_region
        if self.config.anomaly.feature_space == "raw" and RUL_COLUMN in frame:
            return (frame[RUL_COLUMN].astype(float) >= self.config.anomaly.calibration_healthy_rul) & (
                frame["cycle"] > self.config.twin.commissioning_cycles
            )
        if RUL_COLUMN in frame.columns:
            rul_ok = frame[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=frame.index)
        max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = frame["cycle"].astype(float) <= (float(settings.cycle_fraction) * max_cycle)
        return rul_ok & cycle_ok

    def _calibrate_fusion_threshold(self, frame: pd.DataFrame) -> float:
        """Calibrate the fusion threshold against the healthy population's own fused score.

        Every individual detector already sets its own flagging threshold from
        the healthy region's score distribution (typically its 99th
        percentile); the fused score needs the identical treatment.  A flat
        constant here would be meaningless the moment any detector's score
        scale changes, since "0.5" only means something relative to what
        healthy behaviour actually produces on that scale.

        Args:
            frame: Residuals frame (the same one detectors were trained on).

        Returns:
            The calibrated threshold from healthy calibration engines.

        Raises:
            AnomalyHealthError: If calibration data or valid scores are absent.
        """
        healthy = frame.loc[self._healthy_mask(frame)]
        if healthy.empty:
            raise AnomalyHealthError("No healthy calibration rows are available; a fusion cutoff cannot be fitted.")
        if list(self.detectors) == ["condition_consensus"]:
            # This score already owns a separate-engine healthy-tail cutoff
            # and conservative safety margin. A fresh p99 would undo that
            # margin and silently change the validated false-alert policy.
            return float(self.detectors["condition_consensus"].validate_calibration(frame))
        try:
            # Score complete histories before applying the healthy label mask.
            healthy_scores = self.score(frame)
            healthy_fused = self.fuse(healthy_scores, fusion_threshold=0.0)
            threshold = percentile_threshold(
                healthy_fused.loc[self._healthy_mask(frame).to_numpy(), ANOMALY_SCORE_COLUMN].to_numpy(),
                self.config.anomaly.fusion_threshold_percentile,
            )
            if not np.isfinite(threshold):
                raise ValueError("non-finite threshold")
        except Exception as exc:  # noqa: BLE001
            raise AnomalyHealthError("Fusion threshold calibration failed; the previous artifacts were retained.") from exc

        self.logger.info(
            "Fusion threshold calibrated from %d healthy row(s) at p%.0f: %.4f",
            len(healthy),
            self.config.anomaly.fusion_threshold_percentile,
            threshold,
        )
        return float(threshold)

    def train(self, frame: pd.DataFrame, detector_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fit every requested detector, tolerating individual failures.

        Args:
            frame: Residuals frame.
            detector_names: Subset of detectors to (re)train; defaults to all.

        Returns:
            A summary of which detectors trained and their thresholds.

        Raises:
            AnomalyHealthError: If every detector fails to train.
        """
        if "split" in frame:
            frame = frame.loc[frame["split"].eq("train")].copy()
            if frame.empty:
                raise AnomalyHealthError("Anomaly fitting requires training engines; test-only input cannot retrain.")
        self.feature_columns = self._feature_columns(frame)
        if not self.feature_columns:
            raise AnomalyHealthError("No residual z-score columns found; run the twin stage first.")

        names = detector_names or list(self.detectors.keys())
        fit_frame, calibration_frame = engine_wise_split(
            frame, validation_fraction=self.config.data.validation_engine_fraction,
            random_state=self.config.data.random_state,
        )
        if fit_frame.empty or calibration_frame.empty:
            raise AnomalyHealthError("Anomaly fitting and threshold calibration need separate engines.")
        summary: Dict[str, Any] = {}
        trained_any = False

        for name in names:
            detector = self.detectors.get(name)
            if detector is None:
                self.logger.warning("Unknown detector '%s' skipped", name)
                continue
            try:
                detector.fit(fit_frame, self.feature_columns)
                trained_any = True
                summary[name] = {"threshold": float(detector.threshold)}
            except Exception as exc:  # noqa: BLE001 - one detector must not block the stage
                self.logger.error("Detector '%s' failed to train: %s", name, exc)
                summary[name] = {"error": str(exc)}

        if not trained_any:
            raise AnomalyHealthError("Every anomaly detector failed to train.")

        self.calibrated_fusion_threshold = self._calibrate_fusion_threshold(calibration_frame)
        # Publish a fitted detector only after its cutoff has been established
        # on separate engines, preserving the previous artifact on failure.
        for name in names:
            if name in summary and "error" not in summary[name]:
                self.detectors[name].save()
        atomic_write_json(
            {"calibrated_fusion_threshold": self.calibrated_fusion_threshold},
            self.config.paths.model_path("anomaly_fusion_meta.json"),
        )
        # A feedback threshold belongs to its fitted score distribution. Reset
        # it on retraining instead of silently applying an older model's cut.
        atomic_write_json(
            {"fusion_threshold": self.calibrated_fusion_threshold,
             "base_threshold": self.calibrated_fusion_threshold,
             "scale": 1.0, "effective_scale": 1.0, "feedback_applied": 0},
            self.config.paths.model_path(self.config.paths.residual_thresholds_json),
        )
        return summary

    def load(self) -> Dict[str, Any]:
        """Reload every detector that has a persisted artefact.

        Returns:
            A summary of which detectors were reloaded.
        """
        summary: Dict[str, Any] = {}
        for name, detector in list(self.detectors.items()):
            try:
                detector.load()
                summary[name] = {"threshold": float(detector.threshold)}
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Detector '%s' could not be reloaded: %s", name, exc)
                del self.detectors[name]
        if not self.detectors:
            raise AnomalyHealthError("No anomaly detectors could be reloaded; train them first.")

        meta = read_json(self.config.paths.model_path("anomaly_fusion_meta.json"), default=None)
        if meta and meta.get("calibrated_fusion_threshold") is not None:
            self.calibrated_fusion_threshold = float(meta["calibrated_fusion_threshold"])
        return summary

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score every row with every available detector.

        Args:
            frame: Residuals frame.

        Returns:
            A frame of identity keys plus one score column per detector.
        """
        result = frame[get_identity_columns()].copy().reset_index(drop=True)
        if "calibration_ready" in frame:
            result["calibration_ready"] = frame["calibration_ready"].to_numpy(bool)
        for name, detector in self.detectors.items():
            column = _DETECTOR_REGISTRY[name]
            try:
                result[column] = detector.score(frame)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Detector '%s' failed during scoring: %s", name, exc)
                # Missing evidence must never masquerade as a normal reading.
                result[column] = np.nan
        return result

    def fuse(
        self, scores: pd.DataFrame, fusion_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Compute the weighted fused anomaly score and boolean flag.

        Args:
            scores: Per-detector score frame from :meth:`score`.
            fusion_threshold: Optional override of the fusion threshold.

        Returns:
            The frame with ``anomaly_score``, ``threshold_used`` and
            ``anomaly_flag`` added.
        """
        result = scores.copy()
        weights = dict(self.config.anomaly.fusion_weights)
        present_weighted = {
            name: (column, weights.get(name, 0.0))
            for name, column in _DETECTOR_REGISTRY.items()
            if column in result.columns and weights.get(name, 0.0) > 0
        }
        fused = np.zeros(len(result), dtype=float)
        available_weight = np.zeros(len(result), dtype=float)
        available_count = np.zeros(len(result), dtype=int)
        for column, weight in present_weighted.values():
            values = result[column].astype(float).to_numpy()
            valid = np.isfinite(values) & (values >= 0) & (values <= 1)
            fused[valid] += weight * values[valid]
            available_weight[valid] += weight
            available_count += valid

        if (available_weight <= 0).any():
            raise AnomalyHealthError("No valid enabled detector score for one or more rows; monitoring unavailable.")
        fused /= available_weight
        result["detectors_available"] = available_count
        expected = sum(w > 0 for w in weights.values())
        result["detector_coverage"] = available_count / max(expected, 1)
        # A fused cutoff is calibrated for a particular detector mixture.
        # Changing that mixture after a failure makes the cutoff invalid.
        if (available_count < expected).any():
            raise AnomalyHealthError("An enabled detector is unavailable; restore it before using the calibrated fusion threshold.")

        result[ANOMALY_SCORE_COLUMN] = np.clip(fused, 0.0, 1.0)

        if fusion_threshold is not None:
            threshold = float(fusion_threshold)
        elif self.calibrated_fusion_threshold is not None:
            threshold = float(self.calibrated_fusion_threshold)
        else:
            threshold = float(self.config.anomaly.fusion_threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise AnomalyHealthError("Fusion threshold must be finite and between zero and one.")
        result[THRESHOLD_USED_COLUMN] = threshold
        result[ANOMALY_FLAG_COLUMN] = result[ANOMALY_SCORE_COLUMN] > threshold
        if "calibration_ready" in result:
            result[ANOMALY_FLAG_COLUMN] &= result["calibration_ready"]
        return result

    # ------------------------------------------------------------------ #
    # Operational-vs-fault gate
    # ------------------------------------------------------------------ #
    def classify_type(self, fused: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
        """Separate genuine faults from suppressed operating-condition changes.

        A flagged row is downgraded to ``'operational'`` when it sits inside
        the regime-transition guard window *and* has not yet persisted for
        ``persistence_min_cycles`` consecutive cycles.  A row that keeps being
        flagged past that many cycles is treated as a fault even if it started
        near a regime change, since a transient sensor response to a throttle
        step does not sustain itself the way a real fault does.

        Args:
            fused: Output of :meth:`fuse`.
            context: Frame carrying ``is_mode_transition`` and identity keys.

        Returns:
            The frame with ``anomaly_type`` and ``anomaly_persistence_cycles``
            added.
        """
        merged = merge_on_keys(
            fused,
            context[get_identity_columns() + [MODE_TRANSITION_COLUMN]]
            if MODE_TRANSITION_COLUMN in context.columns
            else context[get_identity_columns()],
            keys=get_identity_columns(),
            how="left",
        )
        if MODE_TRANSITION_COLUMN not in merged.columns:
            merged[MODE_TRANSITION_COLUMN] = False
        merged[MODE_TRANSITION_COLUMN] = merged[MODE_TRANSITION_COLUMN].fillna(False).astype(bool)

        merged = merged.sort_values(get_identity_columns()).reset_index(drop=True)

        # Consecutive-flag run length per engine, causal (no future leakage).
        flagged = merged[ANOMALY_FLAG_COLUMN].astype(int)
        # Missing cycles break consecutive evidence, even if both observed
        # endpoints were flagged. Otherwise two distant samples can bypass
        # the operational-transition guard.
        gap = merged.groupby("unit", sort=False)["cycle"].diff().ne(1)
        breaks = flagged.eq(0) | gap
        group_id = breaks.groupby(merged["unit"]).cumsum()
        persistence = (
            flagged.groupby([merged["unit"], group_id]).cumsum()
        )
        merged[PERSISTENCE_COLUMN] = np.where(flagged.astype(bool), persistence, 0).astype(int)

        guard = int(self.config.anomaly.persistence_min_cycles)
        anomaly_type = np.where(
            ~merged[ANOMALY_FLAG_COLUMN],
            "none",
            np.where(
                merged[MODE_TRANSITION_COLUMN] & (merged[PERSISTENCE_COLUMN] < guard),
                "operational",
                "fault",
            ),
        )
        merged[ANOMALY_TYPE_COLUMN] = anomaly_type

        counts = pd.Series(anomaly_type).value_counts().to_dict()
        self.logger.info(
            "Operational-vs-fault gate: %s (persistence guard=%d cycles)",
            {str(k): int(v) for k, v in counts.items()},
            guard,
        )
        return merged.drop(columns=[MODE_TRANSITION_COLUMN])

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_scores(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``anomaly_scores.csv`` atomically.

        Args:
            frame: Per-detector scores.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.anomaly_scores_csv)
        verify_identity_columns(frame, context="anomaly_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def save_fusion(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``anomaly_fusion.csv`` atomically.

        Args:
            frame: Fused and classified anomalies.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.anomaly_fusion_csv)
        verify_identity_columns(frame, context="anomaly_fusion output")
        return atomic_write_csv(frame, target, config=self.config)

    # ------------------------------------------------------------------ #
    # Stage orchestration
    # ------------------------------------------------------------------ #
    def run(
        self,
        residuals: Optional[pd.DataFrame] = None,
        context: Optional[pd.DataFrame] = None,
        retrain: bool = True,
        fusion_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the full anomaly-detection stage.

        Args:
            residuals: Optional in-memory residuals; otherwise ``residuals.csv``
                is read from disk.
            context: Optional in-memory context data (for the transition flag);
                otherwise ``context_clusters.csv`` is read from disk.
            retrain: Fit fresh detectors, or reload saved ones.
            fusion_threshold: Optional override of the fusion threshold; when
                omitted, an adapted threshold from the feedback loop is applied
                if one exists.

        Returns:
            A ``(fused_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = residuals
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="anomaly_fusion",
            )
        raw_path = self.config.paths.processed_path(self.config.paths.raw_data_csv)
        if raw_path.is_file():
            raw = read_csv(raw_path, stage="anomaly_fusion")
            extras = [c for c in ("RUL", "split", "subset_id") if c in raw and c not in source]
            if extras:
                source = source.merge(raw[get_identity_columns() + extras], on=get_identity_columns(),
                                      how="left", validate="one_to_one")
        context_source = context
        if context_source is None:
            context_path = self.config.paths.processed_path(
                self.config.paths.context_clusters_csv
            )
            if context_path.is_file():
                context_source = read_csv(
                    context_path, required_columns=get_identity_columns(), stage="anomaly_fusion"
                )
            else:
                context_source = source[get_identity_columns()].copy()

        if retrain:
            train_summary = self.train(source)
        else:
            train_summary = self.load()

        scores = self.score(source)
        output_scores = self.save_scores(scores)

        effective_threshold = fusion_threshold
        if effective_threshold is None:
            adapted = read_json(
                self.config.paths.model_path(self.config.paths.residual_thresholds_json),
                default=None,
            )
            if adapted and "fusion_threshold" in adapted:
                effective_threshold = float(adapted["fusion_threshold"])

        fused = self.fuse(scores, fusion_threshold=effective_threshold)
        classified = self.classify_type(fused, context_source)

        # severity and early_warning_score are part of the anomaly_fusion.csv
        # contract; compute them here so callers get one enriched file rather
        # than needing to chain three separate stage invocations by hand.
        # Recalibrate the severity bands only while retraining; inference
        # reloads the frozen cuts.
        classified = self.severity_classifier.run(classified, calibrate=retrain)
        health_scores_path = self.config.paths.processed_path(
            self.config.paths.health_scores_csv
        )
        health_trend = (
            read_csv(health_scores_path, stage="anomaly_fusion")
            if health_scores_path.is_file()
            else None
        )
        classified = self.early_warning_scorer.run(classified, health_trend)

        output_fusion = self.save_fusion(classified)

        comparator = TwinComparator(self.config)
        try:
            comparator_row_summary = comparator.summarize_rows(source)
            top_sensor_lookup = comparator_row_summary
        except Exception:  # noqa: BLE001
            top_sensor_lookup = None

        type_counts = classified[ANOMALY_TYPE_COLUMN].value_counts().to_dict()
        severity_counts = (
            classified["severity"].value_counts().to_dict()
            if "severity" in classified.columns
            else {}
        )
        report: Dict[str, Any] = {
            "severity_distribution": {str(k): int(v) for k, v in severity_counts.items()},
            "rows": int(len(classified)),
            "engines": int(classified["unit"].nunique()),
            "detector_summary": train_summary,
            "fusion_threshold": float(classified[THRESHOLD_USED_COLUMN].iloc[0])
            if len(classified)
            else float(self.config.anomaly.fusion_threshold),
            "total_flagged": int(classified[ANOMALY_FLAG_COLUMN].sum()),
            "fault_anomalies": int(type_counts.get("fault", 0)),
            "operational_anomalies": int(type_counts.get("operational", 0)),
            "anomaly_type_distribution": {str(k): int(v) for k, v in type_counts.items()},
            "scores_output_path": str(output_scores),
            "fusion_output_path": str(output_fusion),
        }
        atomic_write_json(report, self.config.paths.metrics / "anomaly_detection.json")
        self.logger.info("Anomaly fusion report: %s", {k: v for k, v in report.items() if k != "detector_summary"})
        return classified, report


def main() -> None:
    """Run anomaly detection standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    fusion = AnomalyFusion(config)
    classified, report = fusion.run()
    logger.info(
        "anomaly_fusion.csv written: %d rows, %d fault(s), %d operational",
        report["rows"],
        report["fault_anomalies"],
        report["operational_anomalies"],
    )
    print(classified.head(10).to_string())


if __name__ == "__main__":
    main()
