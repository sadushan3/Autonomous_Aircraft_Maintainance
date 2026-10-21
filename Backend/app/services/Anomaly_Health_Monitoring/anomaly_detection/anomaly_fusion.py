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


ANOMALY_SCORE_COLUMN = "anomaly_score"


ANOMALY_FLAG_COLUMN = "anomaly_flag"


ANOMALY_TYPE_COLUMN = "anomaly_type"


THRESHOLD_USED_COLUMN = "threshold_used"


PERSISTENCE_COLUMN = "anomaly_persistence_cycles"

_DETECTOR_REGISTRY: Dict[str, str] = {
    "isolation_forest": IFOREST_SCORE_COLUMN,
    "mahalanobis": MAHAL_SCORE_COLUMN,
    "lstm_autoencoder": LSTM_SCORE_COLUMN,
    "residual_rule": RULE_SCORE_COLUMN,
    "condition_consensus": CONSENSUS_SCORE_COLUMN,
}


class AnomalyFusion:


    def __init__(self, config: Optional[Config] = None) -> None:


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


        self.calibrated_fusion_threshold: Optional[float] = None


    def _feature_columns(self, frame: pd.DataFrame) -> List[str]:


        if self.config.anomaly.feature_space == "raw":
            return present_residual_columns(frame)
        return present_residual_z_columns(frame)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:


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


        healthy = frame.loc[self._healthy_mask(frame)]
        if healthy.empty:
            raise AnomalyHealthError("No healthy calibration rows are available; a fusion cutoff cannot be fitted.")
        if list(self.detectors) == ["condition_consensus"]:


            return float(self.detectors["condition_consensus"].validate_calibration(frame))
        try:

            healthy_scores = self.score(frame)
            healthy_fused = self.fuse(healthy_scores, fusion_threshold=0.0)
            threshold = percentile_threshold(
                healthy_fused.loc[self._healthy_mask(frame).to_numpy(), ANOMALY_SCORE_COLUMN].to_numpy(),
                self.config.anomaly.fusion_threshold_percentile,
            )
            if not np.isfinite(threshold):
                raise ValueError("non-finite threshold")
        except Exception as exc:
            raise AnomalyHealthError("Fusion threshold calibration failed; the previous artifacts were retained.") from exc

        self.logger.info(
            "Fusion threshold calibrated from %d healthy row(s) at p%.0f: %.4f",
            len(healthy),
            self.config.anomaly.fusion_threshold_percentile,
            threshold,
        )
        return float(threshold)

    def train(self, frame: pd.DataFrame, detector_names: Optional[List[str]] = None, context: Optional[pd.DataFrame] = None) -> Dict[str, Any]:


        if "split" in frame:
            frame = frame.loc[frame["split"].eq("train")].copy()
            if frame.empty:
                raise AnomalyHealthError("Anomaly fitting requires training engines; test-only input cannot retrain.")
        self.feature_columns = self._feature_columns(frame)
        if not self.feature_columns:
            raise AnomalyHealthError("No residual z-score columns found; run the twin stage first.")

        names = detector_names or list(self.detectors.keys())
        from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope
        plan = load_training_split(self.config)
        if plan:
            fit_frame = training_scope(frame, self.config, required=True)
            calibration_frame = training_scope(frame, self.config, ("calibration",), required=True)
            selection_frame = training_scope(frame, self.config, ("selection",), required=True)
        else:
            fit_frame, calibration_frame = engine_wise_split(
                frame, validation_fraction=self.config.data.validation_engine_fraction,
                random_state=self.config.data.random_state,
            )
            selection_frame = calibration_frame
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
                if name == "condition_consensus" and plan:
                    detector.fit(pd.concat([fit_frame, calibration_frame]), self.feature_columns)
                    classify = (lambda values: self.classify_type(values, context)) if context is not None else None
                    detector.calibrate_development_margin(selection_frame, classify=classify)
                else:
                    detector.fit(fit_frame, self.feature_columns)
                trained_any = True
                summary[name] = {"threshold": float(detector.threshold)}
            except Exception as exc:
                self.logger.error("Detector '%s' failed to train: %s", name, exc)
                summary[name] = {"error": str(exc)}

        if not trained_any:
            raise AnomalyHealthError("Every anomaly detector failed to train.")
        failed = [name for name in names if name not in summary or "error" in summary[name]]
        if failed:
            raise AnomalyHealthError(f"Requested anomaly detectors failed to train: {', '.join(failed)}")

        cutoff_frame = selection_frame if list(self.detectors) == ["condition_consensus"] else calibration_frame
        self.calibrated_fusion_threshold = self._calibrate_fusion_threshold(cutoff_frame)


        for name in names:
            if name in summary and "error" not in summary[name]:
                self.detectors[name].save()
        atomic_write_json(
            {"calibrated_fusion_threshold": self.calibrated_fusion_threshold,
             "fusion_weights": {name: self.config.anomaly.fusion_weights[name] for name in names},
             "detectors": summary,
             "feature_space": self.config.anomaly.feature_space,
             "training_plan_id": plan["plan_id"] if plan else None,
             "evaluation_status": "retrospective_condition_proxy"},
            self.config.paths.model_path("anomaly_fusion_meta.json"),
        )


        atomic_write_json(
            {"fusion_threshold": self.calibrated_fusion_threshold,
             "base_threshold": self.calibrated_fusion_threshold,
             "scale": 1.0, "effective_scale": 1.0, "feedback_applied": 0},
            self.config.paths.model_path(self.config.paths.residual_thresholds_json),
        )
        return summary

    def load(self) -> Dict[str, Any]:


        meta = read_json(self.config.paths.model_path("anomaly_fusion_meta.json"), default=None)
        expected_weights = {name: weight for name, weight in self.config.anomaly.fusion_weights.items() if weight > 0}
        if meta and meta.get("fusion_weights") is not None and meta["fusion_weights"] != expected_weights:
            raise AnomalyHealthError("Detector weights differ from the calibrated bundle; recalibrate the fusion threshold.")
        summary: Dict[str, Any] = {}
        for name, detector in list(self.detectors.items()):
            try:
                detector.load()
                summary[name] = {"threshold": float(detector.threshold)}
            except Exception as exc:
                self.logger.warning("Detector '%s' could not be reloaded: %s", name, exc)
                del self.detectors[name]
        if not self.detectors:
            raise AnomalyHealthError("No anomaly detectors could be reloaded; train them first.")

        if meta and meta.get("calibrated_fusion_threshold") is not None:
            self.calibrated_fusion_threshold = float(meta["calibrated_fusion_threshold"])
        return summary


    def score(self, frame: pd.DataFrame) -> pd.DataFrame:


        result = frame[get_identity_columns()].copy().reset_index(drop=True)
        for metadata in ("split", "subset_id"):
            if metadata in frame:
                result[metadata] = frame[metadata].to_numpy()
        if "calibration_ready" in frame:
            result["calibration_ready"] = frame["calibration_ready"].to_numpy(bool)
        for name, detector in self.detectors.items():
            column = _DETECTOR_REGISTRY[name]
            try:
                result[column] = detector.score(frame)
            except Exception as exc:
                self.logger.error("Detector '%s' failed during scoring: %s", name, exc)

                result[column] = np.nan
        return result

    def fuse(
        self, scores: pd.DataFrame, fusion_threshold: Optional[float] = None
    ) -> pd.DataFrame:


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


    def classify_type(self, fused: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:


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


        flagged = merged[ANOMALY_FLAG_COLUMN].astype(int)


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


    def save_scores(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.anomaly_scores_csv)
        verify_identity_columns(frame, context="anomaly_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def save_fusion(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.anomaly_fusion_csv)
        verify_identity_columns(frame, context="anomaly_fusion output")
        return atomic_write_csv(frame, target, config=self.config)


    def run(
        self,
        residuals: Optional[pd.DataFrame] = None,
        context: Optional[pd.DataFrame] = None,
        retrain: bool = True,
        fusion_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


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
            train_summary = self.train(source, context=context_source)
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
        except Exception:
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
