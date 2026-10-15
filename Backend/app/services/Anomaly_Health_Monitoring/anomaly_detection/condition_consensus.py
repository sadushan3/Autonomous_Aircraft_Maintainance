"""Conservative consensus of residual-history RUL-condition classifiers.

The target is the healthy/critical RUL proxy, not a verified fault label.
Inference consumes only past/current residuals and a frozen covariance model.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.mahalanobis_detector import MahalanobisDetector
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, load_model, save_model
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope, record_training_scope

SCORE_COLUMN = "condition_consensus_score"
CHANNELS = [2, 3, 4, 7, 11, 12, 15, 20, 21]


class ConditionConsensusDetector:
    ARTIFACT_NAME = "condition_consensus_detector.pkl"
    CALIBRATED_CUTOFF = True

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.reference = MahalanobisDetector(self.config)
        self.members: list[dict[str, Any]] = []
        self.margin = float(self.config.anomaly.condition_consensus["margin"])
        self.threshold = .5
        self.feature_columns: list[str] = []
        self.provenance: dict[str, Any] = {}

    def _features(self, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        required = ["unit", "cycle"] + [f"residual_sensor_{i}" for i in CHANNELS]
        missing = sorted(set(required) - set(frame))
        if missing:
            raise AnomalyHealthError(f"Missing consensus inputs: {missing}")
        if frame.empty:
            return {"compact": np.empty((0, 10)), "residual": np.empty((0, 64))}, np.array([], dtype=int)
        ordered = frame.reset_index(drop=True).sort_values(["unit", "cycle"])
        positions = ordered.index.to_numpy()
        ordered = ordered.reset_index(drop=True)
        if ordered.duplicated(["unit", "cycle"]).any() or ordered[["unit", "cycle"]].isna().any().any():
            raise AnomalyHealthError("Consensus inputs require unique, non-missing unit/cycle identities.")
        values = ordered[[f"residual_sensor_{i}" for i in CHANNELS]].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise AnomalyHealthError("Consensus inputs contain non-finite residual evidence.")
        if self.reference.covariance_estimator is None:
            raise AnomalyHealthError("Consensus monitoring reference is unavailable.")
        distance = self.reference.raw_distance(ordered) / self.reference._distance_p99
        gap = ordered.groupby("unit", sort=False).cycle.diff().ne(1).cumsum()
        groups = values.groupby([ordered.unit, gap], sort=False)
        means = {w: groups.transform(lambda s: s.rolling(w, min_periods=1).mean()).to_numpy()
                 for w in (5, 20, 40)}
        std = groups.transform(lambda s: s.rolling(20, min_periods=2).std()).fillna(0).to_numpy()
        baseline = np.log1p(distance)[:, None]
        compact = np.column_stack([np.abs(means[w])[:, [2, 4, 6]] for w in (5, 20, 40)] + [baseline])
        residual = np.column_stack([means[5], means[20], means[40], np.abs(means[5]), np.abs(means[20]),
                                    std, means[5] - means[20], baseline])
        return {"compact": compact.astype(np.float32), "residual": residual.astype(np.float32)}, positions

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise AnomalyHealthError("Condition consensus used before fit/load.")
        features, positions = self._features(frame)
        if not len(frame):
            return np.array([], dtype=float)
        ratios = []
        for member in self.members:
            p = np.mean([model.predict_proba(features[member["recipe"]])[:, 1]
                         for model in member["models"]], axis=0)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise AnomalyHealthError("Invalid consensus classifier output; monitoring unavailable.")
            p = np.clip(p, 1e-8, 1 - 1e-8)
            ratios.append(p / (1 - p) / member["scale"])
        # Two complementary histories, each averaged over three seeded models.
        distance = np.median(np.row_stack(ratios), axis=0)
        result = np.empty(len(frame), dtype=float)
        result[positions] = distance / (distance + self.margin)
        return result

    def predict_flags(self, frame: pd.DataFrame) -> np.ndarray:
        return self.score(frame) > self.threshold

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]):
        """Refit the frozen recipe using engine-disjoint healthy calibration."""
        frame = frame.loc[frame.split.eq("train")] if "split" in frame else frame
        plan = load_training_split(self.config)
        if plan:
            frame = training_scope(frame, self.config, ("fit", "calibration"), required=True)
        frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
        if "RUL" not in frame or not np.isfinite(frame.RUL.to_numpy(float)).all():
            raise AnomalyHealthError("Consensus training requires finite independent RUL condition labels.")
        units = frame.groupby("unit", sort=True).size().index
        if len(units) < 20:
            raise AnomalyHealthError("Consensus training needs at least 20 separate engines.")
        if plan:
            fitting_frame = training_scope(frame, self.config, required=True)
            calibration_frame = training_scope(frame, self.config, ("calibration",), required=True)
            fitting, calibration = fitting_frame.unit.unique(), calibration_frame.unit.unique()
            frame = pd.concat([fitting_frame, calibration_frame]).sort_values(["unit", "cycle"]).reset_index(drop=True)
        else:
            strata = frame.groupby("unit").subset_id.first().reindex(units).to_numpy() if "subset_id" in frame else None
            fitting, calibration = train_test_split(units, test_size=.3, stratify=strata, random_state=20261015)
        fit_units = frame.unit.isin(fitting).to_numpy()
        healthy_cal = frame.unit.isin(calibration).to_numpy() & (frame.RUL.to_numpy() >= 125) & (frame.cycle.to_numpy() > 30)
        if not healthy_cal.any():
            raise AnomalyHealthError("No separate healthy calibration evidence for consensus.")
        self.reference.fit(frame.loc[fit_units], feature_columns)
        record_training_scope("consensus_classifiers", frame.loc[fit_units], self.config)
        record_training_scope("consensus_healthy_scale", frame.loc[frame.unit.isin(calibration)], self.config, ("calibration",))
        self.feature_columns = list(self.reference.feature_columns)
        features, _ = self._features(frame)
        labels = frame.RUL.to_numpy() <= 30
        labeled = (frame.RUL.to_numpy() >= 125) | labels
        rows = np.flatnonzero(fit_units & labeled & (frame.cycle.to_numpy() > 30))
        if len(np.unique(labels[rows])) != 2:
            raise AnomalyHealthError("Consensus fitting requires both healthy and critical condition proxies.")
        rows = np.sort(np.random.default_rng(42).choice(rows, min(len(rows), 45000), replace=False))
        parameters = self.config.anomaly.condition_consensus
        members = []
        for recipe in ("compact", "residual"):
            models = []
            for seed in parameters["seeds"]:
                model = ExtraTreesClassifier(n_estimators=int(parameters["n_estimators"]),
                    max_depth=int(parameters["max_depth"]), min_samples_leaf=int(parameters["min_samples_leaf"]),
                    class_weight="balanced", n_jobs=4, random_state=int(seed))
                model.fit(features[recipe][rows], labels[rows])
                models.append(model)
            p = np.clip(np.mean([m.predict_proba(features[recipe][healthy_cal])[:, 1] for m in models], axis=0), 1e-8, 1 - 1e-8)
            members.append({"recipe": recipe, "models": models, "scale": max(float(np.max(p / (1 - p))), 1e-10)})
        self.members = members
        self.provenance = {"label_basis": "RUL condition proxies", "fit_engines": len(fitting),
                           "healthy_calibration_engines": len(calibration), "calibration": "maximum healthy odds with fixed safety margin"}
        self.provenance.update(training_plan_id=plan["plan_id"] if plan else None,
                               fit_unit_ids=sorted(map(int, fitting)), calibration_unit_ids=sorted(map(int, calibration)),
                               benchmark_reused=True, evaluation_status="retrospective_condition_proxy")
        return self

    def validate_calibration(self, frame: pd.DataFrame) -> float:
        """Preserve the conservative cutoff; do not replace it by a low percentile.

        If an artifact is already deployed, retraining must improve condition
        performance on the outer held-out engines before it can replace it.
        """
        if "RUL" not in frame:
            raise AnomalyHealthError("Consensus validation needs RUL condition proxies.")
        if load_training_split(self.config):
            frame = training_scope(frame, self.config, ("selection",), required=True)
        candidate = frame.copy().reset_index(drop=True)
        candidate["calibration_ready"] = candidate.cycle > 30
        candidate["anomaly_score"] = self.score(frame)
        candidate["anomaly_flag"] = candidate.anomaly_score > self.threshold
        after = condition_metrics(candidate, bootstrap=False)
        if not after["healthy_rows"] or not after["critical_rows"]:
            raise AnomalyHealthError("Consensus validation requires healthy and critical engines.")
        previous_path = self.config.paths.model_path(self.ARTIFACT_NAME)
        if previous_path.is_file():
            previous = ConditionConsensusDetector(self.config).load()
            baseline = candidate.copy()
            baseline["anomaly_score"] = previous.score(frame)
            baseline["anomaly_flag"] = baseline.anomaly_score > previous.threshold
            before = condition_metrics(baseline, bootstrap=False)
            bound = float(self.config.anomaly.condition_consensus["max_false_alarm_increase"])
            improved = (after["recall"] > before["recall"] and after["f1"] > before["f1"]
                        and after["false_alarm_rate"] <= before["false_alarm_rate"] + bound
                        and all(after[k] >= before[k] for k in ("auc", "average_precision")))
            if "subset_id" in frame:
                for subset in sorted(frame.subset_id.unique()):
                    rows = frame.subset_id.eq(subset).to_numpy()
                    a = condition_metrics(candidate.loc[rows], bootstrap=False)
                    b = condition_metrics(baseline.loc[rows], bootstrap=False)
                    if a["healthy_rows"] and a["critical_rows"]:
                        improved &= (a["recall"] >= b["recall"] and a["f1"] >= b["f1"]
                                     and a["false_alarm_rate"] <= b["false_alarm_rate"] + bound)
            if not improved:
                raise AnomalyHealthError("Refitted consensus did not improve held-out performance; previous artifacts retained.")
        else:
            baseline = candidate.copy()
            baseline["anomaly_score"] = self.reference.score(frame)
            baseline["anomaly_flag"] = baseline.anomaly_score > self.reference.threshold
            before = condition_metrics(baseline, bootstrap=False)
            bound = float(self.config.anomaly.condition_consensus["max_false_alarm_increase"])
            if (after["f1"] <= before["f1"] or after["recall"] <= before["recall"]
                    or after["auc"] < before["auc"] or after["average_precision"] < before["average_precision"]
                    or after["false_alarm_rate"] > min(.005, before["false_alarm_rate"] + bound)):
                raise AnomalyHealthError("Consensus did not improve held-out covariance predictions; no artifact published.")
        return self.threshold

    def save(self):
        if not self.members:
            raise AnomalyHealthError("No fitted consensus to save.")
        reference = {"covariance_estimator": self.reference.covariance_estimator,
                     "feature_columns": self.reference.feature_columns, "distance_p99": self.reference._distance_p99,
                     "temporal_window": self.reference.temporal_window, "temporal_method": self.reference.temporal_method}
        return save_model({"schema_version": 1, "members": self.members, "margin": self.margin,
                           "threshold": self.threshold, "reference": reference, "provenance": self.provenance},
                          self.config.paths.model_path(self.ARTIFACT_NAME), config=self.config)

    def load(self):
        bundle = load_model(self.config.paths.model_path(self.ARTIFACT_NAME))
        if bundle.get("schema_version") != 1:
            raise AnomalyHealthError("Unsupported consensus artifact schema.")
        margin, threshold = float(bundle["margin"]), float(bundle["threshold"])
        members, ref = bundle["members"], bundle["reference"]
        if not np.isfinite([margin, threshold, ref["distance_p99"]]).all() or margin <= 0 or ref["distance_p99"] <= 0 or not 0 < threshold < 1:
            raise AnomalyHealthError("Invalid consensus calibration.")
        if len(members) != 2 or {m["recipe"] for m in members} != {"compact", "residual"} or any(not m["models"]
                or not np.isfinite(m["scale"]) or m["scale"] <= 0 for m in members):
            raise AnomalyHealthError("Invalid consensus members.")
        if not ref["feature_columns"] or int(ref["temporal_window"]) < 1 or ref["temporal_method"] not in ("rolling", "ewm"):
            raise AnomalyHealthError("Invalid consensus monitoring reference.")
        self.members, self.margin, self.threshold = members, margin, threshold
        self.reference.covariance_estimator = ref["covariance_estimator"]
        self.reference.feature_columns = list(ref["feature_columns"])
        self.reference._distance_p99 = float(ref["distance_p99"])
        self.reference.temporal_window = int(ref["temporal_window"])
        self.reference.temporal_method = ref["temporal_method"]
        self.feature_columns = list(self.reference.feature_columns)
        self.provenance = bundle.get("provenance", {})
        plan = load_training_split(self.config)
        saved_plan = self.provenance.get("training_plan_id")
        if (saved_plan or plan) and (plan is None or saved_plan != plan["plan_id"]):
            raise AnomalyHealthError("Consensus artifact does not belong to the active training bundle.")
        return self
