"""Failure, temporal causality, calibration isolation, and evaluation contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.anomaly_detection.mahalanobis_detector import MahalanobisDetector
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics, boolean_values
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_health import HealthEvaluator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import HealthStateClassifier
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, atomic_write_json
from scripts.Anomaly_Health_Monitoring.evaluate_holdout import load_frames


class MonitoringStrengthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config()
        self.config.mlops.log_to_file = False
        self.config.mlops.enable_mlflow = False
        self.config.twin.commissioning_cycles = 3
        for name, value in vars(self.config.paths).items():
            if isinstance(value, Path):
                setattr(self.config.paths, name, Path(self.temp.name) / name)
        self.frame = pd.DataFrame({"unit": [1] * 6 + [2] * 6,
                                   "cycle": list(range(1, 7)) * 2,
                                   "RUL": [150, 145, 140, 30, 20, 10] * 2,
                                   "residual_sensor_4": [1., 2., 3., 5., 7., 10.] * 2})

    def calibration(self, **updates):
        payload = {"informative_columns": ["residual_sensor_4"], "healthy_reference_deviation": 1.}
        payload.update(updates)
        engine = HealthIndexCalculator(self.config)
        atomic_write_json(payload, engine.calibration_path())
        return engine

    def test_failed_detector_does_not_report_healthy(self):
        fusion = AnomalyFusion(self.config)
        with patch.object(fusion.detectors["mahalanobis"], "score", side_effect=ValueError("broken artifact")):
            with self.assertRaisesRegex(AnomalyHealthError, "No valid"):
                fusion.fuse(fusion.score(self.frame))

    def test_partial_detector_failure_cannot_use_original_mixture_threshold(self):
        self.config.anomaly.fusion_weights = {"mahalanobis": .5, "residual_rule": .5}
        scores = self.frame[["unit", "cycle"]].assign(mahalanobis_score=.9, residual_rule_score=np.nan)
        with self.assertRaisesRegex(AnomalyHealthError, "enabled detector"):
            AnomalyFusion(self.config).fuse(scores)

    def test_disabled_detector_nan_does_not_contaminate_score(self):
        scores = self.frame[["unit", "cycle"]].assign(mahalanobis_score=.8, residual_rule_score=np.nan)
        result = AnomalyFusion(self.config).fuse(scores)
        np.testing.assert_allclose(result.anomaly_score, .8)

    def test_invalid_detector_scores_are_rejected(self):
        for value in (np.nan, np.inf, -.1, 1.1):
            scores = self.frame[["unit", "cycle"]].assign(mahalanobis_score=value)
            with self.assertRaises(AnomalyHealthError):
                AnomalyFusion(self.config).fuse(scores)

    def test_cycle_gaps_restart_fault_persistence(self):
        fused = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 50], "anomaly_flag": True})
        context = fused[["unit", "cycle"]].assign(is_mode_transition=True)
        result = AnomalyFusion(self.config).classify_type(fused, context)
        self.assertEqual(result.anomaly_persistence_cycles.tolist(), [1, 2, 1])
        self.assertTrue(result.anomaly_type.eq("operational").all())

    def test_failed_cutoff_calibration_does_not_publish_detector(self):
        fusion = AnomalyFusion(self.config)
        detector = fusion.detectors["mahalanobis"]
        with patch.object(detector, "fit"), patch.object(detector, "save") as save, \
             patch.object(fusion, "_calibrate_fusion_threshold", side_effect=AnomalyHealthError("bad calibration")):
            with self.assertRaises(AnomalyHealthError):
                fusion.train(self.frame)
            save.assert_not_called()

    def test_empty_healthy_calibration_cannot_use_arbitrary_fallback(self):
        fusion = AnomalyFusion(self.config)
        with self.assertRaisesRegex(AnomalyHealthError, "No healthy calibration"):
            fusion._calibrate_fusion_threshold(self.frame.assign(RUL=0))

    def test_temporal_window_restarts_after_missing_cycle(self):
        detector = MahalanobisDetector(self.config)
        detector.feature_columns = ["residual_sensor_4"]
        frame = pd.DataFrame({"unit": 1, "cycle": [1, 2, 50], "residual_sensor_4": [1., 3., 100.]})
        np.testing.assert_allclose(detector._feature_values(frame)[:, 0], [1., 2., 100.])

    def test_missing_anomaly_residual_is_not_replaced_by_zero(self):
        detector = MahalanobisDetector(self.config)
        detector.feature_columns = ["residual_sensor_4"]
        frame = self.frame.copy()
        frame.loc[4, "residual_sensor_4"] = np.nan
        with self.assertRaisesRegex(AnomalyHealthError, "non-finite"):
            detector._feature_values(frame)

    def test_ewm_features_are_prefix_stable_and_engine_local(self):
        detector = MahalanobisDetector(self.config)
        detector.feature_columns = ["residual_sensor_4"]
        detector.temporal_method = "ewm"
        full = detector._feature_values(self.frame)
        prefix = self.frame[self.frame.cycle <= 4]
        np.testing.assert_allclose(detector._feature_values(prefix), full[self.frame.cycle <= 4])
        self.assertEqual(full[6, 0], 1.)
        shuffled = self.frame.sample(frac=1, random_state=4)
        actual = pd.Series(detector._feature_values(shuffled)[:, 0], index=shuffled.index).sort_index()
        np.testing.assert_allclose(actual, full[:, 0])

    def test_health_inference_cannot_fit_reference_from_degraded_batch(self):
        engine = HealthIndexCalculator(self.config)
        with self.assertRaisesRegex(AnomalyHealthError, "explicitly calibrate"):
            engine.compute(self.frame)
        self.assertFalse(engine.calibration_path().exists())

    def test_missing_selected_health_channel_cannot_recalibrate(self):
        engine = self.calibration()
        with self.assertRaisesRegex(AnomalyHealthError, "Missing calibrated"):
            engine.compute(self.frame.rename(columns={"residual_sensor_4": "residual_sensor_8"}))

    def test_health_calibration_ignores_official_test_rows(self):
        frame = self.frame.assign(RUL=150, split="train")
        test = frame.assign(unit=3, split="test", residual_sensor_4=10000.)
        first = HealthIndexCalculator(self.config).fit_calibration(frame)
        second = HealthIndexCalculator(self.config).fit_calibration(pd.concat([frame, test]))
        self.assertEqual(first, second)
        with self.assertRaisesRegex(AnomalyHealthError, "training"):
            HealthIndexCalculator(self.config).fit_calibration(test)

    def test_health_reference_uses_same_weighted_signal_as_inference(self):
        engine = self.calibration(sensor_weights={"sensor_4": 1., "sensor_8": 0.})
        frame = self.frame.assign(RUL=150, residual_sensor_8=999.)
        fitted = engine.fit_calibration(frame)
        self.assertEqual(fitted["healthy_reference_deviation"], 7.)

    def test_nonfinite_health_sensor_and_invalid_weights_are_rejected(self):
        engine = self.calibration()
        bad = self.frame.copy()
        bad.loc[0, "residual_sensor_4"] = np.nan
        with self.assertRaises(AnomalyHealthError):
            engine.compute(bad)
        with self.assertRaises(AnomalyHealthError):
            engine.compute(self.frame, sensor_weights={"sensor_4": -1.})

    def test_invalid_threshold_update_does_not_mutate_or_persist(self):
        classifier = HealthStateClassifier(self.config)
        before = copy.deepcopy(classifier.thresholds)
        with self.assertRaises(AnomalyHealthError):
            classifier.update_thresholds({"healthy": 10.})
        self.assertEqual(classifier.thresholds, before)
        self.assertFalse(classifier.thresholds_path.exists())

    def test_nan_health_score_cannot_be_classified_as_failure_risk(self):
        with self.assertRaises(AnomalyHealthError):
            HealthStateClassifier(self.config)._classify_scalar(np.nan)

    def test_calibrated_health_denoising_is_causal_and_order_independent(self):
        engine = self.calibration(residual_window=3, ewma_alpha=.1, decay_lambda=.4)
        full = engine.compute(self.frame)
        prefix = engine.compute(self.frame[self.frame.cycle <= 4])
        expected = full.merge(prefix[["unit", "cycle"]], on=["unit", "cycle"])
        np.testing.assert_allclose(prefix.health_index, expected.health_index)
        shuffled = engine.compute(self.frame.sample(frac=1, random_state=9))
        np.testing.assert_allclose(shuffled.health_index, full.health_index)
        changed = self.frame.copy()
        changed.loc[changed.cycle > 4, "residual_sensor_4"] = 999.
        mutated = engine.compute(changed)
        np.testing.assert_allclose(mutated.loc[mutated.cycle <= 4, "health_index"],
                                   full.loc[full.cycle <= 4, "health_index"])

    def test_health_explanation_uses_persisted_denoising_settings(self):
        from app.services.Anomaly_Health_Monitoring.explainability.health_explainer import HealthExplainer
        engine = self.calibration(residual_window=3, ewma_alpha=.1, decay_lambda=.4)
        health = engine.compute(self.frame)
        evidence = HealthExplainer(self.config).explain(self.frame, health)
        self.assertTrue(all(e["health_explanation_status"] == "computed" for e in evidence.values()))
        for e in evidence.values():
            values = e["health_evidence"]
            self.assertEqual(values["residual_window"], 3)
            self.assertEqual(values["decay_lambda"], .4)
            self.assertAlmostEqual(values["current_cycle_term"] + values["history_term"], values["health_index"])

    def scored(self):
        return self.frame.assign(anomaly_score=[.1, .9, .2, .9, .8, .7] * 2,
                                 anomaly_flag=[False, True, False, True, True, True] * 2)

    def test_proxy_metrics_have_correct_confusion_counts_and_censored_denominator(self):
        data = self.scored()
        censored = data.iloc[:3].assign(unit=3)
        result = condition_metrics(pd.concat([data, censored]))
        self.assertEqual((result["true_positive"], result["false_positive"], result["true_negative"]), (6, 3, 6))
        self.assertEqual(result["engine_events"]["engines_reaching_critical"], 2)
        self.assertEqual(result["engine_events"]["detection_rate_of_reaching"], 1.)
        self.assertFalse(result["confirmed_label_metrics"]["available"])

    def test_operational_flags_are_not_counted_as_escalated_faults(self):
        data = self.scored().assign(anomaly_type="operational")
        result = condition_metrics(data)
        self.assertEqual(result["recall"], 0.)
        self.assertEqual(result["false_alarm_rate"], 0.)

    def test_bootstrap_is_deterministic_and_warmup_excluded(self):
        data = self.scored().assign(calibration_ready=lambda d: d.cycle > 3)
        a, b = condition_metrics(data), condition_metrics(data)
        self.assertEqual(a, b)
        self.assertEqual(a["warmup_or_unready_rows_excluded"], 6)

    def test_real_fault_labels_remain_separate_from_rul_proxy(self):
        data = self.scored().assign(fault_label=lambda d: d.cycle == 2)
        result = condition_metrics(data)
        self.assertEqual(result["confirmed_label_metrics"]["true_positive"], 2)
        self.assertEqual(result["true_positive"], 6)

    def test_string_false_is_parsed_correctly(self):
        np.testing.assert_array_equal(boolean_values(pd.Series(["False", "True"])), [False, True])
        with self.assertRaises(ValueError):
            boolean_values(pd.Series(["maybe"]))

    def test_flat_health_cannot_pass_by_being_monotonic(self):
        frame = pd.DataFrame({"unit": np.repeat([1, 2], 30), "cycle": np.tile(np.arange(1, 31), 2),
                              "health_index": 100., "RUL": np.tile(np.arange(29, -1, -1), 2)})
        result = HealthEvaluator(self.config).evaluate(frame)
        self.assertEqual(result["monotonicity_score"], 1.)
        self.assertFalse(result["passed"])
        self.assertFalse(result["acceptance"]["detects_degradation"])

    def test_holdout_loader_accepts_score_metadata_and_rejects_partial_join(self):
        root = Path(self.temp.name)
        labels = self.scored().assign(split="test", subset_id="FD001")
        path, truth = root / "scores.csv", root / "labels.csv"
        labels.to_csv(path, index=False)
        labels[["unit", "cycle", "RUL", "split", "subset_id"]].to_csv(truth, index=False)
        self.assertEqual(len(load_frames(path, truth)), len(labels))
        labels.iloc[:-1][["unit", "cycle", "RUL", "split", "subset_id"]].to_csv(truth, index=False)
        with self.assertRaisesRegex(SystemExit, "Every scored row"):
            load_frames(path, truth)

    def test_holdout_loader_rejects_stale_rul_metadata(self):
        root = Path(self.temp.name)
        labels = self.scored().assign(split="test", subset_id="FD001")
        scores = labels.assign(RUL=999.)
        scores.to_csv(root / "scores.csv", index=False)
        labels[["unit", "cycle", "RUL", "split", "subset_id"]].to_csv(root / "labels.csv", index=False)
        with self.assertRaisesRegex(SystemExit, "metadata disagrees"):
            load_frames(root / "scores.csv", root / "labels.csv")

    def test_holdout_loader_separates_reused_engine_ids_across_splits(self):
        root = Path(self.temp.name)
        train = self.scored().assign(split="train", subset_id="FD001")
        test = train.assign(split="test", RUL=lambda d: d.RUL + 5)
        scores = pd.concat([train, test], ignore_index=True)
        scores.to_csv(root / "scores.csv", index=False)
        scores[["unit", "cycle", "RUL", "split", "subset_id"]].to_csv(root / "labels.csv", index=False)
        merged = load_frames(root / "scores.csv", root / "labels.csv")
        self.assertEqual(len(merged), 24)
        self.assertEqual(merged.split.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
