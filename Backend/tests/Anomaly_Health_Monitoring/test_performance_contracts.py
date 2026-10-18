import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import RandomForestTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin, STARTUP_PROFILE_ARTIFACT
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import ResidualCalculator
from app.services.Anomaly_Health_Monitoring.anomaly_detection.mahalanobis_detector import MahalanobisDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator


class ZeroPopulation:
    def predict(self, x):
        return np.zeros((len(x), 1))


class EchoStartupProfile:
    def predict(self, x):

        return x[:, 38:39]


class PerformanceContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Config()

        self.config.anomaly.fusion_weights = {"mahalanobis": 1.0}
        for key, value in vars(self.config.paths).items():
            if isinstance(value, Path):
                setattr(self.config.paths, key, Path(self.directory.name) / key)
        self.config.mlops.log_to_file = False
        self.config.mlops.enable_mlflow = False
        self.config.twin.commissioning_cycles = 3
        self.frame = pd.DataFrame({"unit": [1]*6 + [2]*6, "cycle": list(range(1, 7))*2,
                                   "sensor_1": [1, 2, 3, 40, 50, 60] + [11, 12, 13, 140, 150, 160],
                                   "op_setting_1": 0., "op_setting_2": 0., "op_setting_3": 0.,
                                   "operating_mode": 0, "cycle_norm": 99., "RUL": 150})
        self.twin = RandomForestTwin(self.config)
        self.twin.target_names = ["sensor_1"]

    def calibrate(self, frame):
        return self.twin.calibrate_predictions(frame, np.zeros((len(frame), 1)))

    def test_commissioning_excludes_current_and_freezes(self):
        result = self.calibrate(self.frame)[:, 0]
        np.testing.assert_allclose(result[:6], [0, 1, 1.5, 2, 2, 2])
        np.testing.assert_allclose(result[6:], [0, 11, 11.5, 12, 12, 12])
        changed = self.frame.copy()
        changed.loc[2, "sensor_1"] = 999
        self.assertEqual(self.calibrate(changed)[2, 0], result[2])

    def test_prefix_predictions_do_not_change_when_future_is_appended(self):
        prefix = self.frame[self.frame.cycle <= 4]
        full = self.calibrate(self.frame)[:, 0]
        np.testing.assert_allclose(self.calibrate(prefix)[:, 0], full[self.frame.cycle <= 4])

    def test_predictions_are_independent_of_row_order(self):
        shuffled = self.frame.sample(frac=1, random_state=9)
        p = pd.Series(self.calibrate(shuffled)[:, 0], index=shuffled.index).sort_index()
        np.testing.assert_allclose(p, self.calibrate(self.frame)[:, 0])

    def test_late_start_uses_population_expectation(self):
        late = self.frame[self.frame.cycle > 3]
        np.testing.assert_array_equal(self.calibrate(late), np.zeros((len(late), 1)))

    def test_age_ignores_future_length_and_provided_normalized_age(self):
        prefix = self.frame[self.frame.cycle <= 4].copy()
        x, _ = self.twin.build_design_matrix(prefix)
        prefix["cycle_norm"] = -9999
        x2, _ = self.twin.build_design_matrix(prefix)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_allclose(x[:, 4], prefix.cycle / 300)

    def detector(self):
        detector = MahalanobisDetector(self.config)
        detector.feature_columns = ["sensor_1"]
        return detector

    def test_temporal_features_are_past_only_and_engine_local(self):
        detector = self.detector()
        detector.temporal_window = 3
        values = detector._feature_values(self.frame)[:, 0]
        np.testing.assert_allclose(values[:4], [1, 1.5, 2, 15])
        self.assertEqual(values[6], 11)
        prefix = self.frame[self.frame.cycle <= 4]
        np.testing.assert_allclose(detector._feature_values(prefix)[:, 0], values[self.frame.cycle <= 4])
        shuffled = self.frame.sample(frac=1, random_state=8)
        aligned = pd.Series(detector._feature_values(shuffled)[:, 0], index=shuffled.index).sort_index()
        np.testing.assert_allclose(aligned, values)

    def test_rational_tail_preserves_fault_ranking(self):
        detector = self.detector()
        detector._distance_p99 = 10
        scores = detector._normalize(np.array([0., 10., 100., 1000.]))
        self.assertEqual(scores[1], .5)
        self.assertTrue((np.diff(scores) > 0).all())

    def test_model_roundtrip_preserves_feature_and_score_semantics(self):
        detector = self.detector()
        detector.fit(self.frame, ["sensor_1"])
        before = detector.score(self.frame)
        detector.save()
        restored = self.detector().load()
        np.testing.assert_allclose(restored.score(self.frame), before)
        self.assertEqual(restored.temporal_window, 5)
        self.assertEqual(restored.score_normalization, "rational")

    def test_legacy_model_retains_unsmoothed_empirical_scoring(self):
        detector = self.detector().fit(self.frame, ["sensor_1"])
        detector.save()
        path = self.config.paths.models / detector.ARTIFACT_NAME
        bundle = joblib.load(path)
        bundle.pop("temporal_window")
        bundle.pop("score_normalization")
        joblib.dump(bundle, path)
        restored = self.detector().load()
        self.assertEqual(restored.temporal_window, 1)
        self.assertEqual(restored.score_normalization, "empirical")

    def test_residuals_are_not_centered_a_second_time(self):
        predictions = self.frame[["unit", "cycle"]].copy()
        predictions["expected_sensor_1"] = self.calibrate(self.frame)[:, 0]
        calculator = ResidualCalculator(self.config)
        with patch.object(calculator, "_zscore", side_effect=lambda r, c, **kw: r[c]):
            residuals = calculator.compute(self.frame, predictions)
        np.testing.assert_allclose(residuals.residual_sensor_1,
                                   self.frame.sensor_1.to_numpy() - predictions.expected_sensor_1.to_numpy())
        self.assertFalse(residuals.loc[residuals.cycle <= 3, "calibration_ready"].any())
        self.assertTrue(residuals.loc[residuals.cycle > 3, "calibration_ready"].all())

    def test_warmup_cannot_raise_a_fault(self):
        fusion = AnomalyFusion(self.config)
        scores = self.frame[["unit", "cycle"]].copy()
        scores["mahalanobis_score"] = .99
        scores["calibration_ready"] = scores.cycle > 3
        fused = fusion.fuse(scores, fusion_threshold=.5)
        self.assertFalse(fused.loc[fused.cycle <= 3, "anomaly_flag"].any())
        self.assertTrue(fused.loc[fused.cycle > 3, "anomaly_flag"].all())
        self.assertEqual(list(fusion.detectors), ["mahalanobis"])

    def test_anomaly_fit_and_threshold_use_disjoint_engines(self):
        fusion = AnomalyFusion(self.config)
        fitted = []
        calibrated = []
        detector = fusion.detectors["mahalanobis"]
        frame = self.frame.rename(columns={"sensor_1": "residual_sensor_1"})
        with patch.object(detector, "fit", side_effect=lambda f, c: fitted.extend(f.unit.unique())), \
             patch.object(detector, "save"), \
             patch.object(fusion, "_calibrate_fusion_threshold", side_effect=lambda f: calibrated.extend(f.unit.unique()) or .5):
            fusion.train(frame)
        self.assertTrue(fitted and calibrated)
        self.assertFalse(set(fitted) & set(calibrated))

    def test_health_uses_frozen_weights_and_honors_explicit_override(self):
        frame = self.frame[["unit", "cycle"]].copy()
        frame["residual_sensor_1"] = .1
        frame["residual_sensor_2"] = 10.
        calibration = {"informative_columns": ["residual_sensor_1", "residual_sensor_2"],
                       "healthy_reference_deviation": .1,
                       "sensor_weights": {"sensor_1": 1., "sensor_2": 0.}}
        engine = HealthIndexCalculator(self.config)
        with patch.object(engine, "load_calibration", return_value=calibration):
            calibrated = engine.compute(frame)
            override = engine.compute(frame, sensor_weights={"sensor_1": 0., "sensor_2": 1.})
        np.testing.assert_allclose(calibrated.health_index_raw, 100.)
        self.assertTrue((override.health_index_raw < 1).all())

    def test_reference_recalibration_preserves_health_weight_policy(self):
        engine = HealthIndexCalculator(self.config)
        frame = self.frame.rename(columns={"sensor_1": "residual_sensor_1"})
        previous = {"sensor_weights": {"sensor_1": .5}, "weight_selection": "engine holdout"}
        with patch("app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator.read_json", return_value=previous):
            calibrated = engine.fit_calibration(frame)
        self.assertEqual(calibrated["sensor_weights"], previous["sensor_weights"])

    def test_health_does_not_report_perfect_health_when_selected_channels_are_missing(self):
        frame = pd.DataFrame({"residual_sensor_1": [2., 4.]})
        values = HealthIndexCalculator(self.config)._deviation(
            frame, ["residual_sensor_1"], {"sensor_1": 0.})
        np.testing.assert_allclose(values, [2., 4.])

    def profile_ensemble(self):
        ensemble = EnsembleTwin(self.config, twins=[self.twin])
        ensemble.target_names = ["sensor_1"]
        ensemble.weights = {"random_forest": 1.}
        ensemble.startup_profile = {"population": ZeroPopulation(), "correction": EchoStartupProfile(),
            "version": 1,
            "target_names": ["sensor_1"], "commissioning_cycles": 3,
            "age_reference_cycles": 400, "mode_prior": 5, "n_modes": 6,
            "subsets": ["FD001", "FD002", "FD003", "FD004"]}
        return ensemble

    def test_startup_profile_never_uses_current_or_late_sensor_readings(self):
        frame = self.frame.assign(subset_id="FD001")
        ensemble = self.profile_ensemble()
        prediction, ready = ensemble._profile_prediction(frame, np.zeros((len(frame), 1)))
        np.testing.assert_allclose(prediction[:, 0], [0, 0, 0, 2, 2, 2, 0, 0, 0, 12, 12, 12])
        np.testing.assert_array_equal(ready, frame.cycle > 3)
        changed = frame.copy()
        changed.loc[changed.cycle > 3, "sensor_1"] = 99999
        np.testing.assert_array_equal(ensemble._profile_prediction(changed, np.zeros((len(frame), 1)))[0], prediction)
        prefix = frame[frame.cycle <= 4]
        np.testing.assert_array_equal(ensemble._profile_prediction(prefix, np.zeros((len(prefix), 1)))[0],
                                      prediction[frame.cycle <= 4])

    def test_startup_profile_is_engine_local_and_row_order_independent(self):
        frame = self.frame.assign(subset_id="FD001")
        ensemble = self.profile_ensemble()
        original, _ = ensemble._profile_prediction(frame, np.zeros((len(frame), 1)))
        shuffled = frame.sample(frac=1, random_state=30)
        result, _ = ensemble._profile_prediction(shuffled, np.zeros((len(frame), 1)))
        np.testing.assert_array_equal(pd.Series(result[:, 0], index=shuffled.index).sort_index(), original[:, 0])
        alone = frame[frame.unit == 2]
        np.testing.assert_array_equal(ensemble._profile_prediction(alone, np.zeros((len(alone), 1)))[0], original[frame.unit == 2])

    def test_startup_profile_requires_complete_history_and_known_context(self):
        ensemble = self.profile_ensemble()
        frame = self.frame.assign(subset_id="FD001")
        for incomplete in (frame[frame.cycle != 2], frame[frame.cycle > 3],
                           frame.drop(columns="subset_id"), frame.assign(subset_id="unknown")):
            reference = np.full((len(incomplete), 1), 7.)
            predicted, ready = ensemble._profile_prediction(incomplete, reference)
            np.testing.assert_array_equal(predicted, reference)
            self.assertFalse(ready.any())

    def test_enhanced_twin_preserves_monitoring_reference_and_spread(self):
        ensemble = self.profile_ensemble()
        frame = self.frame.assign(subset_id="FD001")
        reference = frame[["unit", "cycle"]].copy()
        reference["expected_sensor_1"] = 0.
        with patch.object(self.twin, "predict", return_value=reference):
            predictions = ensemble.predict(frame)
        np.testing.assert_array_equal(predictions.reference_expected_sensor_1, np.zeros(len(frame)))
        np.testing.assert_array_equal(predictions.twin_spread_sensor_1, np.zeros(len(frame)))
        calculator = ResidualCalculator(self.config)
        with patch.object(calculator, "_zscore", side_effect=lambda r, c, **kw: r[c]):
            result = calculator.compute(frame, predictions)
        np.testing.assert_array_equal(result.residual_sensor_1, frame.sensor_1)
        np.testing.assert_allclose(result.prediction_residual_sensor_1,
                                   frame.sensor_1 - predictions.expected_sensor_1)

    def test_partial_monitoring_reference_is_rejected(self):
        frame = self.frame.assign(sensor_2=1.)
        predictions = frame[["unit", "cycle"]].assign(expected_sensor_1=0., expected_sensor_2=0.,
                                                        reference_expected_sensor_1=0.)
        with self.assertRaisesRegex(Exception, "every predicted sensor"):
            ResidualCalculator(self.config).compute(frame, predictions)

    def test_startup_profile_disabled_retains_original_predictions(self):
        ensemble = self.profile_ensemble()
        self.config.twin.startup_profile_enabled = False
        frame = self.frame.assign(subset_id="FD001")
        reference = np.full((len(frame), 1), 7.)
        predicted, ready = ensemble._profile_prediction(frame, reference)
        np.testing.assert_array_equal(predicted, reference)
        self.assertFalse(ready.any())

    def test_startup_profile_save_load_and_unsupported_version_fallback(self):
        original = self.profile_ensemble()
        original._persist_startup_profile()
        restored = EnsembleTwin(self.config, twins=[self.twin])
        with patch.object(self.twin, "load", return_value=self.twin):
            restored.load()
        frame = self.frame.assign(subset_id="FD001")
        reference = np.zeros((len(frame), 1))
        np.testing.assert_array_equal(original._profile_prediction(frame, reference)[0],
                                      restored._profile_prediction(frame, reference)[0])
        original.startup_profile["version"] = 999
        original._persist_startup_profile()
        with patch.object(self.twin, "load", return_value=self.twin):
            restored.load()
        self.assertIsNone(restored.startup_profile)

    def test_profile_inference_failure_does_not_break_monitoring(self):
        ensemble = self.profile_ensemble()
        frame = self.frame.assign(subset_id="FD001")
        reference = frame[["unit", "cycle"]].assign(expected_sensor_1=7.)
        with patch.object(self.twin, "predict", return_value=reference), \
             patch.object(ensemble, "_profile_prediction", side_effect=ValueError("invalid profile")):
            predictions = ensemble.predict(frame)
        np.testing.assert_array_equal(predictions.expected_sensor_1, reference.expected_sensor_1)
        np.testing.assert_array_equal(predictions.reference_expected_sensor_1, reference.expected_sensor_1)
        self.assertFalse(predictions.startup_profile_ready.any())

    def test_profile_training_requires_r2_and_rmse_improvement(self):
        self.config.twin.commissioning_cycles = 30
        ensemble = self.profile_ensemble()
        frame = pd.DataFrame({"unit": np.repeat(np.arange(6), 60), "cycle": np.tile(np.arange(1, 61), 6),
            "sensor_1": np.random.default_rng(42).normal(size=360), "subset_id": "FD001", "RUL": 200,
            "op_setting_1": 0., "op_setting_2": 0., "op_setting_3": 0., "operating_mode": 0})
        reference = frame[["unit", "cycle"]].assign(expected_sensor_1=0.)
        scores = [{"r2": .3, "rmse": .5}, {"r2": .4, "rmse": .6}]
        with patch.object(ensemble, "predict", return_value=reference), \
             patch.object(BaseTwin, "_score", side_effect=scores):
            ensemble._fit_startup_profile(frame)
        self.assertIsNone(ensemble.startup_profile)

    def test_rejected_retrain_cannot_leave_stale_profile_active(self):
        ensemble = self.profile_ensemble()
        ensemble._persist_startup_profile()
        ensemble.startup_profile = None
        ensemble._persist_startup_profile()
        self.assertIsNone(joblib.load(self.config.paths.models / STARTUP_PROFILE_ARTIFACT))

    def profile_training_fixture(self):
        self.config.twin.commissioning_cycles = 30
        ensemble = self.profile_ensemble()
        frame = pd.DataFrame({"unit": np.repeat(np.arange(6), 60), "cycle": np.tile(np.arange(1, 61), 6),
            "sensor_1": np.random.default_rng(42).normal(size=360), "subset_id": "FD001", "RUL": 200,
            "op_setting_1": 0., "op_setting_2": 0., "op_setting_3": 0., "operating_mode": 0})
        reference = frame[["unit", "cycle"]].assign(expected_sensor_1=0.)
        return ensemble, frame, reference

    def test_kernel_training_requires_joint_gain_over_validated_ridge(self):
        ensemble, frame, reference = self.profile_training_fixture()
        scores = [{"r2": .3, "rmse": .6}, {"r2": .5, "rmse": .4}, {"r2": .6, "rmse": .5}]
        with patch.object(ensemble, "predict", return_value=reference), \
             patch.object(BaseTwin, "_score", side_effect=scores):
            ensemble._fit_startup_profile(frame)
        self.assertEqual(ensemble.startup_profile["correction_recipe"], "ridge_alpha100")
        self.assertEqual(ensemble.startup_profile["validation_metrics"], scores[1])

    def test_kernel_training_selects_joint_improvement(self):
        ensemble, frame, reference = self.profile_training_fixture()
        scores = [{"r2": .3, "rmse": .6}, {"r2": .5, "rmse": .4}, {"r2": .6, "rmse": .39}]
        with patch.object(ensemble, "predict", return_value=reference), \
             patch.object(BaseTwin, "_score", side_effect=scores):
            ensemble._fit_startup_profile(frame)
        self.assertEqual(ensemble.startup_profile["correction_recipe"], "kernel_rbf_0.001_alpha1")
        self.assertEqual(ensemble.startup_profile["validation_metrics"], scores[2])
        self.assertEqual(ensemble.startup_profile["ridge_validation_metrics"], scores[1])

    def test_optional_kernel_failure_retains_validated_ridge(self):
        ensemble, frame, reference = self.profile_training_fixture()
        scores = [{"r2": .3, "rmse": .6}, {"r2": .5, "rmse": .4}]
        with patch.object(ensemble, "predict", return_value=reference), \
             patch.object(BaseTwin, "_score", side_effect=scores), \
             patch("sklearn.kernel_ridge.KernelRidge.fit", side_effect=ValueError("optional fit failed")):
            ensemble._fit_startup_profile(frame)
        self.assertEqual(ensemble.startup_profile["correction_recipe"], "ridge_alpha100")

    def test_profile_correction_batches_are_bounded_and_preserve_order(self):
        sizes = []

        class RecordingModel:
            def predict(self, features):
                sizes.append(len(features))
                return features[:, :1]

        x = np.arange(2051).reshape(-1, 1)
        predicted = EnsembleTwin._predict_profile_correction(RecordingModel(), x)
        self.assertEqual(sizes, [1024, 1024, 3])
        np.testing.assert_array_equal(predicted, x)


if __name__ == "__main__":
    unittest.main()
