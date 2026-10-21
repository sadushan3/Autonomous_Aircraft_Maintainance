import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.condition_consensus import ConditionConsensusDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError


class ConditionConsensusContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config()
        self.cfg.paths.models = Path(self.temp.name)
        self.cfg.mlops.log_to_file = False
        self.cfg.mlops.enable_mlflow = False
        self.cfg.anomaly.fusion_weights = {"condition_consensus": 1.0}
        rng = np.random.default_rng(123)
        self.frame = pd.DataFrame({"unit": np.repeat(np.arange(4), 80), "cycle": np.tile(np.arange(1, 81), 4)})
        self.frame["RUL"] = np.where(self.frame.cycle <= 50, 200, 20)
        for i in range(1, 22):
            self.frame[f"residual_sensor_{i}"] = rng.normal(0, .1, len(self.frame)) + np.maximum(self.frame.cycle - 50, 0) * .05
        self.model = ConditionConsensusDetector(self.cfg)
        columns = [f"residual_sensor_{i}" for i in range(1, 22)]
        self.model.reference.fit(self.frame, columns)
        self.model.feature_columns = columns
        features, _ = self.model._features(self.frame)
        for recipe in ("compact", "residual"):
            estimator = ExtraTreesClassifier(n_estimators=8, max_depth=5, random_state=17).fit(features[recipe], self.frame.RUL <= 30)
            self.model.members.append({"recipe": recipe, "models": [estimator], "scale": 1.})

    def test_inference_never_uses_rul_or_future_trajectory_length(self):
        expected = self.model.score(self.frame)
        changed = self.frame.assign(RUL=-999, cycle_norm=999, subset_id="unrelated")
        np.testing.assert_array_equal(self.model.score(changed), expected)
        prefix = self.frame[self.frame.cycle <= 60]
        np.testing.assert_array_equal(self.model.score(prefix), expected[self.frame.cycle <= 60])

    def test_engine_history_and_row_order_are_independent(self):
        expected = self.model.score(self.frame)
        one = self.frame[self.frame.unit == 2]
        np.testing.assert_array_equal(self.model.score(one), expected[self.frame.unit == 2])
        shuffled = self.frame.sample(frac=1, random_state=4)
        actual = pd.Series(self.model.score(shuffled), index=shuffled.index).sort_index().to_numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_gap_restarts_history(self):
        frame = self.frame[self.frame.unit == 0].copy()
        with_gap = frame.loc[(frame.cycle <= 10) | (frame.cycle >= 60)]
        late = frame.loc[frame.cycle >= 60]
        actual, _ = self.model._features(with_gap)
        expected, _ = self.model._features(late)
        for recipe in actual:
            np.testing.assert_allclose(actual[recipe][-len(late):], expected[recipe])

    def test_missing_nonfinite_and_duplicate_inputs_fail_closed(self):
        bad = [self.frame.drop(columns="residual_sensor_4"), pd.concat([self.frame, self.frame.iloc[:1]])]
        corrupt = self.frame.copy()
        corrupt.loc[0, "residual_sensor_4"] = np.nan
        bad.append(corrupt)
        for frame in bad:
            with self.assertRaises(AnomalyHealthError):
                self.model.score(frame)

    def test_persistence_freezes_margin_and_reference(self):
        expected = self.model.score(self.frame)
        self.model.save()
        self.cfg.anomaly.condition_consensus["margin"] = .01
        loaded = ConditionConsensusDetector(self.cfg).load()
        self.assertEqual(loaded.margin, self.model.margin)
        self.assertEqual(loaded.margin, 1.)
        np.testing.assert_array_equal(loaded.score(self.frame), expected)

    def test_covariance_cutoff_and_normalization_survive_reload(self):
        cutoff = self.model.reference.threshold
        expected = self.model.reference.score(self.frame)
        self.model.save()
        loaded = ConditionConsensusDetector(self.cfg).load()
        self.assertEqual(loaded.reference.threshold, cutoff)
        self.assertEqual(loaded.reference.score_normalization, self.model.reference.score_normalization)
        np.testing.assert_allclose(loaded.reference.score(self.frame), expected, rtol=0, atol=1e-14)

    def test_consensus_profile_requires_consensus_artifact(self):
        fusion = AnomalyFusion(self.cfg)
        self.assertEqual(list(fusion.detectors), ["condition_consensus"])
        with self.assertRaises(AnomalyHealthError):
            fusion.load()

    def test_failed_consensus_cannot_become_a_normal_score(self):
        fusion = AnomalyFusion(self.cfg)
        with patch.object(fusion.detectors["condition_consensus"], "score", side_effect=ValueError("broken")):
            with self.assertRaises(AnomalyHealthError):
                fusion.fuse(fusion.score(self.frame))

    def test_fusion_preserves_calibrated_safety_margin(self):
        fusion = AnomalyFusion(self.cfg)
        with patch.object(fusion.detectors["condition_consensus"], "validate_calibration", return_value=.5) as validation:
            self.assertEqual(fusion._calibrate_fusion_threshold(self.frame), .5)
            validation.assert_called_once()

    def test_commissioning_rows_never_escalate(self):
        self.model.save()
        fusion = AnomalyFusion(self.cfg)
        fusion.load()
        frame = self.frame.assign(calibration_ready=self.frame.cycle > 30)
        result = fusion.fuse(fusion.score(frame), fusion_threshold=0.)
        self.assertFalse(result.loc[frame.cycle <= 30, "anomaly_flag"].any())

    def test_unchanged_retrain_cannot_replace_previous_artifact(self):
        self.model.save()
        path = self.cfg.paths.models / self.model.ARTIFACT_NAME
        previous = path.read_bytes()
        with self.assertRaisesRegex(AnomalyHealthError, "previous artifacts retained"):
            self.model.validate_calibration(self.frame)
        self.assertEqual(previous, path.read_bytes())

    def test_invalid_artifact_is_rejected(self):
        self.model.save()
        path = self.cfg.paths.models / self.model.ARTIFACT_NAME
        bundle = joblib.load(path)
        bundle["members"][0]["scale"] = np.nan
        joblib.dump(bundle, path)
        with self.assertRaisesRegex(AnomalyHealthError, "Invalid consensus"):
            ConditionConsensusDetector(self.cfg).load()

    def test_official_test_rows_cannot_train_consensus(self):
        with patch.object(self.model.reference, "fit") as fit:
            with self.assertRaises(AnomalyHealthError):
                self.model.fit(self.frame.assign(split="test"), self.model.feature_columns)
            fit.assert_not_called()

    def test_first_training_cannot_publish_an_all_normal_classifier(self):
        with patch.object(self.model, "score", return_value=np.zeros(len(self.frame))):
            with self.assertRaisesRegex(AnomalyHealthError, "no artifact published"):
                self.model.validate_calibration(self.frame)
        self.assertFalse((self.cfg.paths.models / self.model.ARTIFACT_NAME).exists())


if __name__ == "__main__":
    unittest.main()
