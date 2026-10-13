"""Contracts for optional corrections to the frozen startup prediction."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin

from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
    StartupCorrectionRegressor, BlendedStartupCorrection, SensorBlendStartupCorrection,
)


class StartupCorrectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(80, 101))
        self.x[:, 10:14] = 0.
        self.x[:40, 10] = 1.
        self.x[40:, 11] = 1.
        self.y = np.column_stack([self.x[:, 38] * 2., self.x[:, 59] * .3])

    def test_serialization_and_batching_preserve_multisensor_predictions(self):
        model = StartupCorrectionRegressor(grouped=True, alpha=.1).fit(self.x, self.y)
        expected = model.predict(self.x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correction.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict(self.x), expected)
        np.testing.assert_allclose(np.vstack([restored.predict(self.x[:17]), restored.predict(self.x[17:])]), expected)

    def test_local_models_do_not_mix_training_labels_between_subsets(self):
        first = StartupCorrectionRegressor(grouped=True).fit(self.x, self.y)
        changed = self.y.copy()
        changed[40:] += 100.
        second = StartupCorrectionRegressor(grouped=True).fit(self.x, changed)
        np.testing.assert_allclose(first.predict(self.x[:40]), second.predict(self.x[:40]))

    def test_unseen_subset_has_finite_fallback(self):
        model = StartupCorrectionRegressor(grouped=True).fit(self.x, self.y)
        unseen = self.x[:5].copy()
        unseen[:, 10:14] = [0., 0., 1., 0.]
        predicted = model.predict(unseen)
        self.assertEqual(predicted.shape, (5, 2))
        self.assertTrue(np.isfinite(predicted).all())

    def test_balanced_targets_restore_original_prediction_scale(self):
        plain = StartupCorrectionRegressor(family="ridge", alpha=10.).fit(self.x, self.y)
        balanced = StartupCorrectionRegressor(family="ridge", alpha=10., balanced=True).fit(self.x, self.y)
        np.testing.assert_allclose(plain.predict(self.x), balanced.predict(self.x), atol=1e-10)

    def test_blend_preserves_fixed_weights_after_reload(self):
        first = StartupCorrectionRegressor(family="ridge").fit(self.x, self.y)
        second = StartupCorrectionRegressor(grouped=True).fit(self.x, self.y)
        blend = BlendedStartupCorrection(first, second, .25)
        np.testing.assert_allclose(blend.predict(self.x), .75 * first.predict(self.x) + .25 * second.predict(self.x))

    def test_feature_transforms_keep_predictions_independent_of_row_order(self):
        order = np.random.default_rng(10).permutation(len(self.x))
        for feature in ("compact", "quadratic", "spline", "interactions", "profile"):
            with self.subTest(feature=feature):
                model = StartupCorrectionRegressor(family="ridge", features=feature).fit(self.x, self.y)
                np.testing.assert_allclose(model.predict(self.x)[order], model.predict(self.x[order]), atol=1e-10)

    def check_retrain_gate(self, candidate_score):
        config = Config()
        config.mlops.log_to_file = False
        config.mlops.enable_mlflow = False
        sensors = [f"sensor_{i}" for i in range(1, 22)]
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(360, 21)), columns=sensors)
        frame = frame.assign(unit=np.repeat(np.arange(6), 60), cycle=np.tile(np.arange(1, 61), 6),
            subset_id="FD001", operating_mode=0, RUL=200,
            op_setting_1=0., op_setting_2=0., op_setting_3=0.)
        reference = frame[["unit", "cycle"]].copy()
        reference[["expected_" + s for s in sensors]] = 0.
        with tempfile.TemporaryDirectory() as directory:
            config.paths.models = Path(directory)
            recipe = {"name": "validated_test", "spec": {"family": "ridge", "bins": 40}, "weight": .5}
            (Path(directory) / "startup_profile_recipe.json").write_text(json.dumps(recipe))
            ensemble = EnsembleTwin(config, twins=[])
            ensemble.target_names = sensors
            scores = [{"r2": .3, "rmse": .6}, {"r2": .5, "rmse": .4},
                      {"r2": .6, "rmse": .39}, candidate_score]
            with patch.object(ensemble, "predict", return_value=reference), \
                 patch.object(BaseTwin, "_score", side_effect=scores):
                ensemble._fit_startup_profile(frame)
        return ensemble.startup_profile

    def test_saved_recipe_retrain_rejects_r2_gain_with_worse_rmse(self):
        bundle = self.check_retrain_gate({"r2": .7, "rmse": .4})
        self.assertEqual(bundle["correction_recipe"], "kernel_rbf_0.001_alpha1")
        self.assertNotIn("validated_correction_spec", bundle)

    def test_saved_recipe_retrain_accepts_joint_improvement(self):
        bundle = self.check_retrain_gate({"r2": .7, "rmse": .38})
        self.assertEqual(bundle["correction_recipe"], "validated_test")
        self.assertEqual(bundle["validated_blend_weight"], .5)

    def test_sensor_blend_uses_a_separate_convex_weight_for_each_target(self):
        base = StartupCorrectionRegressor(family="ridge").fit(self.x, self.y)
        member = StartupCorrectionRegressor(grouped=True).fit(self.x, self.y)
        model = SensorBlendStartupCorrection(base, [member], [[1., .25]])
        predicted = model.predict(self.x)
        np.testing.assert_allclose(predicted[:,0], member.predict(self.x)[:,0])
        np.testing.assert_allclose(predicted[:,1], .75*base.predict(self.x)[:,1] + .25*member.predict(self.x)[:,1])
        for invalid in ([[1.1, 0.]], [[-.1, 0.]], [[float("nan"), 0.]]):
            with self.assertRaisesRegex(ValueError, "convex"):
                SensorBlendStartupCorrection(base, [member], invalid).predict(self.x)


if __name__ == "__main__":
    unittest.main()
