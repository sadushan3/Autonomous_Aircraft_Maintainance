import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import ProfileRegressor
from app.services.Anomaly_Health_Monitoring.digital_twin.validated_profile import joint_improvement
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError


class ProfileModelContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.targets = [f"sensor_{i}" for i in range(1,22)]
        self.frame = pd.DataFrame(rng.normal(size=(120,21)), columns=self.targets)
        self.frame = self.frame.assign(unit=np.repeat([1,2],60), cycle=np.tile(np.arange(1,61),2),
            operating_mode=np.tile(np.arange(60)%6,2), subset_id="FD001", RUL=200,
            op_setting_1=0., op_setting_2=0., op_setting_3=0.)
        self.bundle = {"target_names":self.targets, "subsets":["FD001","FD002","FD003","FD004"],
            "n_modes":6, "commissioning_cycles":30, "age_reference_cycles":400., "mode_prior":5.,
            "feature_schema":"commissioning_statistics_v1"}

    def features(self, frame):
        return EnsembleTwin._profile_features(frame, np.zeros((len(frame),21)), self.bundle)

    def test_late_observations_and_RUL_cannot_change_frozen_statistics(self):
        original, ready = self.features(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.cycle.gt(30), self.targets] += 10000.
        changed.RUL = -10000
        actual, actual_ready = self.features(changed)
        np.testing.assert_array_equal(ready, actual_ready)
        np.testing.assert_allclose(original, actual, atol=1e-12, rtol=0)

    def test_prefix_and_row_order_invariance(self):
        expected, _ = self.features(self.frame)
        prefix = self.frame.loc[self.frame.cycle.le(40)]
        actual, _ = self.features(prefix)
        np.testing.assert_allclose(actual, expected[prefix.index], atol=1e-12, rtol=0)
        shuffled = self.frame.sample(frac=1., random_state=2)
        actual, _ = self.features(shuffled)
        np.testing.assert_allclose(actual, expected[shuffled.index], atol=1e-12, rtol=0)

    def test_missing_commissioning_observation_keeps_profile_unready(self):
        incomplete = self.frame.loc[~((self.frame.unit==1) & (self.frame.cycle==17))]
        _, ready = self.features(incomplete)
        self.assertFalse(ready[incomplete.unit.eq(1).to_numpy()].any())
        self.assertTrue(ready[incomplete.unit.eq(2).to_numpy()].any())

    def test_unknown_feature_schema_is_rejected(self):
        self.bundle["feature_schema"] = "unknown"
        with self.assertRaises(AnomalyHealthError):
            self.features(self.frame)

    def test_model_reload_and_batching_preserve_outputs(self):
        x, _ = self.features(self.frame)
        y = self.frame[self.targets].to_numpy(float)
        model = ProfileRegressor(family="ridge", grouping="subset_mode", alpha=10.).fit(x,y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"profile.pkl"
            joblib.dump(model,path)
            restored = joblib.load(path)
        np.testing.assert_allclose(model.predict(x), restored.predict(x), atol=1e-12, rtol=0)
        np.testing.assert_allclose(model.predict(x), np.vstack([restored.predict(x[:35]), restored.predict(x[35:])]), atol=1e-12, rtol=0)

    def test_r2_gain_with_worse_MAE_is_rejected(self):
        self.assertFalse(joint_improvement({"r2":.5,"rmse":.4,"mae":.2},
                                          {"r2":.6,"rmse":.3,"mae":.21}))


if __name__ == "__main__":
    unittest.main()
