import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features
from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import (
    TrendDecoder, _trend_features, trend_features,
)


class TrendForecastContracts(unittest.TestCase):
    def setUp(self):
        self.targets = [f'sensor_{i}' for i in range(1, 22)]
        self.frame = self.make_frame(2, 70)
        self.bundle = dict(target_names=self.targets, subsets=['FD001', 'FD002', 'FD003', 'FD004'],
                           n_modes=6, commissioning_cycles=30, age_reference_cycles=400., mode_prior=5.,
                           feature_schema='causal_trend_history_v1')

    def make_frame(self, engines, cycles):
        rng = np.random.default_rng(78)
        frame = pd.DataFrame(rng.normal(size=(engines * cycles, 21)), columns=self.targets)
        frame = frame.assign(unit=np.repeat(np.arange(1, engines + 1), cycles),
                             cycle=np.tile(np.arange(1, cycles + 1), engines),
                             operating_mode=np.tile(np.arange(cycles) % 6, engines),
                             subset_id='FD001', op_setting_1=np.arange(len(frame)) * .001,
                             op_setting_2=0., op_setting_3=0., RUL=140.)
        frame.sensor_1 = .1 * frame.unit + .01 * frame.cycle
        return frame

    def features(self, frame):
        return trend_features(frame, np.zeros((len(frame), 21)), self.bundle)

    def test_known_past_means_and_linear_extrapolation(self):
        x, ready = self.features(self.frame)
        self.assertEqual(x.shape, (140, 1141))
        self.assertAlmostEqual(x[10, 952], .155)
        self.assertAlmostEqual(x[10, 973], .15)
        self.assertAlmostEqual(x[10, 994], .15)
        for offset in (1015, 1036, 1057, 1078, 1099, 1120):
            self.assertAlmostEqual(x[34, offset], .45, places=10)
        np.testing.assert_array_equal(x[[0, 70], 952:], 0.)
        self.assertEqual(int(ready.sum()), 80)

    def test_current_future_rul_prefix_and_order_cannot_leak(self):
        expected, ready = self.features(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.cycle.ge(44), self.targets] += 1000.
        changed.RUL = -9999.
        actual, changed_ready = self.features(changed)
        safe = self.frame.cycle.between(31, 44).to_numpy()
        np.testing.assert_array_equal(actual[safe], expected[safe])
        np.testing.assert_array_equal(ready, changed_ready)
        without_labels, _ = self.features(self.frame.drop(columns='RUL'))
        np.testing.assert_array_equal(without_labels, expected)
        prefix = self.frame.cycle.le(55).to_numpy()
        short, short_ready = self.features(self.frame.loc[prefix].reset_index(drop=True))
        np.testing.assert_array_equal(short, expected[prefix])
        np.testing.assert_array_equal(short_ready, ready[prefix])
        order = np.random.default_rng(14).permutation(len(self.frame))
        shuffled, shuffled_ready = self.features(self.frame.iloc[order].reset_index(drop=True))
        np.testing.assert_allclose(shuffled, expected[order], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(shuffled_ready, ready[order])

    def test_chunking_and_router_preserve_original_952_columns_exactly(self):
        frame = self.make_frame(25, 40).sample(frac=1., random_state=17).reset_index(drop=True)
        population = np.zeros((len(frame), 21))
        batched, ready = trend_features(frame, population, self.bundle)
        reference, reference_ready = _trend_features(frame, population, self.bundle)
        np.testing.assert_array_equal(batched, reference)
        np.testing.assert_array_equal(ready, reference_ready)
        legacy, legacy_ready = fast_mode_features(frame, population, self.bundle)
        np.testing.assert_array_equal(batched[:, :952], legacy)
        np.testing.assert_array_equal(ready, legacy_ready)
        routed, routed_ready = EnsembleTwin._profile_features(frame, population, self.bundle)
        np.testing.assert_array_equal(routed, batched)
        np.testing.assert_array_equal(routed_ready, ready)

    def test_decoder_reload_batches_constants_and_cache_precision(self):
        x, ready = self.features(self.frame)
        target = self.frame.loc[ready, self.targets].to_numpy()
        target[:, 0] = 1.3
        specification = dict(family='extra_trees', n_estimators=4, leaf=3)
        model = TrendDecoder(**specification).fit(x[ready], target)
        expected = model.predict(x)
        self.assertEqual(expected.shape, (140, 21))
        np.testing.assert_allclose(expected[:, 0], 1.3, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.predict(x.astype(np.float32)), expected, rtol=0, atol=1e-12)
        refit = TrendDecoder(**specification).fit(x[ready].astype(np.float32), target)
        np.testing.assert_allclose(refit.predict(x), expected, rtol=0, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trend.pkl'
            joblib.dump(model, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(loaded.predict(x), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(np.vstack([loaded.predict(x[:37]), loaded.predict(x[37:])]), expected, rtol=0, atol=1e-12)
        unknown = x[:3].copy()
        unknown[:, 10:14] = [0, 1, 0, 0]
        self.assertTrue(np.isfinite(model.predict(unknown)).all())

    def test_engine_streaming_matches_global_prediction_and_preserves_fallbacks(self):
        frame = self.make_frame(25, 50).sample(frac=1., random_state=8).reset_index(drop=True)
        frame.loc[frame.unit.eq(25), 'subset_id'] = 'unknown'
        frame.loc[frame.unit.eq(24), 'operating_mode'] = 8
        population = np.zeros((len(frame), 21))
        x, ready = self.features(frame)
        model = Ridge(alpha=1.).fit(x[ready], frame.loc[ready, self.targets].to_numpy())
        predictor = object.__new__(EnsembleTwin)
        predictor.config = SimpleNamespace(twin=SimpleNamespace(startup_profile_enabled=True, commissioning_cycles=30))
        predictor.target_names = self.targets
        predictor.startup_profile = dict(self.bundle, correction=model,
            population=DummyRegressor(strategy='constant', constant=np.zeros(21)).fit(np.zeros((3, 38)), np.zeros((3, 21))))
        reference = np.full((len(frame), 21), -7.)
        expected = reference.copy()
        ready &= frame.subset_id.isin(self.bundle['subsets']).to_numpy()
        ready &= frame.operating_mode.between(0, 5).to_numpy()
        expected[ready] = EnsembleTwin._predict_profile_correction(model, x, row_mask=ready)
        actual, actual_ready = predictor._profile_prediction(frame, reference)
        np.testing.assert_array_equal(actual_ready, ready)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(actual[~ready], reference[~ready])

    def test_invalid_schema_policy_and_labels_are_rejected(self):
        x, ready = self.features(self.frame)
        y = self.frame.loc[ready, self.targets].to_numpy()
        with self.assertRaisesRegex(ValueError, '1141'):
            TrendDecoder().fit(x[ready, :952], y)
        with self.assertRaisesRegex(ValueError, 'policy'):
            TrendDecoder(history='future').fit(x[ready], y)
        with self.assertRaisesRegex(ValueError, 'policy'):
            TrendDecoder(features='unknown').fit(x[ready], y)
        with self.assertRaisesRegex(ValueError, 'labels'):
            TrendDecoder().fit(x[ready], y[:, :20])


if __name__ == '__main__':
    unittest.main()
