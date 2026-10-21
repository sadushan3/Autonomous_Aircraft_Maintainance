import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import CausalSensorRegressor,prior_sensor_features
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin


class CausalForecastContracts(unittest.TestCase):
    def setUp(self):
        self.targets=[f"sensor_{i}" for i in range(1,22)]
        rng=np.random.default_rng(42)
        self.frame=pd.DataFrame(rng.normal(size=(120,21)),columns=self.targets)
        self.frame=self.frame.assign(unit=np.repeat([1,2],60),cycle=np.tile(np.arange(1,61),2),
            operating_mode=np.tile(np.arange(60)%6,2),subset_id="FD001",RUL=200,
            op_setting_1=0.,op_setting_2=0.,op_setting_3=0.)
        self.bundle={"target_names":self.targets,"subsets":["FD001","FD002","FD003","FD004"],
            "n_modes":6,"commissioning_cycles":30,"age_reference_cycles":400.,"mode_prior":5.,
            "feature_schema":"causal_sensor_history_v1"}

    def features(self,frame):
        return EnsembleTwin._profile_features(frame,np.zeros((len(frame),21)),self.bundle)

    def test_current_and_future_readings_do_not_change_a_forecast(self):
        x,ready=self.features(self.frame)
        changed=self.frame.copy()
        changed.loc[changed.cycle.ge(44),self.targets]+=1000.
        actual,_=self.features(changed)
        past=self.frame.cycle.le(44).to_numpy()
        np.testing.assert_allclose(x[past],actual[past],atol=1e-12,rtol=0)
        self.assertTrue(np.any(np.abs(x[~past]-actual[~past])>1.))
        model=CausalSensorRegressor().fit(x[ready],self.frame.loc[ready,self.targets].to_numpy())
        np.testing.assert_allclose(model.predict(x[past]),model.predict(actual[past]),atol=1e-12,rtol=0)

    def test_shifted_ewm_and_lags_match_hand_computed_values(self):
        x,_=self.features(self.frame)
        values=self.frame.sensor_1.to_numpy()
        ewm=values[0]
        for value in values[1:10]:
            ewm=.5*value+.5*ewm
        self.assertAlmostEqual(x[10,234],ewm)
        self.assertAlmostEqual(x[10,234+84],values[4])
        self.assertEqual(x[10,234+147],values[9])
        self.assertEqual(x[10,234+168],values[4])

        np.testing.assert_array_equal(x[60,234:234+189],0.)

    def test_prefix_shuffling_and_RUL_do_not_change_forecasts(self):
        x,ready=self.features(self.frame)
        prefix=self.frame.loc[self.frame.cycle.le(44)].copy()
        prefix.RUL=-100000
        actual,actual_ready=self.features(prefix)
        np.testing.assert_allclose(actual,x[prefix.index],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(actual_ready,ready[prefix.index])
        shuffled=self.frame.sample(frac=1.,random_state=42)
        actual,_=self.features(shuffled)
        np.testing.assert_allclose(actual,x[shuffled.index],atol=1e-12,rtol=0)

    def test_incomplete_commissioning_never_activates_forecast(self):
        incomplete=self.frame.loc[~((self.frame.unit==1)&(self.frame.cycle==12))]
        _,ready=self.features(incomplete)
        self.assertFalse(ready[incomplete.unit.eq(1).to_numpy()].any())
        self.assertTrue(ready[incomplete.unit.eq(2).to_numpy()].any())

    def test_duplicate_cycles_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"unique engine/cycle"):
            self.features(pd.concat([self.frame,self.frame.iloc[[40]]]))

    def test_reload_and_prediction_batches_match(self):
        x,ready=self.features(self.frame)
        model=CausalSensorRegressor(features="combined").fit(x[ready],self.frame.loc[ready,self.targets].to_numpy())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"forecast.pkl"
            joblib.dump(model,path)
            loaded=joblib.load(path)
        expected=model.predict(x)
        np.testing.assert_allclose(expected,loaded.predict(x),atol=1e-12,rtol=0)
        np.testing.assert_allclose(expected,np.vstack([loaded.predict(x[:35]),loaded.predict(x[35:])]),atol=1e-12,rtol=0)

    def test_invalid_feature_shape_and_unknown_policy_are_rejected(self):
        x,ready=self.features(self.frame)
        y=self.frame.loc[ready,self.targets].to_numpy()
        with self.assertRaisesRegex(ValueError,"427 finite"):
            CausalSensorRegressor().fit(x[ready,:100],y)
        with self.assertRaisesRegex(ValueError,"policy"):
            CausalSensorRegressor(family="unknown").fit(x[ready],y)

    def test_carried_mode_states_exclude_current_and_future_observations(self):
        def multimode(frame):
            return prior_sensor_features(frame,np.zeros((len(frame),21)),self.bundle,multimode=True)
        x,ready=multimode(self.frame)
        self.assertEqual(x.shape,(120,691))
        self.bundle["feature_schema"]="causal_multimode_history_v1"
        routed,routed_ready=self.features(self.frame)
        np.testing.assert_allclose(routed,x,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(routed_ready,ready)
        self.assertEqual(x[9,427+3*21],self.frame.sensor_1.iloc[3])
        np.testing.assert_array_equal(x[60,427:685],0.)
        changed=self.frame.copy()
        changed.loc[changed.cycle.ge(44),self.targets]+=1000.
        actual,_=multimode(changed)
        past=self.frame.cycle.le(44).to_numpy()
        np.testing.assert_allclose(actual[past],x[past],atol=1e-12,rtol=0)
        prefix=self.frame.loc[self.frame.cycle.le(44)]
        px,_=multimode(prefix)
        np.testing.assert_allclose(px,x[prefix.index],atol=1e-12,rtol=0)
        shuffled=self.frame.sample(frac=1.,random_state=42)
        sx,_=multimode(shuffled)
        np.testing.assert_allclose(sx,x[shuffled.index],atol=1e-12,rtol=0)
        model=CausalSensorRegressor(features="multimode").fit(x[ready],self.frame.loc[ready,self.targets].to_numpy())
        np.testing.assert_allclose(model.predict(x[past]),model.predict(actual[past]),atol=1e-12,rtol=0)


if __name__=="__main__":
    unittest.main()
