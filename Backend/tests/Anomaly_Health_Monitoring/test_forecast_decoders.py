import tempfile,unittest,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import prior_sensor_features
from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features
from app.services.Anomaly_Health_Monitoring.digital_twin.independent_forecast import IndependentSensorForecast
from app.services.Anomaly_Health_Monitoring.digital_twin.multitask_forecast import MultitaskSensorForecast
from app.services.Anomaly_Health_Monitoring.digital_twin.forecast_policy import healthy_forecast_guard
from app.services.Anomaly_Health_Monitoring.digital_twin.context_mixture import ContextSensorMixture
from sklearn.dummy import DummyRegressor

class ForecastDecoderContracts(unittest.TestCase):
    def setUp(self):
        self.targets=[f'sensor_{i}' for i in range(1,22)]
        rng=np.random.default_rng(19)
        self.frame=pd.DataFrame(rng.normal(size=(180,21)),columns=self.targets)
        self.frame=self.frame.assign(unit=np.repeat([1,2],90),cycle=np.tile(np.arange(1,91),2),
            operating_mode=np.tile(np.arange(90)%6,2),subset_id='FD001',RUL=np.tile(150-1.5*np.arange(90),2),
            op_setting_1=np.arange(180),op_setting_2=0.,op_setting_3=0.)
        self.bundle={'target_names':self.targets,'subsets':['FD001','FD002','FD003','FD004'],
            'n_modes':6,'commissioning_cycles':30,'age_reference_cycles':400.,'mode_prior':5.,
            'feature_schema':'causal_fast_modes_v1'}

    def features(self,frame):
        return fast_mode_features(frame,np.zeros((len(frame),21)),self.bundle)

    def test_fast_history_hand_values_and_engine_reset(self):
        x,_=self.features(self.frame); values=self.frame.sensor_1.to_numpy()
        self.assertEqual(x.shape,(180,952))
        self.assertAlmostEqual(x[10,691+3*21],.5*values[3]+.5*values[9])
        self.assertEqual(x[9,817+3*21],values[3])
        self.assertEqual(x[10,943+3],1.)
        self.assertEqual(x[10,949],9.)
        np.testing.assert_array_equal(x[90,691:],0.)

    def test_feature_schema_routes_and_preserves_original_columns(self):
        expected,ready=self.features(self.frame)
        actual,actual_ready=EnsembleTwin._profile_features(self.frame,np.zeros((len(self.frame),21)),self.bundle)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(ready,actual_ready)
        old,_=prior_sensor_features(self.frame,np.zeros((len(self.frame),21)),self.bundle,multimode=True)
        np.testing.assert_array_equal(expected[:,:691],old)

    def test_current_future_labels_prefix_and_row_order_are_safe(self):
        expected,_=self.features(self.frame)
        changed=self.frame.copy(); changed.loc[changed.cycle.ge(44),self.targets]+=1000.; changed.RUL=-9999
        actual,_=self.features(changed)
        past=self.frame.cycle.le(44).to_numpy()
        np.testing.assert_allclose(actual[past],expected[past],rtol=0,atol=1e-12)
        self.assertTrue(np.any(np.abs(actual[~past]-expected[~past])>1.))
        prefix=self.frame.loc[self.frame.cycle.le(44)].drop(columns='RUL')
        actual,_=self.features(prefix)
        np.testing.assert_allclose(actual,expected[prefix.index],rtol=0,atol=1e-12)
        shuffled=self.frame.sample(frac=1.,random_state=7)
        actual,_=self.features(shuffled)
        np.testing.assert_allclose(actual,expected[shuffled.index],rtol=0,atol=1e-12)

    def test_decoder_reload_and_batches_preserve_all_sensor_outputs(self):
        x,ready=self.features(self.frame); y=self.frame.loc[ready,self.targets].to_numpy()
        for history,width in (('standard',691),('fast_modes',952)):
            with self.subTest(history=history):
                model=IndependentSensorForecast(history=history).fit(x[ready,:width],y)
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'decoder.pkl'; joblib.dump(model,path); loaded=joblib.load(path)
                expected=model.predict(x[:,:width])
                self.assertEqual(expected.shape,(180,21))
                np.testing.assert_allclose(expected,loaded.predict(x[:,:width]),rtol=0,atol=1e-12)
                np.testing.assert_allclose(expected,np.vstack([loaded.predict(x[:41,:width]),loaded.predict(x[41:,:width])]),rtol=0,atol=1e-12)

    def test_boosted_refitting_is_deterministic_and_constants_are_predicted(self):
        x,ready=self.features(self.frame); y=self.frame.loc[ready,self.targets].to_numpy(); y[:,0]=1.3
        spec=dict(family='lightgbm',grouping='subset',n_estimators=5,leaf=3,leaves=3)
        first=IndependentSensorForecast(**spec).fit(x[ready,:691],y)
        second=IndependentSensorForecast(**spec).fit(x[ready,:691],y)
        a,b=first.predict(x[:,:691]),second.predict(x[:,:691])
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        np.testing.assert_allclose(a[:,0],1.3,rtol=0,atol=1e-12)

    def test_auxiliary_lifetime_is_fitting_only_and_not_a_scored_output(self):
        x,ready=self.features(self.frame); y=self.frame.loc[ready,self.targets].to_numpy()
        with self.assertRaisesRegex(ValueError,'required only during fitting'):
            MultitaskSensorForecast(auxiliary_weight=1.).fit(x[ready,:427],y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model=MultitaskSensorForecast(auxiliary_weight=1.,max_iter=3).fit(x[ready,:427],y,self.frame.loc[ready,'RUL'].to_numpy())
        self.assertEqual(model.predict(x[:,:427]).shape,(180,21))
        with self.assertRaises(TypeError): model.predict(x[:,:427],rul=self.frame.RUL.to_numpy())

    def test_invalid_history_and_forecast_policy_are_rejected(self):
        x,ready=self.features(self.frame); y=self.frame.loc[ready,self.targets].to_numpy()
        with self.assertRaisesRegex(ValueError,'policy'):
            IndependentSensorForecast(grouping='unknown').fit(x[ready,:691],y)
        with self.assertRaisesRegex(ValueError,'history'):
            IndependentSensorForecast(history='unknown').fit(x[ready,:691],y)

    def test_streamed_ready_rows_match_explicitly_filtered_forecasts(self):
        x,ready=self.features(self.frame); y=self.frame.loc[ready,self.targets].to_numpy()
        model=IndependentSensorForecast(history='fast_modes').fit(x[ready],y)
        expected=EnsembleTwin._predict_profile_correction(model,x[ready])
        actual=EnsembleTwin._predict_profile_correction(model,x,row_mask=ready)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'row mask'):
            EnsembleTwin._predict_profile_correction(model,x,row_mask=ready[:-1])

    def test_secondary_healthy_guard_enforces_all_published_limits(self):
        before={'r2':.6,'rmse':.4,'mae':.2}
        self.assertTrue(healthy_forecast_guard(before,{'r2':.596,'rmse':.403,'mae':.201}))
        self.assertFalse(healthy_forecast_guard(before,{'r2':.594,'rmse':.4,'mae':.2}))
        self.assertFalse(healthy_forecast_guard(before,{'r2':.6,'rmse':.405,'mae':.2}))
        self.assertFalse(healthy_forecast_guard(before,{'r2':.6,'rmse':.4,'mae':.203}))
        self.assertFalse(healthy_forecast_guard(before,{'r2':float('nan'),'rmse':.4,'mae':.2}))

    def context_model(self, weights, grouping='subset_age', age_cut=.25):
        def constant(value):
            return DummyRegressor(strategy='constant',constant=np.full(21,value)).fit(np.zeros((3,14)),np.zeros((3,21)))
        return ContextSensorMixture(constant(0.),[constant(2.),constant(4.)],weights,
                                    np.full((2,21),.25),grouping,age_cut)

    def test_context_weights_follow_known_age_and_fallback_for_unknown_group(self):
        first=np.vstack([np.full(21,.5),np.zeros(21)])
        second=np.vstack([np.zeros(21),np.full(21,.5)])
        model=self.context_model({'0':first.tolist(),'1':second.tolist()})
        x=np.zeros((4,14));x[:,10]=1.;x[:,3]=[.2,.25,.3,.8]
        x[3,10]=0.;x[3,11]=1.
        np.testing.assert_array_equal(model.predict(x),np.repeat([[1.],[1.],[2.],[1.5]],21,axis=1))

    def test_context_mixture_reload_batches_and_order_preserve_predictions(self):
        model=self.context_model({0:np.full((2,21),.2),1:np.full((2,21),.4)})
        x=np.zeros((2080,14));x[:,10]=1.;x[:,3]=np.linspace(0,1,len(x))
        expected=model.predict(x)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'context.pkl';joblib.dump(model,path);loaded=joblib.load(path)
            np.testing.assert_array_equal(loaded.predict(x),expected)
        np.testing.assert_allclose(EnsembleTwin._predict_profile_correction(model,x),expected,rtol=0,atol=1e-12)
        order=np.random.default_rng(44).permutation(len(x))
        np.testing.assert_array_equal(model.predict(x[order]),expected[order])

    def test_context_mixture_rejects_nonconvex_or_invalid_policies(self):
        x=np.zeros((2,14));x[:,10]=1.
        for weights in (np.full((2,21),.6),np.full((2,21),-.1),np.full((2,21),np.nan)):
            with self.subTest(weights=weights[0,0]),self.assertRaisesRegex(ValueError,'convex'):
                self.context_model({0:weights}).predict(x)
        with self.assertRaisesRegex(ValueError,'policy'):
            self.context_model({},grouping='unknown').predict(x)

if __name__=='__main__': unittest.main()
