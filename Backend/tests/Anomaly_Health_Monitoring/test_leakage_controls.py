import copy
import json
import importlib
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.feature_engineering import FeatureEngineer
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.data_loader import DataLoader
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import ResidualCalculator
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import RandomForestTwin
from app.services.Anomaly_Health_Monitoring.context_modeling.context_drift import ContextDriftDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.condition_consensus import ConditionConsensusDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.anomaly_detection.severity_classifier import SeverityClassifier
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_candidate_improves
from app.utils.Anomaly_Health_Monitoring.training_split import (
    prepare_training_split, training_scope, record_training_scope, load_training_split,
)
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, atomic_write_json


class LeakageControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config()
        for name in ("models", "scalers", "processed", "metrics", "reports", "logs"):
            setattr(self.cfg.paths, name, Path(self.temp.name) / name)
        self.cfg.paths.scalers = self.cfg.paths.models / "scalers"
        self.cfg.mlops.log_to_file = False
        self.cfg.mlops.enable_mlflow = False
        self.cfg.data.drop_constant_sensors = False
        units = np.repeat(np.arange(40), 80)
        self.frame = pd.DataFrame({"unit": units, "cycle": np.tile(np.arange(1, 81), 40),
            "split": "train", "subset_id": np.repeat(np.tile(["FD001", "FD002", "FD003", "FD004"], 10), 80),
            "operating_mode": 0, "RUL": np.tile(np.r_[np.full(50, 200), np.full(30, 20)], 40)})
        rng = np.random.default_rng(20)
        for sensor in range(1, 22):
            self.frame[f"sensor_{sensor}"] = rng.normal(size=len(self.frame)) + units
        for setting in range(1, 4):
            self.frame[f"op_setting_{setting}"] = rng.normal(size=len(self.frame))

    def plan(self):
        return prepare_training_split(self.frame, self.cfg)

    def test_split_is_disjoint_and_ignores_labels_and_features(self):
        first = self.plan()
        changed = self.frame.assign(RUL=-9999, sensor_1=9999)
        second = prepare_training_split(changed, self.cfg)
        self.assertEqual(first, second)
        roles = [set(values) for values in first["engine_roles"].values()]
        self.assertEqual(sum(map(len, roles)), len(set.union(*roles)))

    def test_validated_forecast_retraining_keeps_role_and_provenance(self):
        from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
        plan = self.plan()
        recipe = {"name":"fixture_forecast", "family":"full_fit_causal_forecast",
                  "training_plan_id":plan["plan_id"], "baseline_recipe":{"name":"fixture_baseline"}}
        atomic_write_json(recipe,self.cfg.paths.models / "startup_profile_recipe.json")
        twin = EnsembleTwin(self.cfg,twins=[])
        twin.target_names = [f"sensor_{i}" for i in range(1,22)]
        twin.predict = Mock(side_effect=lambda frame:pd.DataFrame(
            {f"expected_{sensor}":np.zeros(len(frame)) for sensor in twin.target_names}))
        selected_model = Mock()
        result = {"accepted":True, "model":selected_model,
            "metrics":{"r2":.7,"rmse":.3,"mae":.2,"rows":100},
            "baseline_metrics":{"r2":.6,"rmse":.4,"mae":.3},
            "feature_schema":"causal_sensor_history_v1", "prediction_role":"adaptive_sensor_forecast",
            "fit_unit_ids":plan["engine_roles"]["fit"], "selection_unit_ids":plan["engine_roles"]["selection"]}
        with patch("app.services.Anomaly_Health_Monitoring.digital_twin.validated_profile.refit_validated_profile",return_value=result) as refit:
            twin._fit_startup_profile(self.frame)
        refit.assert_called_once()
        self.assertIs(twin.startup_profile["correction"],selected_model)
        self.assertEqual(twin.startup_profile["training_plan_id"],plan["plan_id"])
        self.assertEqual(twin.startup_profile["prediction_role"],"adaptive_sensor_forecast")
        self.assertEqual(twin.startup_profile["validation_role"],"selection")
        self.assertEqual(twin.startup_profile["correction_recipe"],recipe["name"])
        self.assertTrue(twin.startup_profile["reference_preserved_for_monitoring"])

    def test_test_population_cannot_fit_any_role(self):
        self.plan()
        with self.assertRaises(AnomalyHealthError):
            training_scope(self.frame.assign(split="test"), self.cfg, required=True)

    def test_provenance_rejects_outer_holdout_engines(self):
        self.plan()
        confirmation = training_scope(self.frame, self.cfg, ("confirmation",))
        with self.assertRaises(AnomalyHealthError):
            record_training_scope("scaling", confirmation, self.cfg)

    def test_shared_training_refuses_missing_population_metadata(self):
        self.plan()
        with self.assertRaisesRegex(AnomalyHealthError, "population metadata"):
            training_scope(self.frame.drop(columns="split"), self.cfg)

    def test_training_manifest_cannot_change_under_an_existing_bundle_id(self):
        plan = self.plan()
        plan["engine_roles"]["fit"], plan["engine_roles"]["calibration"] = plan["engine_roles"]["calibration"], plan["engine_roles"]["fit"]
        atomic_write_json(plan, self.cfg.paths.models / "training_split.json")
        with self.assertRaisesRegex(AnomalyHealthError, "bundle identity"):
            load_training_split(self.cfg)

    def test_scaler_fit_does_not_change_when_holdout_features_change(self):
        plan = self.plan()
        changed = self.frame.copy()
        changed.loc[~changed.unit.isin(plan["engine_roles"]["fit"]), "sensor_1"] = 1e8
        first = PerModeScaler(self.cfg).fit(training_scope(self.frame, self.cfg))
        second = PerModeScaler(self.cfg).fit(training_scope(changed, self.cfg))
        np.testing.assert_array_equal(first.scalers[0].mean_, second.scalers[0].mean_)
        np.testing.assert_array_equal(first.scalers[0].var_, second.scalers[0].var_)

    def test_cleaner_reference_ignores_validation_and_applies_frozen_fallback(self):
        cleaned, _ = DataCleaner(self.cfg).clean(self.frame)
        path = self.cfg.paths.models / "imputation_reference.json"
        first = json.loads(path.read_text())
        plan = load_training_split(self.cfg)
        changed = self.frame.copy()
        changed.loc[~changed.unit.isin(plan["engine_roles"]["fit"]), "sensor_1"] = 1e9
        DataCleaner(self.cfg).clean(changed)
        self.assertEqual(json.loads(path.read_text()), first)
        missing = pd.DataFrame({"unit": [500, 500, 500], "cycle": [1, 2, 3], "sensor_1": [np.nan, 2., 3.]})
        cleaner = DataCleaner(self.cfg)
        full, _ = cleaner.repair_values(missing)
        prefix, _ = cleaner.repair_values(missing.iloc[:1])
        self.assertEqual(full.sensor_1.iloc[0], prefix.sensor_1.iloc[0])
        self.assertEqual(full.sensor_1.iloc[0], first["medians"]["sensor_1"])

    def test_missing_history_without_reference_fails_instead_of_backfill(self):
        frame = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "sensor_1": [np.nan, 2.]})
        with self.assertRaisesRegex(AnomalyHealthError, "frozen training"):
            DataCleaner(self.cfg).repair_values(frame)

    def test_forward_fill_is_sorted_engine_local_and_causal(self):
        frame = pd.DataFrame({"unit": [1, 1, 2], "cycle": [2, 1, 1], "sensor_1": [np.nan, 3., 99.]})
        repaired, _ = DataCleaner(self.cfg).repair_values(frame)
        np.testing.assert_array_equal(repaired.sensor_1, [3., 3., 99.])

    def test_engineered_age_is_prefix_invariant_and_has_no_future_maximum(self):
        engineer = FeatureEngineer(self.cfg)
        full = engineer.add_cycle_features(self.frame)
        prefix = self.frame.loc[self.frame.cycle.le(30)]
        early = engineer.add_cycle_features(prefix)
        np.testing.assert_array_equal(early.cycle_norm, full.loc[prefix.index, "cycle_norm"])
        self.assertNotIn("engine_max_cycle", full)
        self.cfg.twin.age_reference_cycles = 0
        with self.assertRaises(ValueError):
            engineer.add_cycle_features(self.frame)

    def test_mixed_reader_populations_have_separate_unit_ids(self):
        reader = CMAPSSReader(self.cfg)
        def read(subset, split, offset):
            result = self.frame.loc[self.frame.unit.eq(0)].copy()
            result["unit"], result["split"], result["subset_id"] = offset + 1, split, subset
            return result
        with patch.object(reader, "read_subset", side_effect=read):
            frame = reader.read_subsets(subsets=["FD001"], split="both")
        a = set(frame.loc[frame.split.eq("train"), "unit"])
        b = set(frame.loc[frame.split.eq("test"), "unit"])
        self.assertFalse(a & b)
        self.assertFalse(frame.duplicated(["unit", "cycle"]).any())

    def test_consensus_uses_global_fit_and_distinct_scale_engines(self):
        self.plan()
        cfg = self.cfg
        cfg.anomaly.condition_consensus.update(n_estimators=3, seeds=[42], max_depth=4)
        frame = self.frame.rename(columns={f"sensor_{i}": f"residual_sensor_{i}" for i in range(1, 22)})
        model = ConditionConsensusDetector(cfg).fit(frame, [f"residual_sensor_{i}" for i in range(1, 22)])
        plan = load_training_split(cfg)
        self.assertEqual(model.provenance["fit_unit_ids"], plan["engine_roles"]["fit"])
        self.assertEqual(model.provenance["calibration_unit_ids"], plan["engine_roles"]["calibration"])
        provenance = json.loads((cfg.paths.models / "training_provenance.json").read_text())
        self.assertEqual(provenance["covariance_reference"]["units"], plan["engine_roles"]["fit"])
        outer = training_scope(frame, cfg, ("confirmation",))
        with self.assertRaises(AnomalyHealthError):
            model.validate_calibration(outer)

    def test_heldout_labels_cannot_change_classifier_fitting(self):
        self.plan()
        self.cfg.anomaly.condition_consensus.update(n_estimators=3, seeds=[42], max_depth=4)
        frame = self.frame.rename(columns={f"sensor_{i}": f"residual_sensor_{i}" for i in range(1, 22)})
        columns = [f"residual_sensor_{i}" for i in range(1, 22)]
        first = ConditionConsensusDetector(self.cfg).fit(frame, columns)
        changed = frame.copy()
        held = load_training_split(self.cfg)["engine_roles"]["confirmation"]
        changed.loc[changed.unit.isin(held), "RUL"] = np.nan
        changed.loc[changed.unit.isin(held), columns] = 1e10
        second = ConditionConsensusDetector(self.cfg).fit(changed, columns)
        np.testing.assert_allclose(first.score(frame), second.score(frame), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(first.predict_flags(frame), second.predict_flags(frame))

    def test_unlabelled_inference_cannot_invent_failure_targets(self):
        loader = DataLoader(self.cfg)
        unknown = self.frame.drop(columns=["RUL", "split"])
        self.assertNotIn("RUL", loader._ensure_rul(unknown))
        self.assertNotIn("RUL", loader._ensure_rul(unknown.assign(split="test")))
        training = loader._ensure_rul(unknown.assign(split="train"))
        self.assertTrue(training.loc[training.cycle.eq(80), "RUL"].eq(0).all())

    def test_invalid_training_labels_cannot_become_critical_zeros(self):
        frame = self.frame.copy()
        frame.loc[0, "RUL"] = np.nan
        with self.assertRaisesRegex(AnomalyHealthError, "missing labels"):
            DataLoader(self.cfg)._coerce_types(frame)

    def test_no_batch_relative_residual_standardization_at_inference(self):
        frame = self.frame.rename(columns={"sensor_1": "residual_sensor_1"})
        with self.assertRaisesRegex(AnomalyHealthError, "frozen residual"):
            ResidualCalculator(self.cfg)._zscore(frame, ["residual_sensor_1"])

    def test_context_drift_never_fits_a_reference_from_inference(self):
        drift = ContextDriftDetector(self.cfg)
        with self.assertRaisesRegex(AnomalyHealthError, "frozen context"):
            drift.detect(self.frame.assign(split="test"))
        self.assertFalse(drift.reference_path.exists())

    def test_context_drift_reference_is_training_only(self):
        self.plan()
        reference = ContextDriftDetector(self.cfg).fit_reference(self.frame)
        fitting = training_scope(self.frame, self.cfg)
        self.assertEqual(reference["n_rows"], len(fitting))
        self.assertEqual(reference["setting_means"]["op_setting_1"], fitting.op_setting_1.mean())

    def test_twin_fitting_and_model_selection_use_the_shared_engines(self):
        plan = self.plan()
        frame = self.frame.assign(op_setting_1=self.frame.unit.to_numpy(float))
        self.cfg.twin.random_forest.update(n_estimators=2, max_depth=3, n_jobs=1)
        twin = RandomForestTwin(self.cfg)
        captured = {}
        original = twin._fit_estimator
        def fit(estimator, x_train, y_train, x_val, y_val):
            captured["fit"] = set(map(int, x_train[:, 0]))
            captured["selection"] = set(map(int, x_val[:, 0]))
            return original(estimator, x_train, y_train, x_val, y_val)
        with patch.object(twin, "_fit_estimator", side_effect=fit):
            twin.train(frame)
        self.assertEqual(captured["fit"], set(plan["engine_roles"]["fit"]))
        self.assertEqual(captured["selection"], set(plan["engine_roles"]["selection"]))

    def test_unknown_scaler_mode_does_not_depend_on_later_batch_rows(self):
        frame = self.frame.iloc[:100].copy()
        frame["operating_mode"] = np.where(frame.index < 50, 0, 1)
        frame["op_setting_1"] = np.where(frame.index < 50, 0., 100.)
        scaler = PerModeScaler(self.cfg).fit(frame)
        fresh = frame.iloc[:2].copy()
        fresh["operating_mode"] = 99
        fresh["op_setting_1"] = [0., 1000.]
        full = scaler.transform(fresh)
        prefix = scaler.transform(fresh.iloc[:1])
        np.testing.assert_array_equal(full.iloc[0][scaler.columns], prefix.iloc[0][scaler.columns])

    def test_confirmation_cannot_select_a_development_margin(self):
        self.plan()
        outer = training_scope(self.frame, self.cfg, ("confirmation",))
        model = ConditionConsensusDetector(self.cfg)
        with patch.object(model, "score") as score:
            with self.assertRaises(AnomalyHealthError):
                model.calibrate_development_margin(outer)
            score.assert_not_called()

    def test_near_one_severity_cuts_remain_reachable_and_persist(self):
        frame = pd.DataFrame({"unit": 1, "cycle": np.arange(100), "threshold_used": .5,
            "anomaly_type": "fault", "anomaly_score": np.linspace(.999998, .99999999, 100)})
        classifier = SeverityClassifier(self.cfg)
        bands = classifier.calibrate(frame)
        self.assertLess(bands["critical"], 1.)
        self.assertLess(bands["high"], bands["critical"])
        self.assertEqual(classifier.classify(frame).severity.iloc[-1], "critical")
        loaded = SeverityClassifier(self.cfg)
        self.assertEqual(loaded.band_starts, bands)
        self.assertEqual(list(classifier.classify(frame).severity), list(loaded.classify(frame).severity))

    def test_degenerate_severity_calibration_persists_its_fallback(self):
        frame = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "anomaly_type": "fault", "anomaly_score": .8})
        classifier = SeverityClassifier(self.cfg)
        bands = classifier.calibrate(frame)
        self.assertEqual(classifier.band_starts, bands)
        self.assertEqual(SeverityClassifier(self.cfg).band_starts, bands)

    def test_development_policy_prioritizes_f1_recall_with_overall_ranking(self):
        before = dict(f1=.9, recall=.8, false_alarm_rate=.001, auc=.99, average_precision=.98)
        after = dict(f1=.93, recall=.85, false_alarm_rate=.0011, auc=.995, average_precision=.99)
        before["per_subset"] = {"FD001": copy.deepcopy(before)}
        after["per_subset"] = {"FD001": copy.deepcopy(after)}
        after["per_subset"]["FD001"].update(auc=.989, average_precision=.979)
        self.assertTrue(condition_candidate_improves(before, after, .001))
        after["per_subset"]["FD001"]["false_alarm_rate"] = .0021
        self.assertFalse(condition_candidate_improves(before, after, .001))

    def test_offline_pipeline_never_adapts_from_evaluation_feedback(self):
        pipeline = importlib.import_module("app.services.Anomaly_Health_Monitoring.pipeline.10_full_pipeline")
        stages = {name: Mock(return_value={}) for name in pipeline._STAGE_FILES}
        with patch.object(pipeline, "_load_stage_module", side_effect=lambda name: SimpleNamespace(run_stage=stages[name])), \
             patch.object(pipeline.DashboardDataGenerator, "run", return_value=(pd.DataFrame(), {})), \
             patch.object(pipeline.DashboardApiService, "run", return_value={}), \
             patch.object(pipeline, "_run_evaluations", return_value={}):
            for retrain in (False, True):
                result = pipeline.run_full_pipeline(self.cfg, retrain=retrain, include_shap=False)
                self.assertEqual(result["status"], "success")
                feedback = [item for item in result["stages"] if item["stage"] == "09_feedback_learning"][0]
                self.assertTrue(feedback["details"]["skipped"])
        stages["09_feedback_learning.py"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
