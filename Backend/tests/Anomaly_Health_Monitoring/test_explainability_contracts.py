"""Faithfulness, failure reporting and evidence regression checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import RandomForestTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.explainability.shap_explainer import ShapExplainer
from app.services.Anomaly_Health_Monitoring.explainability.monitoring_reference_explainer import MonitoringReferenceExplainer
from app.services.Anomaly_Health_Monitoring.explainability.explanation_generator import ExplanationGenerator
from app.services.Anomaly_Health_Monitoring.explainability.health_explainer import HealthExplainer
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import HealthScoreEngine
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import HealthStateClassifier
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_explainability import ExplainabilityEvaluator
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_service import DashboardService
from app.schemas.Anomaly_Health_Monitoring.explainability_schema import ExplanationRecord
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json


class ContextModel:
    def __init__(self, multiplier=1.):
        self.multiplier = multiplier

    def predict(self, x):
        # These assertions detect invalid independent perturbations of mode
        # indicators/settings or of the two representations of cycle age.
        np.testing.assert_allclose(x[:, 3] / 300., x[:, 4])
        np.testing.assert_allclose(x[:, 5] + x[:, 6], 1.)
        np.testing.assert_allclose(x[:, 0], x[:, 6])
        return self.multiplier * np.column_stack([x[:, 3] * (.1 + x[:, 0]), x[:, 0] - .2 * x[:, 3]])


class ExplainabilityContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Config()
        for name, value in vars(self.config.paths).items():
            if isinstance(value, Path):
                setattr(self.config.paths, name, Path(directory.name) / name)
        self.config.mlops.log_to_file = False
        self.config.mlops.enable_mlflow = False
        self.config.twin.commissioning_cycles = 3
        self.config.explainability.reference_background_rows = 6
        self.config.explainability.local_shap_max_rows = 3
        self.frame = pd.DataFrame({
            "unit": [1] * 6 + [2] * 6, "cycle": list(range(1, 7)) * 2,
            "operating_mode": [0, 1, 0, 1, 0, 1] * 2,
            "op_setting_1": [0, 1, 0, 1, 0, 1] * 2,
            "op_setting_2": 0., "op_setting_3": 0.,
            "sensor_2": np.arange(12, dtype=float) + 3., "sensor_3": np.arange(12, dtype=float) - 2.,
        })
        twins = []
        for i, factor in enumerate((1., 2.)):
            t = RandomForestTwin(self.config)
            t.name = f"twin_{i}"
            t.target_names = ["sensor_2", "sensor_3"]
            t.n_modes = 2
            t.model = ContextModel(factor)
            twins.append(t)
        self.ensemble = EnsembleTwin(self.config, twins=twins)
        self.ensemble.target_names = ["sensor_2", "sensor_3"]
        self.ensemble.weights = {"twin_0": .3, "twin_1": .7}
        self.predictions = self.ensemble.predict(self.frame)
        self.events = pd.DataFrame({"unit": [1, 2], "cycle": [4, 5], "shap_sensor": ["sensor_2", "sensor_3"]})
        self.residuals = self.frame[["unit", "cycle"]].copy()
        for sensor in self.ensemble.target_names:
            self.residuals[f"residual_{sensor}"] = self.frame[sensor] - self.predictions[f"expected_{sensor}"]
            self.residuals[f"residual_z_{sensor}"] = self.residuals[f"residual_{sensor}"]
        self.root = self.events.drop(columns="shap_sensor").assign(
            root_cause="Possible compressor issue", severity="low", anomaly_type="fault",
            evolution_pattern="gradual", onset_cycle=2, duration_cycles=3, trend_slope=.03,
            cause_likelihood=.6, recommended_action="Inspect sensors")
        ranker = ExplanationGenerator(self.config).ranker
        self.root["top_sensors"] = [str([v["sensor"] for v in ranker.rank_row(r, self.ensemble.target_names)])
                                    for _, r in self.root.merge(self.residuals, on=["unit", "cycle"]).iterrows()]
        self.anomalies = self.root[["unit", "cycle"]].assign(anomaly_score=.8, threshold_used=.5)
        self.evaluator = ExplainabilityEvaluator(self.config)

    def explain(self, frame=None, predictions=None):
        return MonitoringReferenceExplainer(self.ensemble, self.config).explain(
            self.frame if frame is None else frame, self.events,
            self.predictions if predictions is None else predictions)

    def test_missing_columns_and_nulls_are_incomplete(self):
        result = self.evaluator._completeness(pd.DataFrame({"why": [None]}))
        self.assertEqual(result["completeness_fraction"], 0.)
        self.assertEqual(len(result["missing_columns"]), 3)

    def test_empty_sensor_lists_are_incomplete(self):
        data = pd.DataFrame({"why": ["why"], "which_sensors": ["[]"], "what_pattern": ["p"], "how_severe": ["low"]})
        self.assertEqual(self.evaluator._completeness(data)["completeness_fraction"], 0.)

    def test_invalid_lists_never_agree_by_accident(self):
        data = pd.DataFrame({"unit": [1], "cycle": [1], "which_sensors": ["invalid"]})
        roots = pd.DataFrame({"unit": [1], "cycle": [1], "top_sensors": ["invalid"]})
        self.assertEqual(self.evaluator._consistency(data, roots)["consistency_fraction"], 0.)

    def test_unmatched_rows_count_as_inconsistent(self):
        data = pd.DataFrame({"unit": [1, 2], "cycle": [1, 1], "which_sensors": ["['sensor_2']"] * 2})
        roots = pd.DataFrame({"unit": [1], "cycle": [1], "top_sensors": ["['sensor_2']"]})
        result = self.evaluator._consistency(data, roots)
        self.assertEqual(result["consistency_fraction"], .5)
        self.assertEqual(result["unmatched_rows"], 1)

    def test_duplicate_identities_fail_consistency(self):
        data = pd.DataFrame({"unit": [1, 1], "cycle": [1, 1], "which_sensors": ["['sensor_2']"] * 2})
        self.assertEqual(self.evaluator._consistency(data, data.rename(columns={"which_sensors": "top_sensors"}))["consistency_fraction"], 0.)

    def test_legacy_rows_do_not_prove_shap_ran(self):
        atomic_write_json({"rows_explained": 999, "shap_computed": True}, self.config.paths.metrics / "explainability.json")
        data = pd.DataFrame({"unit": [1], "cycle": [1], "why": ["x"], "which_sensors": ["['sensor_2']"],
                             "what_pattern": ["x"], "how_severe": ["low"]})
        report = self.evaluator.evaluate(data, data.rename(columns={"which_sensors": "top_sensors"}))
        self.assertFalse(report["shap_computed"])
        self.assertFalse(report["acceptance"]["automated_checks_passed"])

    def test_permutation_fallback_reconstructs_a_linear_prediction(self):
        explainer = ShapExplainer(self.config)
        self.config.explainability.permutation_max_evals = 1
        x = np.arange(24, dtype=float).reshape(8, 3)
        predict = lambda a: a @ np.array([1., -2., .3])
        values = explainer.explain_generic_model(predict, x[:2], x)
        np.testing.assert_allclose(explainer.base_values + values.sum(axis=1), predict(x[:2]), atol=1e-8)

    def test_multioutput_tree_explains_requested_target(self):
        x = np.arange(30, dtype=float).reshape(10, 3)
        y = np.column_stack([np.zeros(10), x[:, 0] ** 2])
        model = RandomForestRegressor(n_estimators=4, random_state=0).fit(x, y)
        explainer = ShapExplainer(self.config)
        values = explainer.explain_tree_model(model, x[:2], ["a", "b", "c"], output_index=1)
        np.testing.assert_allclose(explainer.base_values + values.sum(axis=1), model.predict(x[:2])[:, 1], atol=1e-5)

    def test_legacy_tree_list_uses_requested_target(self):
        class LegacyExplainer:
            expected_value = np.array([1., 2.])
            def shap_values(self, x):
                return [np.zeros_like(x), np.ones_like(x)]
        with patch('shap.TreeExplainer', return_value=LegacyExplainer()):
            explainer = ShapExplainer(self.config)
            result = explainer.explain_tree_model(None, np.zeros((2, 3)), ["a", "b", "c"], 1)
        np.testing.assert_array_equal(result, np.ones((2, 3)))
        np.testing.assert_array_equal(explainer.base_values, [2., 2.])

    def test_local_group_shap_reconstructs_calibrated_ensemble(self):
        records, report = self.explain()
        self.assertEqual(report["explainer_used"], "exact_grouped")
        self.assertEqual(len(records), 2)
        for record in records.values():
            reconstructed = record["shap_base_value"] + sum(v["shap_value"] for v in record["shap_top_features"])
            self.assertAlmostEqual(reconstructed, record["shap_prediction"], places=10)
            self.assertLess(record["shap_reference_error"], 1e-10)
        self.assertNotEqual(records[(1, 4)]["shap_top_features"], records[(2, 5)]["shap_top_features"])

    def test_stale_predictions_are_rejected(self):
        stale = self.predictions.copy()
        stale.loc[stale.cycle == 4, "expected_sensor_2"] += 1.
        with self.assertRaisesRegex(Exception, "do not match"):
            self.explain(predictions=stale)

    def test_future_sensor_values_do_not_change_local_shap(self):
        before, _ = self.explain()
        modified = self.frame.copy()
        modified.loc[modified.cycle > 5, ["sensor_2", "sensor_3"]] = 99999.
        after, _ = self.explain(frame=modified)
        self.assertEqual(before, after)

    def test_generator_reports_disabled_shap(self):
        generator = ExplanationGenerator(self.config)
        result = generator.generate(self.root, self.residuals, include_shap=False)
        self.assertTrue(result.shap_status.eq("disabled").all())
        self.assertFalse(generator.last_report["shap_computed"])

    def test_generator_reports_failed_shap_without_losing_narratives(self):
        generator = ExplanationGenerator(self.config)
        result = generator.generate(self.root, self.residuals, include_shap=True)
        self.assertEqual(len(result), len(self.root))
        self.assertTrue(result.shap_status.eq("failed").all())
        self.assertTrue(generator.last_report["shap_errors"])

    def test_generator_honours_engine_and_row_limits(self):
        generator = ExplanationGenerator(self.config)
        result = generator.generate(self.root, self.residuals, False, engine_id=2, max_rows=1)
        self.assertEqual(result.unit.tolist(), [2])

    def calibrated_health(self):
        calculator = HealthIndexCalculator(self.config)
        calibration = {"informative_columns": ["residual_sensor_2", "residual_sensor_3"],
                       "healthy_reference_deviation": .5,
                       "sensor_weights": {"sensor_2": 2., "sensor_3": 1.}}
        atomic_write_json(calibration, calculator.calibration_path())
        return calculator.compute(self.residuals)

    def test_health_evidence_uses_actual_weights_and_smoothing(self):
        health = self.calibrated_health()
        result = HealthExplainer(self.config).explain(self.residuals, health)
        self.assertEqual(len(result), len(self.residuals))
        for value in result.values():
            evidence = value["health_evidence"]
            self.assertEqual(value["health_explanation_status"], "computed")
            self.assertAlmostEqual(evidence["current_cycle_term"] + evidence["history_term"], evidence["health_index"])
            self.assertAlmostEqual(sum(s["weighted_deviation"] for s in evidence["top_sensors"]), evidence["weighted_deviation"])
            self.assertAlmostEqual(sum(s["weight"] for s in evidence["top_sensors"]), 1.)

    def test_stale_health_scores_are_identified(self):
        health = self.calibrated_health()
        health.loc[0, "health_index"] += 1.
        result = HealthExplainer(self.config).explain(self.residuals, health)
        self.assertEqual(result[(1, 1)]["health_explanation_status"], "reference_mismatch")

    def test_health_state_explanation_uses_persisted_thresholds(self):
        health = self.calibrated_health()
        classifier = HealthStateClassifier(self.config)
        classifier.thresholds = {"healthy": 90., "degrading": 50., "warning": 25., "critical": 10.}
        classifier.save_thresholds()
        states = classifier.classify(HealthScoreEngine(self.config).compute(health))
        result = HealthExplainer(self.config).explain(self.residuals, states)
        for value in result.values():
            self.assertEqual(value["health_explanation_status"], "computed")
            evidence = value["health_evidence"]
            self.assertAlmostEqual(evidence["health_index"] + evidence["health_score_trend_adjustment"], evidence["health_score"])
            self.assertEqual(classifier._classify_scalar(evidence["health_score"]), evidence["health_state"])
            self.assertTrue(evidence["state_matches"])

    def test_stale_health_state_is_rejected(self):
        states = HealthScoreEngine(self.config).compute(self.calibrated_health())
        states["health_state"] = "invalid state"
        result = HealthExplainer(self.config).explain(self.residuals, states)
        self.assertTrue(all(v["health_explanation_status"] == "reference_mismatch" for v in result.values()))

    def test_end_to_end_numeric_evidence_and_api_serialization(self):
        health = self.calibrated_health()
        generator = ExplanationGenerator(self.config)
        result = generator.generate(self.root, self.residuals, True, scaled=self.frame,
                                    predictions=self.predictions, anomalies=self.anomalies,
                                    health=health, ensemble=self.ensemble)
        report = self.evaluator.evaluate(result, self.root)
        self.assertTrue(report["acceptance"]["automated_checks_passed"], report)
        payload = DashboardService._sanitize(result.iloc[0].dropna().to_dict())
        payload["engine_id"] = payload.pop("unit")
        record = ExplanationRecord(**payload)
        self.assertTrue(record.shap_top_features)
        self.assertTrue(record.health_evidence)
        self.assertTrue(record.sensor_evidence)
        self.assertIn("threshold", record.why)
        self.assertIn("mode-scaled", record.explanation_text)

    def test_additivity_tampering_fails_evaluation(self):
        generator = ExplanationGenerator(self.config)
        result = generator.generate(self.root, self.residuals, True, scaled=self.frame,
                                    predictions=self.predictions, anomalies=self.anomalies,
                                    health=self.calibrated_health(), ensemble=self.ensemble)
        result.loc[0, "shap_base_value"] += 10.
        report = self.evaluator.evaluate(result, self.root)
        self.assertFalse(report["acceptance"]["shap_valid_for_attempted_rows"])


if __name__ == '__main__':
    unittest.main()
