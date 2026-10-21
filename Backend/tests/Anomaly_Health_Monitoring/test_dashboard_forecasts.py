import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config
from app.routers.Anomaly_Health_Monitoring import routes
from app.services.Anomaly_Health_Monitoring.dashboard.forecast_metrics import active_forecast_metrics


class ForecastMetricsTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {"name": "active_forecast"}
        self.metrics = {"r2": .66, "rmse": .42, "mae": .24, "rows": 83,
                        "engines": 6, "n_informative_targets": 17,
                        "per_subset": {"FD001": {"r2": .58, "rmse": .32, "mae": .19}}}
        self.training = {"weighted_mean_rmse": .9, "n_sensors_modelled": 21,
                         "startup_profile": {"active": True, "recipe": "active_forecast",
                                             "official_full_trajectory_metrics": self.metrics,
                                             "official_reused_benchmark_metrics": {"r2": .99}}}

    def test_reports_active_full_trajectory_metrics(self):
        result = active_forecast_metrics(self.training, {}, self.recipe)
        self.assertTrue(result["available"])
        for key in ("r2", "rmse", "mae", "rows", "engines"):
            self.assertEqual(result[key], self.metrics[key])
        self.assertEqual(result["per_subset"]["FD001"]["r2"], .58)
        self.assertEqual(result["informative_targets"], 17)
        self.assertEqual(result["target_channels"], 21)
        self.assertEqual(result["evaluation_status"], "retrospective_reused_benchmark")

    def test_old_or_inactive_metrics_are_unavailable(self):
        for field, value in (("active", False), ("recipe", "old_forecast"),
                             ("official_full_trajectory_metrics", {})):
            training = copy.deepcopy(self.training)
            training["startup_profile"][field] = value
            with self.subTest(field=field):
                result = active_forecast_metrics(training, {}, self.recipe)
                self.assertFalse(result["available"])
                self.assertNotIn("r2", result)
                self.assertNotIn("rmse", result)

    def test_requires_recipe_and_finite_scores(self):
        self.assertFalse(active_forecast_metrics(self.training, {}, {})["available"])
        for value in (None, float("nan"), float("inf"), "0.8"):
            training = copy.deepcopy(self.training)
            training["startup_profile"]["official_full_trajectory_metrics"]["r2"] = value
            with self.subTest(value=value):
                self.assertFalse(active_forecast_metrics(training, {}, self.recipe)["available"])

    def test_preserves_zero_error_and_negative_r2(self):
        self.metrics.update(r2=-.1, rmse=0., mae=0.)
        result = active_forecast_metrics(self.training, {}, self.recipe)
        self.assertTrue(result["available"])
        self.assertEqual((result["r2"], result["rmse"], result["mae"]), (-.1, 0., 0.))

    def test_validation_metadata_can_supply_active_metrics(self):
        self.assertTrue(active_forecast_metrics({}, self.training, self.recipe)["available"])


class TwinDashboardRouteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = Config()
        for key in ("processed", "metrics", "reports", "models"):
            path = Path(directory.name) / key
            path.mkdir()
            setattr(self.config.paths, key, path)
        patcher = patch.object(routes, "config", self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        routes._twin_sensor_frame.cache_clear()
        self.addCleanup(routes._twin_sensor_frame.cache_clear)

    def write_trace(self, ready=True, reference=True, residual=True):
        predictions = {"unit": [1, 1], "cycle": [1, 2],
                       "expected_sensor_11": [13., 13.],
                       "twin_spread_sensor_11": [.5, .5],
                       "startup_profile_ready": [False, ready]}
        if reference:
            predictions["reference_expected_sensor_11"] = [10., 10.]
        pd.DataFrame(predictions).to_csv(self.config.paths.processed / "twin_predictions.csv", index=False)
        if residual:
            pd.DataFrame({"unit": [1, 1], "cycle": [1, 2],
                          "residual_sensor_11": [5., 5.],
                          "prediction_residual_sensor_11": [2., 2.]}).to_csv(
                self.config.paths.processed / "residuals.csv", index=False)

    def trace(self):
        return routes.twin_trace(engine_id=1, sensor="sensor_11", max_points=400)

    def test_separates_forecast_from_monitoring_reference(self):
        self.write_trace()
        result = self.trace()
        first, last = result["points"]
        self.assertIsNone(first["forecast"])
        self.assertEqual(last["actual"], 15.)
        self.assertEqual(last["forecast"], 13.)
        self.assertEqual(last["expected"], 10.)
        self.assertEqual(last["actual"] - last["forecast"], 2.)
        self.assertEqual(last["actual"] - last["expected"], last["residual"])
        self.assertEqual((last["band_low"], last["band_high"]), (9.5, 10.5))

    def test_legacy_trace_keeps_reference_and_has_no_forecast(self):
        self.write_trace(reference=False)
        last = self.trace()["points"][-1]
        self.assertIsNone(last["forecast"])
        self.assertEqual(last["expected"], 13.)
        self.assertEqual(last["actual"], 18.)

    def test_missing_residuals_do_not_invent_actual_or_forecast(self):
        self.write_trace(residual=False)
        last = self.trace()["points"][-1]
        self.assertIsNone(last["actual"])
        self.assertIsNone(last["forecast"])
        self.assertEqual(last["expected"], 10.)

    def test_metrics_route_publishes_active_forecast(self):
        training = {"startup_profile": {"active": True, "recipe": "current",
                     "official_full_trajectory_metrics": {"r2": .66, "rmse": .42, "mae": .24}},
                    "weighted_mean_rmse": .9}
        (self.config.paths.metrics / "digital_twin.json").write_text(json.dumps(training))
        (self.config.paths.models / "startup_profile_recipe.json").write_text(json.dumps({"name": "current"}))
        result = routes.twin_metrics()
        self.assertTrue(result["forecast"]["available"])
        self.assertEqual(result["forecast"]["r2"], .66)
        self.assertEqual(result["forecast"]["rmse"], .42)
        (self.config.paths.models / "startup_profile_recipe.json").write_text(json.dumps({"name": "replacement"}))
        self.assertFalse(routes.twin_metrics()["forecast"]["available"])


if __name__ == "__main__":
    unittest.main()
