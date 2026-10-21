import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config
from app.routers.Anomaly_Health_Monitoring import routes
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError
from scripts.Anomaly_Health_Monitoring.evaluate_holdout import split_metrics


WEIGHTS = {"isolation_forest": .25, "mahalanobis": .25, "lstm_autoencoder": .25, "residual_rule": .25}


class FourDetectorContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config()
        self.cfg.paths.models = Path(self.temp.name) / "models"
        self.cfg.paths.metrics = Path(self.temp.name) / "metrics"
        self.cfg.paths.models.mkdir()
        self.cfg.paths.metrics.mkdir()
        self.cfg.mlops.enable_mlflow = False
        self.cfg.mlops.log_to_file = False
        self.cfg.anomaly.fusion_weights = dict(WEIGHTS)

    def test_default_configuration_enables_the_four_frontend_models(self):
        cfg = Config()
        self.assertEqual(cfg.anomaly.fusion_weights, WEIGHTS)
        self.assertEqual(set(AnomalyFusion(cfg).detectors), set(WEIGHTS))

    def test_failed_member_cannot_publish_a_partial_bundle(self):
        fusion = AnomalyFusion(self.cfg)
        frame = pd.DataFrame({"unit": np.repeat([1, 2, 3, 4], 4), "cycle": list(range(1, 5)) * 4,
                              "residual_sensor_1": .1, "RUL": 200})
        previous = self.cfg.paths.models / "anomaly_fusion_meta.json"
        previous.write_text('{"previous_bundle":true}')
        with ExitStack() as stack:
            for name, detector in fusion.detectors.items():
                stack.enter_context(patch.object(detector, "fit", side_effect=ValueError("broken model")
                                                if name == "lstm_autoencoder" else None))
                stack.enter_context(patch.object(detector, "save"))
            calibrate = stack.enter_context(patch.object(fusion, "_calibrate_fusion_threshold"))
            with self.assertRaisesRegex(AnomalyHealthError, "lstm_autoencoder"):
                fusion.train(frame)
            calibrate.assert_not_called()
        self.assertEqual(json.loads(previous.read_text()), {"previous_bundle": True})

    def test_all_four_scores_contribute_to_fusion(self):
        fusion = AnomalyFusion(self.cfg)
        self.assertEqual(set(fusion.detectors), set(WEIGHTS))
        scores = pd.DataFrame({"unit": [1], "cycle": [31], "calibration_ready": [True],
                               "isolation_forest_score": [.8], "mahalanobis_score": [.4],
                               "lstm_ae_score": [.6], "residual_rule_score": [.2]})
        result = fusion.fuse(scores, fusion_threshold=.7)
        self.assertAlmostEqual(result.anomaly_score.iloc[0], .5)
        self.assertEqual(result.detectors_available.iloc[0], 4)
        self.assertEqual(result.detector_coverage.iloc[0], 1.)
        for column in ("isolation_forest_score", "mahalanobis_score", "lstm_ae_score", "residual_rule_score"):
            with self.subTest(column=column), self.assertRaises(AnomalyHealthError):
                fusion.fuse(scores.drop(columns=column))

    def test_weights_cannot_change_without_recalibration(self):
        path = self.cfg.paths.models / "anomaly_fusion_meta.json"
        path.write_text(json.dumps({"fusion_weights": {"mahalanobis": 1.}, "calibrated_fusion_threshold": .5}))
        with self.assertRaisesRegex(AnomalyHealthError, "recalibrate"):
            AnomalyFusion(self.cfg).load()

    def test_api_describes_saved_models_and_cutoff(self):
        models = {name: {"threshold": .99} for name in WEIGHTS}
        meta = {"fusion_weights": WEIGHTS, "detectors": models, "calibrated_fusion_threshold": .83}
        (self.cfg.paths.models / "anomaly_fusion_meta.json").write_text(json.dumps(meta))
        with patch.object(routes, "config", self.cfg):
            result = routes.detector_config()
        self.assertEqual(result["fusion"]["active_detectors"], 4)
        self.assertEqual(result["fusion"]["calibrated_threshold"], .83)
        self.assertEqual(len(result["detectors"]), 4)
        self.assertTrue(all(d["enabled"] and d["trained"] and d["weight"] == .25 for d in result["detectors"]))

    def test_severity_chart_summaries_cover_only_ready_rows(self):
        frame = pd.DataFrame({"unit": [1] * 5 + [2] * 5, "cycle": list(range(1, 6)) * 2,
                              "RUL": [200, 150, 100, 25, 10] * 2,
                              "anomaly_score": [.1, .2, .4, .8, .95] * 2,
                              "anomaly_flag": [False, False, False, True, True] * 2,
                              "threshold_used": .7, "severity": ["none", "none", "none", "low", "critical"] * 2,
                              "calibration_ready": [False, True, True, True, True] * 2})
        result = split_metrics(frame, 125, 30)
        self.assertEqual(sum(result["severity_row_counts"].values()), result["rows"])
        self.assertEqual(result["severity_row_counts"], {"none": 4, "low": 2, "critical": 2})
        self.assertEqual(result["severity_mean_rul"]["none"], 125.)
        self.assertEqual(result["severity_mean_rul"]["critical"], 10.)
        self.assertNotIn("medium", result["severity_mean_rul"])


if __name__ == "__main__":
    unittest.main()
