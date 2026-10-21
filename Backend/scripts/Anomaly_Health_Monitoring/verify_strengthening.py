from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import HealthScoreEngine
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import HealthStateClassifier


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = Config()
    cfg.mlops.log_to_file = False
    cfg.mlops.enable_mlflow = False
    out = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261011_strengthening"
    old = out.with_name("run_20261010")
    report = json.loads((ROOT / "reports/anomaly_health_strengthening_20261011.json").read_text())
    expected_models = report["active_artifact_sha256"]
    assert all((cfg.paths.models / name).is_file() and digest(cfg.paths.models / name) == value
               for name, value in expected_models.items()), "Active artifact hash differs from the audited version."
    frames = joblib.load(old / "data.joblib")
    frame = frames["test"].reset_index(drop=True)
    residuals = frame[["unit", "cycle"]].copy()
    residuals[["residual_sensor_" + str(i) for i in range(1, 22)]] = joblib.load(old / "official_candidate.joblib")["raw_residuals"]
    residuals["calibration_ready"] = frame.cycle > 30
    fusion = AnomalyFusion(cfg)
    fusion.load()
    threshold = json.loads((cfg.paths.models / "residual_thresholds.json").read_text())["fusion_threshold"]
    raw = fusion.fuse(fusion.score(residuals), fusion_threshold=threshold)
    classified = fusion.classify_type(raw, frame)
    labelled = classified.merge(frame[["unit", "cycle", "RUL", "subset_id", "split"]], on=["unit", "cycle"], validate="one_to_one")
    metrics = condition_metrics(labelled)
    expected = report["active_official_anomaly_after_operational_gate"]
    for key in ("auc", "average_precision", "recall", "precision", "f1", "false_alarm_rate"):
        assert abs(metrics[key] - expected[key]) < 1e-12, (key, metrics[key], expected[key])
    assert metrics["confidence_intervals"] == expected["confidence_intervals"]
    health = HealthIndexCalculator(cfg).compute(residuals)
    from scipy.stats import spearmanr
    mask = frame.cycle > 30
    rho = float(spearmanr(health.loc[mask, "health_index"], frame.loc[mask, "RUL"]).statistic)
    assert abs(rho - report["active_official_health"]["post_commissioning_rul_correlation"]) < 1e-12
    cached = pd.read_csv(cfg.paths.processed / "residuals.csv")
    reconstructed = HealthIndexCalculator(cfg).compute(cached)
    saved = pd.read_csv(cfg.paths.processed / "health_index.csv")
    joined = reconstructed.merge(saved, on=["unit", "cycle"], suffixes=("_new", "_saved"), validate="one_to_one")
    error = float(np.max(np.abs(joined.health_index_new - joined.health_index_saved)))
    assert error <= 1e-5, error
    scored = HealthScoreEngine(cfg).compute(reconstructed)
    classified_health = HealthStateClassifier(cfg).classify(scored)
    stored_states = pd.read_csv(cfg.paths.processed / "health_states.csv")
    states = classified_health.merge(stored_states[["unit", "cycle", "health_score", "health_state"]],
                                     on=["unit", "cycle"], suffixes=("_new", "_saved"), validate="one_to_one")
    score_error = float(np.max(np.abs(states.health_score_new - states.health_score_saved)))
    assert score_error <= 1e-5, score_error
    assert states.health_state_new.eq(states.health_state_saved).all()
    assert all(digest(cfg.paths.models / name) == value for name, value in expected_models.items())
    result = {"passed": True, "official_test_metrics_reproduced": True,
              "confidence_intervals_reproduced": True, "health_correlation_reproduced": True,
              "dashboard_health_rows_verified": len(joined), "max_health_index_error": error,
              "max_health_score_error": score_error, "health_states_reproduced": True,
              "active_artifacts_unchanged": True}
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
