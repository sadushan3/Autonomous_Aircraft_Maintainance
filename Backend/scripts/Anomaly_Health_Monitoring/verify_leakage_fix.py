from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.retrain_without_leakage import (
    DEFAULT_RUN, staged_config, score_raw, verify_provenance, development_gate,
)
from scripts.Anomaly_Health_Monitoring import evaluate_holdout
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.anomaly_detection.condition_consensus import ConditionConsensusDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import HealthScoreEngine
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_health import HealthEvaluator
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_explainability import ExplainabilityEvaluator
from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, load_training_split
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def main(args):
    run = Path(args.run_dir).resolve()
    cfg = staged_config(run)
    before = hashes(cfg.paths.models)
    plan = load_training_split(cfg)
    decision = json.loads((run / "promotion_decision.json").read_text())
    assert decision["accepted_on_development"] and development_gate(decision["baseline"], decision["candidate"], .001)
    model = ConditionConsensusDetector(cfg).load()
    assert model.margin == decision["margin"]
    assert hashlib.sha256((cfg.paths.models / model.ARTIFACT_NAME).read_bytes()).hexdigest() == decision["condition_model_sha256"]
    result = {"shared_scope": verify_provenance(cfg), "development_gate_reproduced": True}

    prepared = joblib.load(run / "prepared.joblib")
    fitting = training_scope(prepared["labelled"], cfg, required=True)
    context = OperatingModeDetector(cfg).load()
    assert int(context.model.scaler.n_samples_seen_) == len(fitting)
    np.testing.assert_allclose(context.model.scaler.mean_, fitting[["op_setting_1", "op_setting_2", "op_setting_3"]].mean().to_numpy(), rtol=0, atol=1e-12)
    scaler = PerModeScaler(cfg).load_scalers()
    for mode, estimator in scaler.scalers.items():
        values = fitting.loc[fitting.operating_mode.eq(mode), scaler.columns].to_numpy(float)
        assert int(estimator.n_samples_seen_) == len(values)
        np.testing.assert_allclose(estimator.mean_, values.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(estimator.var_, values.var(axis=0), rtol=1e-12, atol=1e-12)
    result["preprocessing_reconstructed"] = {"fit_rows": len(fitting), "fit_engines": fitting.unit.nunique(),
        "saved_scaler_samples": int(sum(estimator.n_samples_seen_ for estimator in scaler.scalers.values())),
        "confirmation_engines_used_for_fitting": 0}

    residuals = training_scope(joblib.load(run / "training_residuals.joblib"), cfg, required=True)
    healthy = (residuals.RUL >= cfg.healthy_region.rul_threshold) & (residuals.cycle > cfg.twin.commissioning_cycles)
    values = model.reference._feature_values(residuals)[healthy.to_numpy()]
    rebuilt = LedoitWolf().fit(values)
    np.testing.assert_allclose(rebuilt.location_, model.reference.covariance_estimator.location_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.covariance_, model.reference.covariance_estimator.covariance_, rtol=0, atol=1e-12)
    result["covariance_reconstructed"] = {"healthy_fit_rows": len(values), "fit_engines": residuals.unit.nunique(),
        "cutoff": model.reference.threshold, "normalization": model.reference.score_normalization}
    for name in ("random_forest", "xgboost", "lightgbm", "tensorflow"):
        meta = json.loads((cfg.paths.models / f"{name}_twin_meta.json").read_text())
        assert meta["metrics"]["training_plan_id"] == plan["plan_id"]
        assert meta["metrics"]["age_reference_cycles"] == 300
    assert model.provenance["fit_unit_ids"] == plan["engine_roles"]["fit"]
    assert model.provenance["calibration_unit_ids"] == plan["engine_roles"]["calibration"]

    stored = json.loads((cfg.paths.metrics / "holdout_anomaly_evaluation.json").read_text())
    joined = evaluate_holdout.load_frames(run / "official_scores.csv", run / "official_labels.csv")
    replay = evaluate_holdout.build(joined, 125, 30)
    assert not evaluate_holdout.compare(stored["metrics"], replay, 0)
    result["stored_evaluation_replay_passed"] = True
    print("Fitting populations and stored benchmark replay verified", flush=True)

    if args.refresh_health:


        cfg.twin.startup_profile_enabled = False
        raw = CMAPSSReader(cfg).read_subsets(split="test")
        fusion = AnomalyFusion(cfg)
        fusion.load()
        ensemble = EnsembleTwin(cfg).load()
        observed, official_residuals, _ = score_raw(raw, cfg, context, scaler, ensemble, fusion)
        compared = observed[["unit", "cycle", "anomaly_score", "anomaly_type"]].merge(
            joined[["unit", "cycle", "anomaly_score", "anomaly_type"]], on=["unit", "cycle"], suffixes=("_observed", "_locked"), validate="one_to_one")
        error = float(np.max(np.abs(compared.anomaly_score_observed - compared.anomaly_score_locked)))
        np.testing.assert_allclose(compared.anomaly_score_observed, compared.anomaly_score_locked, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(compared.anomaly_type_observed, compared.anomaly_type_locked)
        result["reloaded_serving_replay"] = {"rows": len(compared), "max_score_error": error, "fault_decisions_identical": True}
        health = HealthScoreEngine(cfg).compute(HealthIndexCalculator(cfg).compute(official_residuals, recalibrate=False))
        extras = [name for name in ("RUL", "split", "subset_id") if name not in health]
        if extras:
            health = health.merge(raw[["unit", "cycle"] + extras], on=["unit", "cycle"], validate="one_to_one")
        report = HealthEvaluator(cfg).evaluate(health)
        report.update(evaluation_scope="Reused official C-MAPSS test trajectories; retrospective condition-proxy evidence",
            evaluation_status="reused_benchmark_exploratory", training_plan_id=plan["plan_id"],
            health_index_mean=float(health.health_index.mean()), health_index_std=float(health.health_index.std()),
            per_subset={str(name): HealthEvaluator(cfg).evaluate(part) for name, part in health.groupby("subset_id")},
            official_results_used_for_promotion=False)
        atomic_write_json(report, cfg.paths.metrics / "holdout_test_evaluation.json")
        result["health_evaluation_refreshed"] = True
        print("All official predictions reproduced; descriptive health evaluation refreshed", flush=True)
    result["current_explanation_checks"] = ExplainabilityEvaluator(cfg).run()
    assert before == hashes(cfg.paths.models), "Verification changed a model artifact"
    result["model_artifacts_unchanged_during_verification"] = True
    result["model_hashes"] = before
    result["evidence_limit"] = "Corrected fitting and causality checks do not erase historical benchmark reuse or establish independent fault-onset accuracy."
    atomic_write_json(result, run / "verification.json")
    print("VERIFICATION COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--refresh-health", action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
