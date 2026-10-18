from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.feature_engineering import FeatureEngineer
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.services.Anomaly_Health_Monitoring.context_modeling.context_drift import ContextDriftDetector
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import ResidualCalculator
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.anomaly_detection.condition_consensus import ConditionConsensusDetector
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics, condition_candidate_improves
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope, record_training_scope
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, atomic_write_json, atomic_write_csv, configure_logging

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261014_leakage_fix"


def staged_config(run):
    cfg = Config()
    for name in ("models", "processed", "outputs", "logs", "mlruns", "experiments", "metrics", "reports", "uploads", "data_exchange"):
        setattr(cfg.paths, name, run / "bundle" / name)
    cfg.paths.scalers = cfg.paths.models / "scalers"
    cfg.mlops.log_to_file = False
    cfg.mlops.enable_mlflow = False


    cfg.context.n_modes = 6
    for parameters in (cfg.twin.random_forest, cfg.twin.xgboost, cfg.twin.lightgbm):
        parameters["n_jobs"] = 4
    cfg.ensure_directories()
    configure_logging(cfg)
    return cfg


def report(path, value):
    atomic_write_json(value, path)


def stage(name):
    return importlib.import_module("app.services.Anomaly_Health_Monitoring.pipeline." + name)


def labelled_scores(fusion, residuals, context, values=None, threshold=.5):
    if values is None:
        result = fusion.fuse(fusion.score(residuals), fusion_threshold=threshold)
    else:
        result = residuals[["unit", "cycle", "split", "subset_id", "calibration_ready"]].copy()
        result["anomaly_score"] = values
        result["threshold_used"] = threshold
        result["anomaly_flag"] = (np.asarray(values) > threshold) & result.calibration_ready.to_numpy(bool)
    result = fusion.classify_type(result, context)
    result["RUL"] = residuals.RUL.to_numpy()
    return result


def measured(frame):
    result = condition_metrics(frame)
    result["accuracy"] = (result["true_positive"] + result["true_negative"]) / result["rows"]
    result["per_subset"] = {str(name): condition_metrics(part, bootstrap=False) for name, part in frame.groupby("subset_id")}
    return result


def development_gate(before, after, bound):
    return condition_candidate_improves(before, after, bound)


def select_development_margin(run, cfg, residuals, labelled):

    if (run / "evaluation.json").exists() or (run / "official_scores.csv").exists():
        raise AnomalyHealthError("Development calibration is closed after confirmation/official evaluation.")
    margins = (1., 1.25, 1.5, 2., 3., 4., 8., 16., 32.)
    report(run / "margin_protocol.json", {"margins": margins,
        "selection_role": "selection", "confirmation_and_official_closed": True,
        "objective": "Higher overall F1/recall/AUC/AP; preserve subset F1/recall and <=0.1 percentage point added healthy false alerts",
        "ranking_rule": "Highest eligible F1, then recall, then fewer false alerts, then wider margin",
        "correction": "The initial runner additionally required per-subset AUC/AP non-regression. This now matches the existing production policy and user priority; the rejected decision is archived unchanged."})
    previous_decision = run / "promotion_decision.json"
    if previous_decision.exists():
        archived = run / "development_attempts/margin1_rejected"
        archived.mkdir(parents=True, exist_ok=True)
        for source in (previous_decision, cfg.paths.models / ConditionConsensusDetector.ARTIFACT_NAME):
            destination = archived / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
    fusion = AnomalyFusion(cfg)
    fusion.load()
    detector = fusion.detectors["condition_consensus"]
    selection = training_scope(residuals, cfg, ("selection",), required=True).reset_index(drop=True)
    context = labelled.loc[labelled.unit.isin(selection.unit.unique())].reset_index(drop=True)
    baseline = labelled_scores(fusion, selection, context, detector.reference.score(selection), detector.reference.threshold)
    before = measured(baseline)
    original_score = detector.score(selection)
    original_margin = detector.margin
    candidates = []
    for margin in margins:

        score = original_score / (original_score + (margin / original_margin) * (1. - original_score))
        candidate = measured(labelled_scores(fusion, selection, context, score))
        candidates.append({"margin": margin, "metrics": candidate,
            "accepted": bool(development_gate(before, candidate, cfg.anomaly.condition_consensus["max_false_alarm_increase"]))})
    eligible = [candidate for candidate in candidates if candidate["accepted"]]
    report(run / "margin_search.json", {"baseline": before, "candidates": candidates})
    if not eligible:
        raise AnomalyHealthError("No development margin improves F1/recall within the subset false-alert bounds; production unchanged.")
    winner = max(eligible, key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], -item["metrics"]["false_positive"], item["margin"]))
    detector.margin = winner["margin"]
    actual = measured(labelled_scores(fusion, selection, context, detector.score(selection)))
    for key in ("f1", "recall", "false_alarm_rate", "auc", "average_precision"):
        if abs(actual[key] - winner["metrics"][key]) > 1e-12:
            raise AnomalyHealthError("Development calibration does not reproduce production scoring.")
    record_training_scope("consensus_margin_selection", selection, cfg, ("selection",))
    detector.provenance.update(margin_selection_unit_ids=sorted(map(int, selection.unit.unique())),
                               margin_selection="Global margin selected on development engines only, before confirmation/official scoring")
    detector.save()
    fusion_meta = json.loads((cfg.paths.models / "anomaly_fusion_meta.json").read_text())
    fusion_meta.update(margin=detector.margin, score_version="causal_condition_consensus_v2",
                       label_basis="RUL condition proxies; not verified fault labels")
    report(cfg.paths.models / "anomaly_fusion_meta.json", fusion_meta)


    fit = training_scope(residuals, cfg, required=True).reset_index(drop=True)
    fit_context = labelled.loc[labelled.unit.isin(fit.unit.unique())].reset_index(drop=True)
    fusion.detectors["condition_consensus"] = detector
    fusion.severity_classifier.calibrate(labelled_scores(fusion, fit, fit_context))
    decision = {"accepted_on_development": True, "baseline": before, "candidate": actual,
        "locked_before_confirmation": True, "locked_before_official": True,
        "selection_units": sorted(map(int, selection.unit.unique())), "threshold": .5, "margin": detector.margin,
        "condition_model_sha256": hashlib.sha256((cfg.paths.models / detector.ARTIFACT_NAME).read_bytes()).hexdigest()}
    report(previous_decision, decision)
    return decision


def score_raw(raw, cfg, context, scaler, ensemble, fusion):
    clean, _ = DataCleaner(cfg).clean(raw, fit_policy=False)
    labelled = context.assign(clean)
    scaled = scaler.transform(labelled)
    predictions = ensemble.predict(scaled)
    residual = ResidualCalculator(cfg).compute(scaled, predictions, recalibrate=False)
    return labelled_scores(fusion, residual, labelled), residual, scaled


def verify_provenance(cfg):
    plan = load_training_split(cfg)
    scope = json.loads((cfg.paths.models / "training_provenance.json").read_text())
    expected = {"cleaning", "context", "context_drift_reference", "scaling", "twin_random_forest", "twin_xgboost", "twin_lightgbm", "twin_tensorflow",
                "covariance_reference", "residual_standardization", "consensus_classifiers", "consensus_healthy_scale", "health_reference", "severity"}
    if not expected <= set(scope):
        raise AnomalyHealthError(f"Missing fitted-stage provenance: {sorted(expected - set(scope))}")
    for name, item in scope.items():
        allowed = set().union(*(set(plan["engine_roles"][role]) for role in item["roles"]))
        if item["plan_id"] != plan["plan_id"] or not set(item["units"]) <= allowed:
            raise AnomalyHealthError(f"Invalid fitted-stage provenance: {name}")
        if set(item["units"]) & set(plan["engine_roles"]["confirmation"]):
            raise AnomalyHealthError(f"Confirmation engine leaked into {name}")
    return {"passed": True, "plan_id": plan["plan_id"], "stages": scope,
            "engine_counts": {key: len(value) for key, value in plan["engine_roles"].items()}}


def refresh_current(cfg, raw):
    atomic_write_csv(raw, cfg.paths.processed / cfg.paths.raw_data_csv, config=cfg)
    cleaner = DataCleaner(cfg)
    clean, _ = cleaner.run(raw, fit_policy=False)
    engineered, _ = FeatureEngineer(cfg).run(clean, fit_policy=False)
    stage("02_context_modeling").run_stage(cfg, retrain=False)
    stage("03_digital_twin").run_stage(cfg, retrain=False)
    stage("04_health_monitoring").run_stage(cfg, recalibrate=False)
    stage("05_anomaly_detection").run_stage(cfg, retrain=False)
    stage("06_reasoning").run_stage(cfg)
    stage("07_explainability").run_stage(cfg, include_shap=True)
    stage("08_uncertainty").run_stage(cfg)
    from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_data_generator import DashboardDataGenerator
    from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_api import DashboardApiService
    dashboard, dashboard_report = DashboardDataGenerator(cfg).run()
    exchange = DashboardApiService(cfg).run()
    return {"rows": len(dashboard), "engines": int(dashboard.unit.nunique()), "dashboard": dashboard_report, "exchange": exchange}


def publish(run, cfg, decision):


    if not decision["accepted_on_development"]:
        raise AnomalyHealthError("Development performance gate rejected this bundle; production is unchanged.")
    model_path = cfg.paths.models / ConditionConsensusDetector.ARTIFACT_NAME
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != decision["condition_model_sha256"]:
        raise AnomalyHealthError("The selected model changed after the development decision was locked.")
    verify_provenance(cfg)
    active = Config()
    copies = []
    registry = active.paths.models / active.paths.model_registry_json
    new_registry = cfg.paths.models / cfg.paths.model_registry_json
    if new_registry.is_file():
        combined = json.loads(registry.read_text()) if registry.is_file() else {}
        updates = json.loads(new_registry.read_text())
        def relocated(value):
            if isinstance(value, dict):
                return {key: relocated(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocated(item) for item in value]
            return value.replace(str(cfg.paths.models), str(active.paths.models)) if isinstance(value, str) else value
        def merge(destination, source):
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(destination.get(key), dict):
                    merge(destination[key], value)
                else:
                    destination[key] = value
        merge(combined, relocated(updates))
        installed_registry = run / "publication_registry.json"
        report(installed_registry, combined)
        copies.append((installed_registry, registry))
    for name in ("models", "processed", "metrics", "data_exchange"):
        source = getattr(cfg.paths, name)
        destination = getattr(active.paths, name)
        for path in source.rglob("*"):
            if path.is_file() and path.name != "model_registry.json":
                copies.append((path, destination / path.relative_to(source)))
    backups = run / "before"
    for _, destination in copies:
        if destination.is_file():
            saved = backups / destination.relative_to(ROOT)
            saved.parent.mkdir(parents=True, exist_ok=True)
            if not saved.exists():
                shutil.copy2(destination, saved)


    installed = []
    try:
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".leakage-fix.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            saved = backups / destination.relative_to(ROOT)
            if saved.exists():
                shutil.copy2(saved, destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    return {"published": True, "files": len(installed), "backup_directory": str(backups),
            "decision_source": "development selection; locked before confirmation/official scoring"}


def main(args):
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    cfg = staged_config(run)
    active = Config()
    raw_current = pd.read_csv(active.paths.processed / active.paths.raw_data_csv)
    protocol_path = run / "protocol.json"
    if not protocol_path.exists():
        report(protocol_path, {"frozen_config": cfg.to_dict(), "training_inputs": "Original training text files only",
            "promotion_data": "selection engines only", "official_data_is_report_only": True,
            "label_basis": "healthy RUL >=125 versus critical RUL <=30, excluding commissioning",
            "evidence_limit": "All C-MAPSS populations were inspected previously; corrected fitting cannot undo past benchmark reuse.",
            "search": "One fixed recipe; no threshold/capacity search using confirmation or official outcomes"})
    else:
        saved_config = json.loads(protocol_path.read_text())["frozen_config"]
        if saved_config != cfg.to_dict():
            raise AnomalyHealthError("The frozen training protocol changed; use a new run directory.")
    prepared_path = run / "prepared.joblib"
    if not prepared_path.exists():
        raw_train = CMAPSSReader(cfg).read_subsets(split="train")
        atomic_write_csv(raw_train, cfg.paths.processed / cfg.paths.raw_data_csv, config=cfg)
        clean, _ = DataCleaner(cfg).run(raw_train, fit_policy=True)


        engineered = FeatureEngineer(cfg).add_cycle_features(clean)
        labelled, _ = OperatingModeDetector(cfg).run(engineered, refit=True)
        scaled, _ = PerModeScaler(cfg).run(labelled, refit=True)
        joblib.dump({"labelled": labelled, "scaled": scaled}, prepared_path, compress=3)
        print("PREPROCESSING READY: fitted only on shared fit engines", flush=True)
    prepared = joblib.load(prepared_path)
    labelled, scaled = prepared["labelled"], prepared["scaled"]
    if not (cfg.paths.models / ContextDriftDetector.REFERENCE_NAME).exists():
        ContextDriftDetector(cfg).fit_reference(labelled)
    residual_path = run / "training_residuals.joblib"
    reused_residual_cache = residual_path.exists()
    if not residual_path.exists():
        predictions, _ = EnsembleTwin(cfg).run(scaled, retrain=True)
        residuals, _ = ResidualCalculator(cfg).run(scaled, predictions, recalibrate=True)
        joblib.dump(residuals, residual_path, compress=3)
        print("CAUSAL TWINS REFITTED: no legacy artifacts reused", flush=True)
    residuals = joblib.load(residual_path)
    if not (cfg.paths.models / "health_index_calibration.json").exists():
        if reused_residual_cache:
            atomic_write_csv(residuals, cfg.paths.processed / cfg.paths.residuals_csv, config=cfg)
        stage("04_health_monitoring").run_stage(cfg, recalibrate=True)
    decision_path = run / "promotion_decision.json"
    if not decision_path.exists():
        fusion = AnomalyFusion(cfg)
        scored, _ = fusion.run(residuals, context=labelled, retrain=True)
        detector = fusion.detectors["condition_consensus"]
        detector.reference.save()
        baseline = labelled_scores(fusion, residuals, labelled, detector.reference.score(residuals), detector.reference.threshold)
        selection_units = load_training_split(cfg)["engine_roles"]["selection"]
        selected = scored.unit.isin(selection_units)
        candidate = scored.loc[selected].copy()
        candidate["RUL"] = residuals.loc[selected, "RUL"].to_numpy()
        before, after = measured(baseline.loc[selected]), measured(candidate)
        accepted = development_gate(before, after, cfg.anomaly.condition_consensus["max_false_alarm_increase"])
        decision = {"accepted_on_development": bool(accepted), "baseline": before, "candidate": after,
            "locked_before_confirmation": True, "locked_before_official": True,
            "selection_units": selection_units, "threshold": .5, "margin": detector.margin,
            "condition_model_sha256": hashlib.sha256((cfg.paths.models / detector.ARTIFACT_NAME).read_bytes()).hexdigest()}
        report(decision_path, decision)
    decision = json.loads(decision_path.read_text())
    if not decision["accepted_on_development"]:
        decision = select_development_margin(run, cfg, residuals, labelled)
    print("DEVELOPMENT DECISION LOCKED:", json.dumps({k: decision["candidate"][k] for k in ("f1", "recall", "false_alarm_rate")}), flush=True)
    fusion = AnomalyFusion(cfg)
    fusion.load()
    provenance = verify_provenance(cfg)
    report(run / "provenance_audit.json", provenance)
    context = OperatingModeDetector(cfg).load()
    scaler = PerModeScaler(cfg).load_scalers()
    ensemble = EnsembleTwin(cfg).load()
    evaluation_path = run / "evaluation.json"
    if not evaluation_path.exists():
        all_training = labelled_scores(fusion, residuals, labelled)
        held = all_training.loc[all_training.unit.isin(load_training_split(cfg)["engine_roles"]["confirmation"])]
        confirmation = measured(held)
        print("CONFIRMATION REPORT ONLY:", json.dumps({k: confirmation[k] for k in ("f1", "recall", "false_alarm_rate")}), flush=True)
        official_raw = CMAPSSReader(cfg).read_subsets(split="test")
        official, official_residuals, official_scaled = score_raw(official_raw, cfg, context, scaler, ensemble, fusion)
        official_metrics = measured(official)
        official.drop(columns="RUL").to_csv(run / "official_scores.csv", index=False)
        official_raw[["unit", "cycle", "RUL", "split", "subset_id"]].to_csv(run / "official_labels.csv", index=False)


        from scripts.Anomaly_Health_Monitoring import evaluate_holdout
        previous_argv = sys.argv
        try:
            sys.argv = ["evaluate_holdout", "--scores", str(run / "official_scores.csv"), "--labels", str(run / "official_labels.csv"),
                        "--out", str(cfg.paths.metrics / "holdout_anomaly_evaluation.json")]
            if evaluate_holdout.main() != 0:
                raise AnomalyHealthError("Could not save the descriptive benchmark evaluation.")
        finally:
            sys.argv = previous_argv
        metric_path = cfg.paths.metrics / "holdout_anomaly_evaluation.json"
        metric_artifact = json.loads(metric_path.read_text())
        metric_artifact["provenance"].update(evaluation_status="reused_benchmark_exploratory",
            training_plan_id=provenance["plan_id"], preprocessing_fitted_on_training_only=True,
            official_results_used_for_promotion=False)
        report(metric_path, metric_artifact)

        chosen = official_raw.loc[official_raw.unit.isin(official_raw.groupby("subset_id").unit.first().to_numpy())]
        full, _, _ = score_raw(chosen, cfg, context, scaler, ensemble, fusion)
        prefix_raw = chosen.loc[chosen.cycle.le(60)].copy()
        prefix, _, _ = score_raw(prefix_raw, cfg, context, scaler, ensemble, fusion)
        joined = prefix[["unit", "cycle", "anomaly_score"]].merge(full[["unit", "cycle", "anomaly_score"]], on=["unit", "cycle"], suffixes=("_prefix", "_full"))
        error = float(np.max(np.abs(joined.anomaly_score_prefix - joined.anomaly_score_full)))
        if error > 1e-10:
            raise AnomalyHealthError("New bundle failed end-to-end prefix causality.")
        evaluation = {"confirmation": confirmation, "official_reused_benchmark": official_metrics,
            "evaluation_status": "retrospective; previously inspected engines, not a blind external test",
            "prefix_score_max_error": error, "official_used_for_promotion": False,
            "preprocessing_scope_audit_passed": True}
        report(evaluation_path, evaluation)
        print("OFFICIAL REPORT ONLY:", json.dumps({k: official_metrics[k] for k in ("accuracy", "f1", "recall", "false_alarm_rate")}), flush=True)
    if not (run / "refresh.json").exists():
        report(run / "refresh.json", refresh_current(cfg, raw_current))
    if args.publish:
        report(run / "publication.json", publish(run, cfg, decision))
        print("PUBLISHED CLEAN BUNDLE", flush=True)
    print("COMPLETE", str(run), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--publish", action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
