import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, configure_logging

WEIGHTS = {"isolation_forest": .25, "mahalanobis": .25, "lstm_autoencoder": .25, "residual_rule": .25}
DEFAULT_RUN = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261018_four_detectors"
TRAINING = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261014_leakage_fix"


def staged_config(run):
    cfg = Config()
    for name in ("models", "processed", "outputs", "logs", "mlruns", "experiments", "metrics", "reports", "uploads", "data_exchange"):
        setattr(cfg.paths, name, run / "bundle" / name)
    cfg.paths.scalers = cfg.paths.models / "scalers"
    cfg.anomaly.fusion_weights = dict(WEIGHTS)
    cfg.anomaly.isolation_forest["n_jobs"] = 4
    cfg.mlops.log_to_file = False
    cfg.mlops.enable_mlflow = False
    cfg.ensure_directories()
    configure_logging(cfg)
    return cfg


def train(run):
    cfg = staged_config(run)
    manifest = Config().paths.models / "training_split.json"
    shutil.copy2(manifest, cfg.paths.models / manifest.name)
    residuals = joblib.load(TRAINING / "training_residuals.joblib")
    prepared = joblib.load(TRAINING / "prepared.joblib")
    context = prepared["labelled"]
    assert residuals["split"].eq("train").all()
    protocol = {"detectors": list(WEIGHTS), "weights": WEIGHTS,
                "training_rows": len(residuals), "training_engines": int(residuals.unit.nunique()),
                "training_plan_id": json.loads(manifest.read_text())["plan_id"],
                "official_test_used_for_fit_or_calibration": False,
                "activation_reason": "User requested all four anomaly detectors; equal weights fixed before evaluation.",
                "digital_twin_refitted": False}
    atomic_write_json(protocol, run / "protocol.json")
    fusion = AnomalyFusion(cfg)
    with threadpool_limits(limits=4):
        summary = fusion.train(residuals, context=context)
    if set(summary) != set(WEIGHTS) or any("error" in value for value in summary.values()):
        raise RuntimeError(f"Every requested detector must train successfully: {summary}")
    print("ALL FOUR DETECTORS TRAINED", flush=True)
    atomic_write_json(summary, run / "training_summary.json")
    from app.utils.Anomaly_Health_Monitoring.training_split import training_scope
    fitting = training_scope(residuals, cfg, ("fit",), required=True)
    with threadpool_limits(limits=4):
        scores = fusion.score(fitting)
        fused = fusion.classify_type(fusion.fuse(scores), context)
    fusion.severity_classifier.calibrate(fused)
    print("FOUR-DETECTOR CALIBRATION AND SEVERITY READY", flush=True)


def prepare_official(run):
    from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import DataCleaner
    from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
    from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
    from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
    from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
    from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import ResidualCalculator
    cfg = Config()
    cfg.mlops.log_to_file = False
    cfg.mlops.enable_mlflow = False
    cfg.twin.startup_profile_enabled = False
    configure_logging(cfg)
    raw = CMAPSSReader(cfg).read_subsets(split="test")
    clean, _ = DataCleaner(cfg).clean(raw, fit_policy=False)
    context = OperatingModeDetector(cfg).load().assign(clean)
    scaled = PerModeScaler(cfg).load_scalers().transform(context).round(6)
    ensemble = EnsembleTwin(cfg).load()
    ensemble.startup_profile = None
    with threadpool_limits(limits=4):
        predictions = ensemble.predict(scaled)
    residuals = ResidualCalculator(cfg).compute(scaled, predictions, recalibrate=False)
    labels = raw.set_index(["unit", "cycle"])["RUL"]
    residuals["RUL"] = labels.reindex(pd.MultiIndex.from_frame(residuals[["unit", "cycle"]])).to_numpy()
    current = pd.read_csv(cfg.paths.processed / cfg.paths.residuals_csv)
    columns = [c for c in current if c.startswith("residual_sensor_")]
    overlap = current[["unit", "cycle"] + columns].merge(
        residuals[["unit", "cycle"] + columns], on=["unit", "cycle"], suffixes=("_old", "_new"))
    if not len(overlap):
        raise ValueError("The official evaluation has no overlap with the current sensor batch.")
    delta = max(float(np.abs(overlap[c + "_old"] - overlap[c + "_new"]).max()) for c in columns)
    joblib.dump({"context": context, "residuals": residuals}, run / "official_inputs.joblib", compress=3)
    atomic_write_json({"rows": len(residuals), "engines": int(residuals.unit.nunique()),
                       "current_batch_overlap": len(overlap), "reference_max_delta": delta,
                       "preprocessing_fitted": False, "adaptive_forecast_used": False}, run / "official_preparation.json")
    if delta > .00001:
        raise ValueError(f"Monitoring reference differs from the persisted batch: {delta}")
    print("OFFICIAL MONITORING RESIDUALS READY", flush=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finalize(run):
    import importlib
    from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_data_generator import DashboardDataGenerator
    from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_api import DashboardApiService
    from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_service import DashboardService
    from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_anomaly import AnomalyEvaluator
    from app.schemas.Anomaly_Health_Monitoring.explainability_schema import DashboardResponse
    from scripts.Anomaly_Health_Monitoring import evaluate_holdout
    active = Config()
    cfg = staged_config(run)
    summary = json.loads((run / "training_summary.json").read_text())
    if set(summary) != set(WEIGHTS) or any("error" in score for score in summary.values()):
        raise RuntimeError("Four successful trained detectors are required before activation.")
    if not (cfg.paths.models / "severity_bands.json").exists():
        raise RuntimeError("Training and severity calibration must finish before activation.")
    metadata_path = cfg.paths.models / "anomaly_fusion_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(fusion_weights=WEIGHTS, detectors=summary, feature_space=cfg.anomaly.feature_space,
                    score_version="four_detector_ensemble_v1", activation_reason="Explicit user request for four models")
    atomic_write_json(metadata, metadata_path)
    reference_models = [p for p in active.paths.models.iterdir() if p.is_file() and
                        (p.name.startswith(("twin_", "startup_profile_", "health_")) or "_twin" in p.name or
                         p.name in ("residual_z_calibration.json", "alert_memory.json"))]
    protected = reference_models + [active.paths.metrics / "digital_twin.json"]
    input_names = [active.paths.raw_data_csv, active.paths.cleaned_data_csv, active.paths.scaled_features_csv,
                   active.paths.context_clusters_csv, active.paths.twin_predictions_csv, active.paths.residuals_csv,
                   active.paths.health_index_csv, active.paths.health_scores_csv, active.paths.health_states_csv]
    protected += [active.paths.processed / name for name in input_names]
    protected = [p for p in protected if p.exists()]
    before = {str(p): digest(p) for p in protected}
    atomic_write_json(before, run / "protected_artifacts.json")
    for source in reference_models:
        target = cfg.paths.models / source.name
        if not target.exists():
            shutil.copy2(source, target)
    for name in input_names:
        source = active.paths.processed / name
        if source.exists():
            shutil.copy2(source, cfg.paths.processed / name)
    with threadpool_limits(limits=4):
        for name in ("05_anomaly_detection", "06_reasoning", "07_explainability", "08_uncertainty"):
            module = importlib.import_module("app.services.Anomaly_Health_Monitoring.pipeline." + name)
            if name == "05_anomaly_detection":
                module.run_stage(cfg, retrain=False)
            elif name == "07_explainability":
                module.run_stage(cfg, include_shap=True)
            else:
                module.run_stage(cfg)
            print("CURRENT BATCH READY:", name, flush=True)
        wide, dashboard_report = DashboardDataGenerator(cfg).run()
        DashboardApiService(cfg).run()
        AnomalyEvaluator(cfg).run()
    payload = DashboardService(cfg).build_all()
    validated = DashboardResponse(fleet=payload["fleet"], engines=payload["engines"], alerts=payload["alerts"],
                                  timeline=payload["timeline"]).model_dump()
    columns = ["isolation_forest_score", "mahalanobis_score", "lstm_ae_score", "residual_rule_score"]
    if not np.isfinite(wide[columns].to_numpy(float)).all():
        raise RuntimeError("Every scored row must have four finite detector scores.")
    for alert in validated["alerts"]:
        for column in columns + ["model_agreement", "detector_agreement", "uncertainty_score", "reliability_score",
                                 "threshold_used", "anomaly_persistence_cycles", "early_warning_score"]:
            if alert.get(column) is None:
                raise RuntimeError(f"Dashboard card data is missing: {column}")
    print("CURRENT DASHBOARD VALIDATED", flush=True)
    official = joblib.load(run / "official_inputs.joblib")
    fusion = AnomalyFusion(cfg)
    fusion.load()
    residuals, context = official["residuals"], official["context"]
    batches = []
    units = residuals.unit.unique()
    for start in range(0, len(units), 50):
        part = residuals.loc[residuals.unit.isin(units[start:start + 50])].copy()
        with threadpool_limits(limits=4):
            values = fusion.classify_type(fusion.fuse(fusion.score(part)), context)
        values = fusion.severity_classifier.classify(values)
        values = values.merge(part[["unit", "cycle", "RUL"]], on=["unit", "cycle"], validate="one_to_one")
        batches.append(values)
        print("OFFICIAL DETECTOR SCORING:", min(start + 50, len(units)), "/", len(units), flush=True)
    scored = pd.concat(batches, ignore_index=True)
    scored.to_csv(run / "official_scores.csv", index=False)
    report = {"schema_version": "2.0.0", "metrics": evaluate_holdout.build(scored, 125, 30),
              "provenance": {"generated_by": str(Path(__file__).relative_to(ROOT)),
                             "evaluation_status": "retrospective_reused_official_benchmark",
                             "benchmark_results_must_not_drive_model_promotion": True,
                             "official_results_used_for_promotion": False, "training_plan_id": metadata["training_plan_id"],
                             "preprocessing_fitted_on_training_only": True, "fusion_weights": WEIGHTS,
                             "rows_joined": len(scored), "saved_scores_sha256": digest(run / "official_scores.csv"),
                             "label_basis": "RUL condition proxies; not confirmed fault labels"}}
    atomic_write_json(report, cfg.paths.metrics / "holdout_anomaly_evaluation.json")
    final_files = ["isolation_forest_detector.pkl", "mahalanobis_detector.pkl", "lstm_autoencoder.keras",
                   "lstm_autoencoder_meta.json", "residual_rule_meta.json", "anomaly_fusion_meta.json",
                   "residual_thresholds.json", "severity_bands.json"]
    copies = [(cfg.paths.models / name, active.paths.models / name) for name in final_files]
    output_names = [active.paths.anomaly_scores_csv, active.paths.anomaly_fusion_csv,
                    active.paths.root_cause_analysis_csv, active.paths.explanation_reports_csv,
                    active.paths.confidence_scores_csv, active.paths.dashboard_data_csv]
    copies += [(cfg.paths.processed / name, active.paths.processed / name) for name in output_names]
    copies += [(p, active.paths.metrics / p.name) for p in cfg.paths.metrics.glob("*.json")]
    copies += [(cfg.paths.data_exchange / active.paths.exchange_predictions_csv,
                active.paths.data_exchange / active.paths.exchange_predictions_csv)]
    provenance_path = active.paths.models / "training_provenance.json"
    existing = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    existing.update(json.loads((cfg.paths.models / "training_provenance.json").read_text()))
    atomic_write_json(existing, run / "merged_training_provenance.json")
    copies.append((run / "merged_training_provenance.json", provenance_path))
    registry_path = active.paths.models / active.paths.model_registry_json
    new_registry_path = cfg.paths.models / cfg.paths.model_registry_json
    if new_registry_path.exists():
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        updates = json.loads(new_registry_path.read_text())
        updates = json.loads(json.dumps(updates).replace(str(cfg.paths.models).replace("\\", "\\\\"),
                                                       str(active.paths.models).replace("\\", "\\\\")))
        registry.setdefault("models", {}).update(updates.get("models", {}))
        atomic_write_json(registry, run / "merged_model_registry.json")
        copies.append((run / "merged_model_registry.json", registry_path))
    for source, target in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            backup = run / "before" / target.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(target, backup)
    installed = []
    try:
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".four-detectors.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            installed.append(target)
        if any(digest(Path(name)) != value for name, value in before.items()):
            raise RuntimeError("A protected digital twin or health artifact changed.")
    except Exception:
        for target in reversed(installed):
            backup = run / "before" / target.relative_to(ROOT)
            if backup.exists():
                shutil.copy2(backup, target)
            else:
                target.unlink(missing_ok=True)
        raise
    verification = {"activated_detectors": list(WEIGHTS), "fusion_weights": WEIGHTS,
                    "fusion_threshold": metadata["calibrated_fusion_threshold"],
                    "current_rows": len(wide), "current_engines": int(wide.unit.nunique()),
                    "alerts_validated": len(validated["alerts"]), "all_four_scores_finite": True,
                    "protected_twin_and_health_artifacts_unchanged": True,
                    "severity_summary": report["metrics"]["test"].get("severity_mean_rul"),
                    "official_metrics": report["metrics"]["test"]}
    atomic_write_json(verification, run / "activation_verification.json")
    print("ALL FOUR DETECTORS ACTIVATED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare-official", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.prepare_official:
        prepare_official(run)
    elif args.finalize:
        finalize(run)
    else:
        train(run)


if __name__ == "__main__":
    main()
