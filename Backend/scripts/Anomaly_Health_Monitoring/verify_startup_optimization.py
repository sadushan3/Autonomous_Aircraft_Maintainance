from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import (
    ROOT, RUN, SOURCE, digest, score, write, improves,
)
from scripts.Anomaly_Health_Monitoring.evaluate_linear_baseline import prepare
from scripts.Anomaly_Health_Monitoring.retrain_without_leakage import labelled_scores
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import ResidualCalculator
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics
from app.services.Anomaly_Health_Monitoring.digital_twin.forecast_policy import healthy_forecast_guard


def inventory(directory):
    return {str(p.relative_to(ROOT)): digest(p) for p in directory.rglob("*") if p.is_file()}


def comparison(frame, before, after, targets, mask):
    actual = frame[targets].to_numpy(float)
    columns = ["expected_" + s for s in targets]
    return {"baseline": score(frame, actual, before[columns].to_numpy(float), mask),
            "candidate": score(frame, actual, after[columns].to_numpy(float), mask)}


def subset_table(result):
    lines = ["## Results for each subset", "",
        "| Population / subset | R² before | R² after | RMSE before | RMSE after | MAE before | MAE after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for key, title in (("confirmation", "Confirmation"), ("official_reused_benchmark", "Official")):
        item = result[key]
        for name, before in item["baseline"]["per_subset"].items():
            after = item["candidate"]["per_subset"][name]
            lines.append(f"| {title} / {name} | {before['r2']:.6f} | {after['r2']:.6f} | {before['rmse']:.6f} | {after['rmse']:.6f} | {before['mae']:.6f} | {after['mae']:.6f} |")
    return lines + ["", "Official FD003 has a small regression in all three metrics. Official data was report-only and did not select the recipe. This change improves aggregate results; it does not improve every metric in every subset.", ""]


def main(args):
    run = Path(args.run_dir).resolve()
    result = json.loads((run / "evaluation.json").read_text())
    lock = json.loads((run / "model_lock.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    if not result["accepted"] or digest(run / "candidate_profile.pkl") != lock["candidate_sha256"]:
        raise ValueError("Only the accepted, frozen candidate may be verified")
    config = Config()
    config.mlops.log_to_file = config.mlops.enable_mlflow = False
    if inventory(config.paths.models) != protocol["protected_models"]:
        raise ValueError("Serving models changed since the experiment; compare against the new baseline")
    candidate = joblib.load(run / "candidate_profile.pkl")
    adaptive = candidate.get("prediction_role") == "adaptive_sensor_forecast"
    recipe = json.loads((run / "selected_recipe.json").read_text())
    source = Path(args.source_run).resolve()
    prepared = joblib.load(source / "prepared.joblib")["scaled"]
    print("Loading serving ensemble and frozen preprocessing", flush=True)
    ensemble = EnsembleTwin(config).load()


    for twin in ensemble.twins:
        if twin.name == "random_forest":
            twin.model.n_jobs = 1
    targets = ensemble.target_names
    context, scaler = OperatingModeDetector(config).load(), PerModeScaler(config).load_scalers()
    raw = CMAPSSReader(config).read_subsets(split="test")
    official = prepare(raw, config, context, scaler)
    print("Scoring locked candidate on official trajectories (report only)", flush=True)
    inference_started = time.perf_counter()
    cpu_started = time.process_time()
    before = ensemble.predict(official)
    baseline_seconds = time.perf_counter() - inference_started
    baseline_cpu_seconds = time.process_time()-cpu_started
    ensemble.startup_profile = candidate
    inference_started = time.perf_counter()
    cpu_started = time.process_time()
    after = ensemble.predict(official)
    result["inference_timing"] = {"baseline_seconds":baseline_seconds,
        "candidate_seconds":time.perf_counter()-inference_started,
        "baseline_cpu_seconds":baseline_cpu_seconds,"candidate_cpu_seconds":time.process_time()-cpu_started,
        "rows":len(official), "forest_audit_threads":1,
        "measurement":"One full-batch run per model; first-run framework warmup can affect timings"}
    ready = after.startup_profile_ready.to_numpy(bool)
    mask = ready & official.RUL.ge(125).to_numpy()
    result["official_reused_benchmark"] = comparison(official, before, after, targets, mask)
    if adaptive:


        result["official_full_trajectory_forecasts"] = comparison(official,before,after,targets,ready)
    result["official_aggregate_metrics_improve"] = improves(
        result["official_reused_benchmark"]["baseline"],result["official_reused_benchmark"]["candidate"])
    full_objective = result.get("objective") == "full_trajectory_forecasts"
    result["official_publication_gate_passes"] = result["official_aggregate_metrics_improve"]
    if full_objective:
        result["official_healthy_metrics_improve"] = result["official_aggregate_metrics_improve"]
        result["official_aggregate_metrics_improve"] = improves(
            result["official_full_trajectory_forecasts"]["baseline"],result["official_full_trajectory_forecasts"]["candidate"])
        result["official_healthy_guard_passes"] = healthy_forecast_guard(
            result["official_reused_benchmark"]["baseline"],result["official_reused_benchmark"]["candidate"],result["healthy_guard"])
        result["official_publication_gate_passes"] = result["official_aggregate_metrics_improve"] and result["official_healthy_guard_passes"]
        write(run/"official_full_comparison.json",result["official_full_trajectory_forecasts"])
    write(run / "official_comparison.json", result["official_reused_benchmark"])
    references = [c for c in before if c.startswith(("reference_expected_", "twin_spread_"))]
    pd.testing.assert_frame_equal(before[references], after[references], check_exact=True)
    before_residuals = ResidualCalculator(config).compute(official, before, recalibrate=False)
    after_residuals = ResidualCalculator(config).compute(official, after, recalibrate=False)
    monitoring = [c for c in before_residuals if not c.startswith("prediction_residual_")]
    pd.testing.assert_frame_equal(before_residuals[monitoring], after_residuals[monitoring], check_exact=True)
    fusion = AnomalyFusion(config)
    fusion.load()
    for detector in fusion.detectors.values():
        for member in getattr(detector, "members", []):
            for model in member["models"]:
                model.n_jobs = 1
    old_scores = labelled_scores(fusion, before_residuals, official)
    new_scores = labelled_scores(fusion, after_residuals, official)
    pd.testing.assert_frame_equal(old_scores, new_scores, check_exact=True)
    pd.testing.assert_frame_equal(HealthIndexCalculator(config).compute(before_residuals),
                                  HealthIndexCalculator(config).compute(after_residuals), check_exact=True)
    result["anomaly_metrics"] = condition_metrics(new_scores)
    print("Monitoring residuals, anomaly decisions and health indices match exactly", flush=True)


    units = raw.groupby("subset_id").unit.first().tolist()
    prefix_raw = raw.loc[raw.unit.isin(units) & raw.cycle.le(60)].copy()
    prefix = prepare(prefix_raw, config, context, scaler)
    prefix_prediction = ensemble.predict(prefix)
    columns = ["expected_" + s for s in targets]
    expected = after.merge(prefix_prediction[["unit", "cycle"]], on=["unit", "cycle"], validate="one_to_one")
    expected = expected.sort_values(["unit", "cycle"])
    prefix_prediction = prefix_prediction.sort_values(["unit", "cycle"])
    np.testing.assert_allclose(expected[columns], prefix_prediction[columns], rtol=0, atol=1e-10)
    changed = prefix.copy()
    changed.loc[changed.cycle.ge(45) if adaptive else changed.cycle.gt(30), targets] += 1000.
    perturbed = ensemble.predict(changed)
    perturbed = perturbed.sort_values(["unit", "cycle"])
    if adaptive:
        past = prefix_prediction.cycle.le(45).to_numpy()
        np.testing.assert_allclose(prefix_prediction.loc[past,columns],perturbed.loc[past,columns],rtol=0,atol=1e-10)
        if np.allclose(prefix_prediction.loc[~past,columns],perturbed.loc[~past,columns],rtol=0,atol=1e-10):
            raise ValueError("Adaptive forecast did not respond to changed prior sensor readings")
        pd.testing.assert_frame_equal(prefix_prediction[references],perturbed[references],check_exact=True)
    else:
        np.testing.assert_allclose(prefix_prediction[columns],perturbed[columns],rtol=0,atol=1e-10)
    stripped = prefix.drop(columns=[c for c in ("RUL", "RUL_true", "cycle_norm") if c in prefix])
    np.testing.assert_allclose(ensemble.predict(stripped)[columns], ensemble.predict(prefix)[columns], rtol=0, atol=1e-10)
    shuffled = prefix.sample(frac=1., random_state=42)
    np.testing.assert_allclose(ensemble.predict(shuffled).sort_values(["unit", "cycle"])[columns],
                               prefix_prediction[columns], rtol=0, atol=1e-10)


    retrain_config = copy.deepcopy(config)
    retrain_config.paths.models = run / "retrain_check/models"
    retrain_config.paths.models.mkdir(parents=True, exist_ok=True)
    for name in ("training_split.json",):
        shutil.copy2(config.paths.models / name, retrain_config.paths.models / name)
    write(retrain_config.paths.models / "startup_profile_recipe.json", recipe)
    retrained = EnsembleTwin(retrain_config, twins=ensemble.twins)
    retrained.weights, retrained.target_names = ensemble.weights, targets
    retrained._fit_startup_profile(prepared)
    if retrained.startup_profile is None or retrained.startup_profile["correction_recipe"] != recipe["name"]:
        raise ValueError("Normal retraining did not accept the selected recipe")
    if retrained.startup_profile.get("training_plan_id") != candidate["training_plan_id"]:
        raise ValueError("Normal retraining lost the verified training provenance")
    if adaptive and retrained.startup_profile.get("prediction_role") != "adaptive_sensor_forecast":
        raise ValueError("Normal retraining lost the adaptive forecast role")
    if full_objective and retrained.startup_profile.get("prediction_objective") != "full_trajectory_forecasts":
        raise ValueError("Normal retraining lost the full-trajectory objective")

    xp = candidate["population"].predict(EnsembleTwin._profile_context(prefix, candidate))
    xf, xr = EnsembleTwin._profile_features(prefix, xp, candidate)
    rp = retrained.startup_profile["population"].predict(EnsembleTwin._profile_context(prefix, retrained.startup_profile))
    rf, rr = EnsembleTwin._profile_features(prefix, rp, retrained.startup_profile)
    np.testing.assert_allclose(xp, rp, rtol=0, atol=1e-10)
    np.testing.assert_allclose(candidate["correction"].predict(xf[xr]),
        retrained.startup_profile["correction"].predict(rf[rr]), rtol=0, atol=1e-10)
    replay_rows = None
    if full_objective:
        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope
        replay = training_scope(prepared,config,("selection",),required=True)
        ep = candidate["population"].predict(EnsembleTwin._profile_context(replay,candidate))
        rp = retrained.startup_profile["population"].predict(EnsembleTwin._profile_context(replay,retrained.startup_profile))
        ef,er = EnsembleTwin._profile_features(replay,ep,candidate)
        rf,rr = EnsembleTwin._profile_features(replay,rp,retrained.startup_profile)
        np.testing.assert_array_equal(er,rr)
        np.testing.assert_allclose(ep,rp,rtol=0,atol=1e-10)
        np.testing.assert_allclose(EnsembleTwin._predict_profile_correction(candidate["correction"],ef[er]),
            EnsembleTwin._predict_profile_correction(retrained.startup_profile["correction"],rf[rr]),rtol=0,atol=1e-10)
        replay_rows = int(er.sum())

    result["verification"] = {"official_input_rows": len(official),
        "official_input_engines": int(official.unit.nunique()),
        "monitoring_references_identical": True, "monitoring_residuals_identical": True,
        "anomaly_scores_and_decisions_identical": True, "health_indices_identical": True,
        "raw_prefix_predictions_identical": True, "prefix_rows": len(prefix),
        ("current_and_future_readings_do_not_change_forecasts" if adaptive else
         "post_commissioning_sensor_perturbation_does_not_change_expectations"): True,
        "prediction_role":candidate.get("prediction_role","frozen_commissioning_expectation"),
        "RUL_and_future_length_not_prediction_features": True,
        "row_order_invariance": True, "normal_retraining_reproduces_candidate": True,
        "normal_retraining_preserves_plan_and_prediction_role":True,
        "normal_retraining_full_selection_rows":replay_rows,
        "forest_audit_threads": 1,
        "candidate_sha256": lock["candidate_sha256"], "production_models_unchanged_during_verification": True}
    if inventory(config.paths.models) != protocol["protected_models"]:
        raise ValueError("Verification mutated serving artifacts")
    write(run / "verified_evaluation.json", result)
    print("OFFICIAL SENSOR FORECAST" if adaptive else "OFFICIAL HEALTHY EXPECTATION",
          json.dumps(result["official_reused_benchmark"]), flush=True)
    if args.apply:
        if not result["official_publication_gate_passes"]:
            raise ValueError("Locked candidate did not pass the declared official publication gates; retaining active model")
        publish(config, run, result, candidate, recipe, ensemble)
        write(run / "verified_evaluation.json",result)
    return result, ensemble, recipe, config


def publish(config, run, result, candidate, recipe, ensemble):

    publication = run / "publication.json"
    if publication.exists():
        raise ValueError("This run was already applied")
    advanced = "metrics" in result["selected"]
    selected_metrics = result["selected"]["metrics"] if advanced else result["selected"]["inner"]
    development_metrics = result["selected"]["metrics"] if advanced else result["selected"]["selection"]
    adaptive = candidate.get("prediction_role") == "adaptive_sensor_forecast"
    report_name = result.get("report_name",("causal_sensor_forecast_20261016.md" if adaptive else
                   "startup_profile_target_080_20261016.md" if advanced else "startup_profile_optimization_20261016.md"))
    metric_name = result.get("metric_name",("causal_sensor_forecast.json" if adaptive else
                   "startup_profile_target_080.json" if advanced else "startup_profile_optimization.json"))
    if Path(report_name).name!=report_name or Path(metric_name).name!=metric_name:
        raise ValueError("Report and metric names must be filenames")
    current = pd.read_csv(config.paths.processed / config.paths.scaled_features_csv)

    stage = run / "publication_staging"
    stage.mkdir(exist_ok=False)
    staged = {}

    def json_file(destination, value):
        output = stage / destination.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output, value)
        staged[destination] = output

    profile_path = config.paths.models / "startup_profile_twin.pkl"
    profile_stage = stage / "models/startup_profile_twin.pkl"
    profile_stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "candidate_profile.pkl", profile_stage)
    staged[profile_path] = profile_stage
    recipe.update(training_plan_id=candidate["training_plan_id"],
                  validation_metrics=selected_metrics,
                  development_validation_metrics=development_metrics)
    json_file(config.paths.models / "startup_profile_recipe.json", recipe)
    startup = {"active": True, "validation_metrics": selected_metrics,
        "prediction_role":candidate.get("prediction_role","frozen_commissioning_expectation"),
        "prediction_objective":candidate.get("prediction_objective","healthy_sensor_forecasts"),
        "feature_schema":candidate.get("feature_schema"),
        "validation_role":"selection" if advanced else "inner_development",
        "development_validation_metrics": development_metrics,
        "confirmation_metrics": result["confirmation"]["candidate"],
        "official_reused_benchmark_metrics":result["official_reused_benchmark"]["candidate"],
        "monitoring_reference": "unchanged_context_ensemble", "spread_describes": "monitoring_reference_ensemble",
        "recipe": recipe["name"], "optimization_report": str(config.paths.reports / report_name)}
    if result.get("objective") == "full_trajectory_forecasts":
        startup.update(full_trajectory_validation_metrics=result["development_full_trajectories"]["candidate"],
            full_trajectory_confirmation_metrics=result["confirmation_full_trajectories"]["candidate"],
            official_full_trajectory_metrics=result["official_full_trajectory_forecasts"]["candidate"],
            healthy_guard=result["healthy_guard"])
    for destination in (config.paths.models / config.paths.twin_weights_json,
            config.paths.metrics / "digital_twin.json", config.paths.metrics / "evaluate_digital_twin.json",
            config.paths.reports / "digital_twin_evaluation.json"):
        if destination.is_file():
            value = json.loads(destination.read_text())
            value["startup_profile"] = startup
            json_file(destination, value)

    predictions = ensemble.predict(current)
    residuals = ResidualCalculator(config).compute(current, predictions, recalibrate=False)


    keys = ["unit", "cycle"]
    saved_predictions = config.paths.processed / config.paths.twin_predictions_csv
    saved_residuals = config.paths.processed / config.paths.residuals_csv
    if saved_predictions.is_file():
        existing = pd.read_csv(saved_predictions).set_index(keys)
        replacement = predictions.set_index(keys)
        pd.testing.assert_index_equal(existing.index.sort_values(), replacement.index.sort_values())
        for column in ["expected_" + s for s in ensemble.target_names] + ["startup_profile_ready"]:
            existing[column] = replacement[column].reindex(existing.index)
        predictions = existing.reset_index()
    if saved_residuals.is_file():
        existing = pd.read_csv(saved_residuals).set_index(keys)
        replacement = residuals.set_index(keys)
        pd.testing.assert_index_equal(existing.index.sort_values(), replacement.index.sort_values())
        for column in replacement:
            if column.startswith("prediction_residual_"):
                existing[column] = replacement[column].reindex(existing.index)
        residuals = existing.reset_index()
    for destination, frame in ((config.paths.processed / config.paths.twin_predictions_csv, predictions),
                               (config.paths.processed / config.paths.residuals_csv, residuals)):
        output = stage / destination.relative_to(ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False, float_format="%.6f")
        staged[destination] = output
    result.update(production_changed=True, applied_recipe=recipe["name"],
                  evaluation_status="retrospective_previously_inspected_fleet")
    json_file(config.paths.metrics / metric_name, result)
    report_path = config.paths.reports / report_name
    output = stage / report_path.relative_to(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.Anomaly_Health_Monitoring.profile_target_report import render
    output.write_text(render(result, run, ROOT), encoding="utf-8")
    staged[report_path] = output
    before = run / "before"
    manifest = {}
    for destination in staged:
        if destination.is_file():
            backup = before / destination.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
            manifest[str(destination.relative_to(ROOT))] = digest(destination)
        else:
            manifest[str(destination.relative_to(ROOT))] = None
    write(run / "publication_before.json", manifest)
    replaced = []
    try:
        for destination, staged_path in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, destination)
            replaced.append(destination)
        restored = joblib.load(profile_path)
        if digest(profile_path) != result["verification"]["candidate_sha256"]:
            raise ValueError("Published model checksum mismatch")
        ensemble.startup_profile = restored
        reloaded = ensemble.predict(current).set_index(keys)
        expected = predictions.set_index(keys).reindex(reloaded.index)
        np.testing.assert_allclose(reloaded[["expected_" + s for s in ensemble.target_names]],
                                   expected[["expected_" + s for s in ensemble.target_names]], rtol=0, atol=1e-10)
        write(publication, {"applied": True, "recipe": recipe["name"],
            "files": {str(p.relative_to(ROOT)): digest(p) for p in staged},
            "reload_predictions_identical": True})
    except Exception:
        for destination in reversed(replaced):
            backup = before / destination.relative_to(ROOT)
            if backup.is_file():
                shutil.copy2(backup, destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    print("APPLIED", recipe["name"], "with identical reloaded predictions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--source-run", default=str(SOURCE))
    parser.add_argument("--apply", action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
