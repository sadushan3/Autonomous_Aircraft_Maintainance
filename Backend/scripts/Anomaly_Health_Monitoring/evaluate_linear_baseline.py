from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import AnomalyFusion
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261014_leakage_fix"
DEFAULT_RUN = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261015_linear_baseline"
FEATURES = [f"sensor_{i}" for i in (2, 3, 4, 7, 11, 12, 15, 20, 21)]
KEYS = ["unit", "cycle", "split", "subset_id"]
PARAMETERS = dict(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000,
                  tol=1e-4, random_state=42)
CUTOFF = 0.5


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def inventory(directories):
    return {str(path): digest(path) for directory in directories
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score(model, frame, config):
    values = frame[FEATURES].to_numpy(float)
    if not np.isfinite(values).all() or frame.duplicated(KEYS).any():
        raise ValueError("Baseline requires finite sensor readings and unique identities")
    output = frame[KEYS].copy()
    output["calibration_ready"] = output.cycle > config.twin.commissioning_cycles
    output["anomaly_score"] = model.predict_proba(values)[:, 1]
    output["threshold_used"] = CUTOFF
    output["anomaly_flag"] = (output.anomaly_score > CUTOFF) & output.calibration_ready

    return AnomalyFusion(config).classify_type(output, frame)


def label(scores, labels):
    result = scores.merge(labels[KEYS + ["RUL"]], on=KEYS, how="left",
                          validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all() or not np.isfinite(result.RUL.to_numpy(float)).all():
        raise ValueError("Every prediction must have a matching finite RUL proxy")
    return result.drop(columns="_merge")


def measured(frame):
    result = condition_metrics(frame)
    result["per_subset"] = {str(name): condition_metrics(part, bootstrap=False)
                            for name, part in frame.groupby("subset_id", sort=True)}
    return result


def prepare(raw, config, context, scaler):
    cleaned, _ = DataCleaner(config).clean(raw, fit_policy=False)
    return scaler.transform(context.assign(cleaned))


def presentation(result, run, report_path):
    baseline = result["official_reused_benchmark"]
    active = result["active_consensus_comparator"]
    link = lambda path: Path(os.path.relpath(path, report_path.parent)).as_posix()
    number = lambda value: "Unavailable" if value is None else f"{value:.4f}"
    lines = ["# Fixed linear anomaly baseline - 15 October 2026", "",
        "**This is a separate baseline experiment. The deployed consensus model remains active.**", "",
        "One logistic-regression classifier uses nine current, per-mode-standardized sensor readings. "
        "It has nine coefficients and an intercept, with L2 regularization (C = 1), balanced class weights, "
        "and a fixed 0.5 cutoff. It uses no age, engine IDs, RUL, future observations, rolling sensor history, "
        "digital twins, covariance distance, or consensus ensemble as prediction features.", "",
        "The sensor channels were specified in the saved protocol before fitting. They follow the existing "
        "production channel list, which was developed on previously inspected data; this is not blind feature selection. "
        "The baseline shares the corrected frozen operating-mode discovery, cleaning, scaling, 30-cycle commissioning, "
        "and operational-transition gate. All classifier fitting uses only the 396 fitting engines. "
        "There is one recipe, no hyperparameter search, and no cutoff adjustment toward a target score.", "",
        "## Comparable reused-benchmark results", "",
        "Both models use the same engine/cycle population and healthy RUL >= 125 versus critical RUL <= 30. "
        "Intermediate RUL 31-124 rows are excluded from these binary metrics.", "",
        "| Metric | Fixed linear baseline | Active consensus |", "|---|---:|---:|"]
    for key, title in (("accuracy", "Accuracy"), ("f1", "F1"), ("precision", "Precision"),
                       ("recall", "Recall"), ("auc", "ROC AUC"), ("average_precision", "Average precision"),
                       ("false_alarm_rate", "Healthy-cycle false-alert rate")):
        lines.append(f"| {title} | {number(baseline[key])} | {number(active[key])} |")
    for key, title in (("false_positive", "False healthy rows"), ("false_negative", "Missed critical rows")):
        lines.append(f"| {title} | {baseline[key]:,} | {active[key]:,} |")
    lines += ["", f"The always-healthy accuracy baseline is **{baseline['always_healthy_accuracy']:.4f}**. "
        "High accuracy must be interpreted alongside class imbalance and F1/recall.", "",
        "| Baseline metric | 95% whole-engine bootstrap interval |", "|---|---:|"]
    for key in ("f1", "precision", "recall", "false_alarm_rate"):
        bounds = baseline["confidence_intervals"][key]
        lines.append(f"| {key} | {bounds[0]:.6f}-{bounds[1]:.6f} |")
    lines += ["", "Intervals use 300 engine-level bootstrap replicates. They do not remove historical benchmark reuse bias.", "",
        "## Baseline results by subset", "",
        "| Subset | Accuracy | F1 | Precision | Recall | False healthy rows | Missed critical rows |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in baseline["per_subset"].items():
        lines.append(f"| {name} | {item['accuracy']:.4f} | {item['f1']:.4f} | {item['precision']:.4f} | "
                     f"{item['recall']:.4f} | {item['false_positive']} | {item['false_negative']} |")
    lines += ["", "## Other retrospective populations", "",
        "These are reports for the same fixed classifier, without selection or retuning after evaluation.", "",
        "| Population | Engines | Accuracy | F1 | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for key, title in (("development", "Development"), ("confirmation", "Retrospective confirmation")):
        item = result[key]
        lines.append(f"| {title} | {item['input_engines']} | {item['accuracy']:.4f} | {item['f1']:.4f} | "
                     f"{item['precision']:.4f} | {item['recall']:.4f} |")
    lines += ["", "## Coverage and interpretation", "",
        f"Official input: {baseline['input_rows']:,} rows from {baseline['input_engines']} engines. "
        f"Ready predictions: {baseline['evaluated_rows']:,} rows from {baseline['evaluated_engines']} engines. "
        f"Binary evaluation: {baseline['rows']:,} rows ({baseline['healthy_rows']:,} healthy and "
        f"{baseline['critical_rows']:,} critical). Excluded intermediate rows: "
        f"{baseline['middle_band_rows_excluded_from_binary_metrics']:,}.", "",
        "These are retrospective RUL-proxy results on previously inspected C-MAPSS data. "
        "Independent blind-test performance and confirmed fault-onset accuracy remain unmeasured. "
        "This baseline must be named separately when presented; its results do not replace the consensus results.", "",
        "## Verification", "",
        f"- Only {result['fitting']['engines']} permitted fitting engines and {result['fitting']['labelled_rows']:,} "
        "post-commissioning healthy/critical rows fitted the linear model.",
        "- Reloaded classifier scores match the saved classifier exactly.",
        "- Removing RUL/population label fields from sensor features cannot change scores; only the nine declared sensors enter prediction.",
        "- Independently recomputed raw-data prefixes preserve scores and operational decisions.",
        "- Benchmark coverage and confusion counts reproduce from the saved prediction CSV.",
        f"- {result['verification']['protected_production_files']} production artifact hashes and the "
        "source model bundle remain unchanged.", "",
        f"[Frozen protocol]({link(run / 'protocol.json')}), "
        f"[model lock]({link(run / 'model_lock.json')}), "
        f"[full evaluation and verification]({link(run / 'evaluation.json')}), "
        f"[measured comparison CSV]({link(run / 'comparison.csv')}).", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    source, run = Path(args.source_run).resolve(), Path(args.run_dir).resolve()
    if not run.is_relative_to(ROOT / "experiments") or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("Use a new experiment directory separate from the source bundle")
    run.mkdir(parents=True, exist_ok=False)
    config = Config()
    config.apply_overrides(json.loads((source / "protocol.json").read_text())["frozen_config"])
    config.paths.models = source / "bundle/models"
    config.paths.scalers = config.paths.models / "scalers"
    config.mlops.log_to_file = False
    config.mlops.enable_mlflow = False
    logging.basicConfig(level=logging.WARNING)
    plan = load_training_split(config)
    if plan is None:
        raise ValueError("A verified shared engine split is required")
    scope = json.loads((config.paths.models / "training_provenance.json").read_text())
    for name in ("cleaning", "context", "scaling"):
        item = scope[name]
        if item["plan_id"] != plan["plan_id"] or item["roles"] != ["fit"] or set(item["units"]) != set(plan["engine_roles"]["fit"]):
            raise ValueError(f"Invalid fitting scope for {name}")
    active_paths = Config().paths
    protected_directories = [getattr(active_paths, name) for name in ("models", "processed", "metrics", "data_exchange")]
    production_before = inventory(protected_directories)
    source_before = inventory([config.paths.models])
    write_json(run / "protected_artifacts_before.json", production_before)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": "logistic_sensor_baseline",
        "features": FEATURES, "parameters": PARAMETERS, "regularization": "L2", "cutoff": CUTOFF,
        "healthy_rul_min": 125, "critical_rul_max": 30, "training_role": "fit",
        "training_plan_id": plan["plan_id"], "commissioning_cycles": config.twin.commissioning_cycles,
        "operational_persistence_cycles": config.anomaly.persistence_min_cycles,
        "operational_transition_guard": config.context.mode_transition_guard,
        "hyperparameter_search": False, "score_target": None, "production_replacement": False,
        "evaluation_status": "retrospective_reused_benchmark", "sklearn_version": sklearn.__version__,
        "prepared_sha256": digest(source / "prepared.joblib"),
        "script_sha256": digest(Path(__file__)), "source_model_hashes": source_before}
    write_json(run / "protocol.json", protocol)
    print("Baseline protocol locked; one fixed recipe, no score target", flush=True)

    prepared = joblib.load(source / "prepared.joblib")["scaled"]
    fitting = training_scope(prepared, config, required=True)
    keep = (fitting.cycle > config.twin.commissioning_cycles) & ((fitting.RUL >= 125) | (fitting.RUL <= 30))
    fitting = fitting.loc[keep]
    x, y = fitting[FEATURES].to_numpy(float), fitting.RUL.le(30).to_numpy(int)
    if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Fitting requires finite readings and both proxy classes")
    if not set(fitting.unit.unique()) <= set(plan["engine_roles"]["fit"]):
        raise ValueError("Baseline fitting escaped the shared training role")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(**PARAMETERS).fit(x, y)
    model_path = run / "logistic_sensor_baseline.joblib"
    joblib.dump({"model": model, "features": FEATURES, "protocol_sha256": digest(run / "protocol.json")}, model_path)
    write_json(run / "model_lock.json", {"model_sha256": digest(model_path), "cutoff": CUTOFF,
        "locked_before_baseline_development_confirmation_and_official_scoring": True,
        "fit_unit_ids": sorted(map(int, fitting.unit.unique())), "labelled_fit_rows": len(fitting),
        "iterations": model.n_iter_.tolist()})
    restored = joblib.load(model_path)["model"]
    np.testing.assert_array_equal(model.predict_proba(x), restored.predict_proba(x))
    result = {"protocol": protocol, "fitting": {"engines": fitting.unit.nunique(), "labelled_rows": len(fitting)},
        "model_sha256": digest(model_path)}
    for role, name in (("selection", "development"), ("confirmation", "confirmation")):
        frame = training_scope(prepared, config, (role,), required=True)
        result[name] = measured(label(score(model, frame, config), frame))
    print("Fixed model locked and development/confirmation reported; no retuning", flush=True)

    context, scaler = OperatingModeDetector(config).load(), PerModeScaler(config).load_scalers()
    raw = CMAPSSReader(config).read_subsets(split="test")
    official = prepare(raw, config, context, scaler)
    official_scores = score(model, official, config)
    official_scores.to_csv(run / "official_scores.csv", index=False)
    official_labels = pd.read_csv(source / "official_labels.csv")
    labelled = label(official_scores, official_labels)
    result["official_reused_benchmark"] = measured(labelled)
    active = json.loads((source / "evaluation.json").read_text())["official_reused_benchmark"]
    result["active_consensus_comparator"] = active
    baseline = result["official_reused_benchmark"]
    for name in ("input_rows", "input_engines", "evaluated_rows", "evaluated_engines", "rows", "healthy_rows", "critical_rows", "middle_band_rows_excluded_from_binary_metrics"):
        if baseline[name] != active[name]:
            raise ValueError(f"Comparator coverage differs for {name}")

    active_scores = pd.read_csv(source / "official_scores.csv")
    pd.testing.assert_frame_equal(official_scores[KEYS].sort_values(KEYS).reset_index(drop=True),
                                 active_scores[KEYS].sort_values(KEYS).reset_index(drop=True), check_dtype=False)
    replay = measured(label(pd.read_csv(run / "official_scores.csv"), official_labels))
    for name in ("accuracy", "f1", "recall", "precision", "true_positive", "false_positive", "false_negative", "true_negative"):
        if abs(replay[name] - baseline[name]) > 1e-12:
            raise ValueError(f"Saved-score replay differs for {name}")
    np.testing.assert_array_equal(model.predict_proba(official[FEATURES].to_numpy(float)),
        model.predict_proba(official.drop(columns=[c for c in ("RUL", "RUL_true", "cycle_norm", "unit", "cycle", "split", "subset_id") if c in official])[FEATURES].to_numpy(float)))
    units = sorted(raw.unit.unique())[:4]
    prefix_raw = raw.loc[raw.unit.isin(units) & raw.cycle.le(60)].copy()
    prefix = score(restored, prepare(prefix_raw, config, context, scaler), config)
    expected = official_scores.merge(prefix[KEYS], on=KEYS, validate="one_to_one").sort_values(KEYS)
    prefix = prefix.sort_values(KEYS)
    error = float(np.max(np.abs(expected.anomaly_score.to_numpy() - prefix.anomaly_score.to_numpy())))
    np.testing.assert_allclose(expected.anomaly_score, prefix.anomaly_score, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(expected.anomaly_type, prefix.anomaly_type)
    if production_before != inventory(protected_directories) or source_before != inventory([config.paths.models]):
        raise ValueError("Baseline evaluation changed protected model or serving artifacts")
    if digest(model_path) != result["model_sha256"]:
        raise ValueError("Baseline model changed after it was locked")
    result["verification"] = {"protected_production_files": len(production_before),
        "production_artifacts_unchanged": True, "source_models_unchanged": True,
        "baseline_model_unchanged": True, "reload_identical": True, "saved_prediction_replay_passed": True,
        "label_metadata_not_model_features": True, "prefix_rows": len(prefix), "prefix_max_score_error": error,
        "prefix_fault_decisions_identical": True, "comparison_population_identical": True}
    write_json(run / "evaluation.json", result)
    rows = [{"model": name, "population": population, "accuracy": metrics["accuracy"], "f1": metrics["f1"],
        "precision": metrics["precision"], "recall": metrics["recall"], "false_alarm_rate": metrics["false_alarm_rate"],
        "false_positive": metrics["false_positive"], "false_negative": metrics["false_negative"],
        "evaluation_status": "retrospective_reused_benchmark"}
        for name, populations in (("logistic_sensor_baseline", {"overall": baseline, **baseline["per_subset"]}),
                                  ("active_consensus", {"overall": active, **active["per_subset"]}))
        for population, metrics in populations.items()]
    pd.DataFrame(rows).to_csv(run / "comparison.csv", index=False)
    presentation(result, run, ROOT / "reports/anomaly_linear_baseline_20261015.md")
    print(json.dumps({"baseline_accuracy": baseline["accuracy"], "baseline_f1": baseline["f1"],
        "baseline_precision": baseline["precision"], "baseline_recall": baseline["recall"],
        "verification": result["verification"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    with threadpool_limits(limits=4):
        main(parser.parse_args())
