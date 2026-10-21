from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.advanced_profile_models import ProfileRegressor, ColumnModel, commissioning_statistics
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import (
    ROOT, SOURCE, aggregate, digest, improves, score, subset_gate, write,
)
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
    BlendedStartupCorrection, SensorBlendStartupCorrection,
)
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope

RUN = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261016_target_080"


def recipes(extended=False):
    if extended:
        for grouping in ("subset", "subset_mode", "none"):
            for alpha in (1., 10., 100., 1000.):
                yield dict(family="ridge", grouping=grouping, alpha=alpha, bins=20)
            for components in (2, 4, 8, 16):
                yield dict(family="pls", grouping=grouping, components=components, bins=20)
            for components in (8, 16, 32):
                for gamma in (.001, .01):
                    yield dict(family="pca_kernel", grouping=grouping, components=components,
                               gamma=gamma, alpha=1., bins=20)
        for alpha in (.1, 1., 10.):
            for gamma in (.0001, .001, .01):
                yield dict(family="kernel", grouping="subset", alpha=alpha, gamma=gamma, bins=20)
        for leaf in (3, 10, 30):
            yield dict(family="extra_trees", grouping="subset", leaf=leaf, bins=20)
        for components in (7, 15):
            yield dict(family="lightgbm", grouping="subset", components=components,
                       leaf=20, n_estimators=400, alpha=10., bins=20)
        return
    for grouping in ("subset", "subset_mode", "mode"):
        for features in ("all", "denoised"):
            for alpha in (.1, 1., 10.):
                for gamma in (.0003, .003, .03):
                    yield dict(family="kernel", grouping=grouping, features=features,
                               alpha=alpha, gamma=gamma, bins=20)
    for grouping in ("subset", "subset_mode", "none"):
        for features in ("all", "interactions", "spline", "global"):
            for alpha in (.1, 10., 1000.):
                yield dict(family="ridge", grouping=grouping, features=features, alpha=alpha, bins=20)
    for grouping in ("subset", "none"):
        for features in ("all", "denoised"):
            for components in (2, 4, 8, 12):
                yield dict(family="pls", grouping=grouping, features=features, components=components, bins=20)
        for components in (4, 8, 16, 32):
            for gamma in (.001, .01):
                yield dict(family="pca_kernel", grouping=grouping, components=components,
                           gamma=gamma, alpha=1., bins=20)
    for grouping in ("subset", "subset_mode"):
        for leaf in (3, 10, 30):
            for depth in (10, None):
                yield dict(family="extra_trees", grouping=grouping, leaf=leaf, depth=depth, bins=20)
    for leaf in (10, 30):
        for components in (7, 15):
            yield dict(family="lightgbm", grouping="subset", components=components,
                       leaf=leaf, n_estimators=350, alpha=10., bins=20)
        for depth in (2, 4):
            yield dict(family="xgboost", grouping="subset", depth=depth,
                       leaf=leaf, n_estimators=500, alpha=10., bins=20)
    for alpha in (.1, 1., 10.):
        yield dict(family="mlp", grouping="subset", alpha=alpha, bins=20)


def diagnostics(frame, y, baseline, mask, targets):
    healthy = frame.loc[mask].copy()
    actual = y[mask]
    prediction = baseline[mask]
    variance = actual.var(axis=0)
    informative = variance > 1e-8
    observed = healthy[["unit", "cycle", "operating_mode"]].reset_index(drop=True)
    values = pd.DataFrame(actual)
    groups = [observed.unit, observed.operating_mode]
    sums = values.groupby(groups).transform("sum").to_numpy()
    counts = values.groupby(groups).transform("count").to_numpy()
    loo = np.where(counts > 1, (sums - actual) / np.maximum(counts-1, 1), prediction)
    ordered = healthy.sort_values(["unit", "operating_mode", "cycle"])
    differences = ordered.groupby(["unit", "operating_mode"])[targets].diff().dropna().to_numpy()
    noise = np.mean(differences ** 2, axis=0) / 2.
    proxy = 1. - noise / np.maximum(variance, 1e-30)
    scores = r2_score(actual, prediction, multioutput="raw_values")
    oracle_scores = r2_score(actual, loo, multioutput="raw_values")
    return {"baseline": BaseTwin._score(actual, prediction),
        "same_engine_mode_lookahead_loo_diagnostic": BaseTwin._score(actual, loo),
        "diagnostic_is_a_deployable_model": False,
        "adjacent_difference_noise_proxy_mean_r2": float(proxy[informative].mean()),
        "noise_proxy_assumption": "Nearby same-engine/mode readings have similar latent levels and independent measurement errors; this is an estimate, not a mathematical performance bound.",
        "per_sensor": {sensor: {"r2": float(scores[i]), "target_variance": float(variance[i]),
            "informative": bool(informative[i]), "noise_variance_proxy": float(noise[i]),
            "noise_proxy_r2": float(proxy[i]), "lookahead_loo_r2": float(oracle_scores[i])}
            for i, sensor in enumerate(targets)}}


def main(args):
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(ROOT / "experiments"):
        raise ValueError("Experiments must stay in Backend/experiments")
    run.mkdir(exist_ok=True)
    config = Config()
    config.mlops.log_to_file = config.mlops.enable_mlflow = False
    plan = load_training_split(config)
    source = joblib.load(Path(args.source_run) / "prepared.joblib")["scaled"]
    fit_ids, validation_ids = plan["engine_roles"]["fit"], plan["engine_roles"]["selection"]
    frame = source.loc[source.unit.isin(fit_ids + validation_ids)].copy()
    bundle = joblib.load(config.paths.models / "startup_profile_twin.pkl")
    targets = bundle["target_names"]
    original_hash = digest(config.paths.models / "startup_profile_twin.pkl")
    protocol_path = run / "protocol.json"
    if not protocol_path.exists():
        write(protocol_path, {"target_r2": .8, "metric_definition_unchanged": True,
            "training_plan_id": plan["plan_id"], "fit_units": fit_ids, "selection_units": validation_ids,
            "source_profile_sha256": original_hash,
            "source_prepared_sha256": digest(Path(args.source_run) / "prepared.joblib"),
            "recipes": list(recipes(args.extended)), "fit_rul_min": 130, "score_rul_min": 125,
            "commissioning_cycles": 30, "max_model_fits": len(list(recipes(args.extended))),
            "extended_commissioning_features": args.extended,
            "confirmation_and_official_closed": True,
            "gate": "Overall R2 increases, RMSE/MAE decrease; each subset RMSE <=1% worse and R2 <=.01 worse",
            "evaluation_status": "retrospective_previously_inspected_fleet"})
    elif json.loads(protocol_path.read_text())["source_profile_sha256"] != original_hash:
        raise ValueError("Serving model changed during the search")
    population = bundle["population"].predict(EnsembleTwin._profile_context(frame, bundle))
    x, ready = EnsembleTwin._profile_features(frame, population, bundle)
    if args.extended:
        x, ready = commissioning_statistics(frame, population, bundle)
    y = frame[targets].to_numpy(float)
    fitting = ready & frame.unit.isin(fit_ids).to_numpy() & frame.RUL.ge(130).to_numpy()
    validation = ready & frame.unit.isin(validation_ids).to_numpy() & frame.RUL.ge(125).to_numpy()
    baseline_model = ColumnModel(bundle["correction"], columns=101)
    baseline_correction = EnsembleTwin._predict_profile_correction(baseline_model, x[validation])
    base = population.copy()
    base[validation] += baseline_correction
    baseline = score(frame, y, base, validation)
    write(run / "baseline.json", baseline)
    diag = diagnostics(frame, y, base, validation, targets)
    write(run / "diagnostics.json", diag)
    print("BASELINE", {k:baseline[k] for k in ("r2", "rmse", "mae", "engines")}, flush=True)
    print("NOISE DIAGNOSTIC", {k:diag[k] for k in ("adjacent_difference_noise_proxy_mean_r2", "same_engine_mode_lookahead_loo_diagnostic")}, flush=True)
    state_path = run / "search.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"trials": [], "best": None}
    done = {trial["index"] for trial in state["trials"]}
    cache = {}
    for index, recipe in enumerate(recipes(args.extended)):
        if index in done:
            continue
        if args.max_trials and index >= args.max_trials:
            break
        started = time.perf_counter()
        spec = dict(recipe)
        bins = spec.pop("bins")
        if bins not in cache:
            cache[bins] = aggregate(frame, x, y-population, fitting, bins)
        xa, ya = cache[bins]
        try:
            model = ProfileRegressor(**spec).fit(xa, ya)
            correction = model.predict(x[validation])
            predictions = population.copy()
            blends = []
            for weight in (.25, .5, .75, 1.):
                predictions[validation] = population[validation] + (1.-weight)*baseline_correction + weight*correction
                measured = score(frame, y, predictions, validation)
                eligible = improves(baseline, measured) and subset_gate(baseline, measured)
                item = {"weight": weight, "eligible": eligible, **measured}
                blends.append(item)
                if eligible and (state["best"] is None or measured["r2"] > state["best"]["metrics"]["r2"]):
                    state["best"] = {"index": index, "spec": recipe, "weight": weight, "metrics": measured}
                    candidate = dict(bundle)
                    candidate.update(correction=BlendedStartupCorrection(baseline_model, model, weight),
                        correction_recipe=f"target_080_{index}", validation_metrics=measured,
                        training_plan_id=plan["plan_id"], fit_unit_ids=fit_ids)
                    if args.extended:
                        candidate["feature_schema"] = "commissioning_statistics_v1"
                    joblib.dump(candidate, run / "candidate_profile.pkl", compress=3)
                    write(run / "selected_recipe.json", {"name": candidate["correction_recipe"],
                        "family": "full_fit_profile", "spec": recipe, "weight": weight,
                        "extended_commissioning_features": args.extended,
                        "training_plan_id": plan["plan_id"], "report": str(run / "evaluation.json")})
            np.savez_compressed(run / f"prediction_{index:03}.npz", correction=correction)
            trial = {"index": index, "spec": recipe, "blends": blends, "seconds": time.perf_counter()-started}
            winner = max(blends, key=lambda item:item["r2"])
            print(index, recipe, "R2", round(winner["r2"],6), "RMSE", round(winner["rmse"],6),
                "eligible", any(b["eligible"] for b in blends), "seconds", round(trial["seconds"],1), flush=True)
        except Exception as exc:
            trial = {"index": index, "spec": recipe, "error": repr(exc), "seconds": time.perf_counter()-started}
            print("FAILED", index, repr(exc), flush=True)
        state["trials"].append(trial)
        write(state_path, state)
    assert digest(config.paths.models / "startup_profile_twin.pkl") == original_hash
    print("DEVELOPMENT BEST", json.dumps(state["best"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(SOURCE))
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--extended", action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
