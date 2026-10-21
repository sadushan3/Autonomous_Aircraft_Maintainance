from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
    BlendedStartupCorrection, StartupCorrectionRegressor,
)
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope
from app.utils.Anomaly_Health_Monitoring.utils import engine_wise_split

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261014_leakage_fix"
RUN = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261016_startup_optimization"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame, actual, predicted, mask):
    result = BaseTwin._score(actual[mask], predicted[mask])
    result.update(rows=int(mask.sum()), engines=int(frame.loc[mask, "unit"].nunique()))
    result["per_subset"] = {}
    for name in sorted(frame.loc[mask, "subset_id"].unique()):
        selected = mask & frame.subset_id.eq(name).to_numpy()
        result["per_subset"][name] = BaseTwin._score(actual[selected], predicted[selected])
    return result


def improves(before, after):
    return (after["r2"] > before["r2"] + 1e-6
            and after["rmse"] < before["rmse"] - 1e-6
            and after["mae"] < before["mae"] - 1e-6)


def subset_gate(before, after):

    return all(after["per_subset"][s]["rmse"] <= item["rmse"] * 1.01
               and after["per_subset"][s]["r2"] >= item["r2"] - .01
               for s, item in before["per_subset"].items())


def recipes():
    for bins in (20, 40):
        for alpha in (.1, 1., 10.):
            for gamma in (.0003, .001, .003):
                yield {"family": "kernel", "grouped": True, "alpha": alpha, "gamma": gamma, "bins": bins}
    for grouped in (False, True):
        for depth in (8, 12, 20):
            for gamma in (.001, .01):
                yield {"family": "pca_kernel", "grouped": grouped, "depth": depth,
                       "alpha": 1., "gamma": gamma, "bins": 20}
    for grouped in (False, True):
        for features in ("compact", "quadratic", "spline", "interactions"):
            for alpha in (100., 1000.):
                yield {"family": "ridge", "features": features, "grouped": grouped,
                       "alpha": alpha, "bins": 20}
    for depth in (5, 10, None):
        yield {"family": "extra_trees", "grouped": True, "depth": depth,
               "leaf": 10, "n_estimators": 160, "bins": 20}


def aggregate(frame, features, targets, mask, bins):
    keys = [frame.loc[mask, "unit"].reset_index(drop=True),
            frame.loc[mask, "operating_mode"].reset_index(drop=True),
            (frame.loc[mask, "cycle"] // bins).reset_index(drop=True)]
    return (pd.DataFrame(features[mask]).groupby(keys, sort=True).mean().to_numpy(),
            pd.DataFrame(targets[mask]).groupby(keys, sort=True).mean().to_numpy())


def main(args):
    run, source = Path(args.run_dir).resolve(), Path(args.source_run).resolve()
    if not run.is_relative_to(ROOT / "experiments"):
        raise ValueError("Keep experiments under Backend/experiments")
    run.mkdir(parents=True, exist_ok=False)
    config = Config()
    config.mlops.log_to_file = config.mlops.enable_mlflow = False
    plan = load_training_split(config)
    if plan is None:
        raise ValueError("A verified shared engine split is required")
    bundle = joblib.load(config.paths.models / "startup_profile_twin.pkl")
    prepared = joblib.load(source / "prepared.joblib")["scaled"]
    fitting = training_scope(prepared, config, required=True)
    inner_train, inner_validation = engine_wise_split(fitting,
        validation_fraction=config.data.validation_engine_fraction,
        random_state=config.data.random_state)
    fit_ids = sorted(map(int, inner_train.unit.unique()))
    protected = {str(p.relative_to(ROOT)): digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    write(run / "protocol.json", {"training_plan_id": plan["plan_id"], "fit_units": fit_ids,
        "selection_units": plan["engine_roles"]["selection"],
        "confirmation_units": plan["engine_roles"]["confirmation"],
        "source_profile_sha256": digest(config.paths.models / "startup_profile_twin.pkl"),
        "healthy_rul_min": 125, "fit_rul_min": 130, "commissioning_cycles": 30,
        "recipes": list(recipes()), "blend_weights": [.25, .5, .75, 1.],
        "gate": "R2 increases; RMSE and MAE decrease on inner and shared development; subset RMSE <=1% worse, R2 <=.01 worse; one frozen candidate then confirmation",
        "confirmation_or_official_used_for_tuning": False,
        "evaluation_status": "retrospective_previously_inspected_fleet", "protected_models": protected})

    development = prepared.loc[prepared.unit.isin(set(fitting.unit) | set(plan["engine_roles"]["selection"]))].copy()
    y = development[bundle["target_names"]].to_numpy(float)
    population = bundle["population"].predict(EnsembleTwin._profile_context(development, bundle))
    x, ready = EnsembleTwin._profile_features(development, population, bundle)
    fit = ready & development.unit.isin(fit_ids).to_numpy() & development.RUL.ge(130).to_numpy()
    inner = ready & development.unit.isin(inner_validation.unit).to_numpy() & development.RUL.ge(125).to_numpy()
    selection = ready & development.unit.isin(plan["engine_roles"]["selection"]).to_numpy() & development.RUL.ge(125).to_numpy()
    evaluated = inner | selection

    correction = np.zeros_like(y)
    correction[evaluated] = EnsembleTwin._predict_profile_correction(bundle["correction"], x[evaluated])
    baseline_predictions = population + correction
    baseline = {"inner": score(development, y, baseline_predictions, inner),
                "selection": score(development, y, baseline_predictions, selection)}
    write(run / "baseline.json", baseline)
    print("BASELINE", json.dumps(baseline), flush=True)
    cache, trials, best = {}, [], None
    for index, recipe in enumerate(recipes()):
        started = time.perf_counter()
        spec = dict(recipe)
        bins = spec.pop("bins")
        if bins not in cache:
            cache[bins] = aggregate(development, x, y - population, fit, bins)
        xa, ya = cache[bins]
        model = StartupCorrectionRegressor(**spec).fit(xa, ya)
        candidate_correction = np.zeros_like(y)
        candidate_correction[evaluated] = model.predict(x[evaluated])
        blends = []
        for weight in (.25, .5, .75, 1.):
            predicted = population + (1. - weight) * correction + weight * candidate_correction
            metrics = {"inner": score(development, y, predicted, inner),
                       "selection": score(development, y, predicted, selection)}
            accepted = (all(improves(baseline[k], metrics[k]) for k in baseline)
                        and subset_gate(baseline["selection"], metrics["selection"]))
            blends.append({"weight": weight, "eligible": accepted, **metrics})
            if accepted and (best is None or metrics["selection"]["r2"] > best["selection"]["r2"]):
                best = {"index": index, "spec": recipe, "weight": weight, **metrics}
                selected = dict(bundle)
                selected["correction"] = BlendedStartupCorrection(bundle["correction"], model, weight)
                selected.update(validation_metrics=metrics["inner"],
                    development_validation_metrics=metrics["selection"],
                    correction_recipe=f"development_optimized_{index}",
                    validated_correction_spec=recipe, validated_blend_weight=weight,
                    training_plan_id=plan["plan_id"])
                joblib.dump(selected, run / "candidate_profile.pkl", compress=3)
                write(run / "selected_recipe.json", {"name": selected["correction_recipe"],
                    "spec": recipe, "weight": weight, "report": str(run / "evaluation.json")})
        trial = {"index": index, "spec": recipe, "blends": blends,
                 "fit_seconds": time.perf_counter() - started}
        trials.append(trial)
        write(run / "search.json", {"trials": trials, "best": best})
        winner = max(blends, key=lambda b: b["selection"]["r2"])
        print(index, recipe, "R2", round(winner["selection"]["r2"], 6),
              "RMSE", round(winner["selection"]["rmse"], 6),
              "eligible", any(b["eligible"] for b in blends),
              "seconds", round(trial["fit_seconds"], 1), flush=True)
    if best is None:
        write(run / "decision.json", {"accepted": False, "reason": "No development candidate passes every gate"})
        print("NO CHANGE: no candidate passed", flush=True)
        return
    write(run / "model_lock.json", {"selection": best, "candidate_sha256": digest(run / "candidate_profile.pkl"),
        "locked_before_confirmation": True, "locked_before_official": True})
    candidate = joblib.load(run / "candidate_profile.pkl")
    confirmation = training_scope(prepared, config, ("confirmation",), required=True)
    cy = confirmation[bundle["target_names"]].to_numpy(float)
    cp = bundle["population"].predict(EnsembleTwin._profile_context(confirmation, bundle))
    cx, cr = EnsembleTwin._profile_features(confirmation, cp, bundle)
    cm = cr & confirmation.RUL.ge(125).to_numpy()
    old = cp.copy()
    new = cp.copy()
    old[cm] += EnsembleTwin._predict_profile_correction(bundle["correction"], cx[cm])
    new[cm] += EnsembleTwin._predict_profile_correction(candidate["correction"], cx[cm])
    confirmed = {"baseline": score(confirmation, cy, old, cm), "candidate": score(confirmation, cy, new, cm)}
    accepted = improves(confirmed["baseline"], confirmed["candidate"]) and subset_gate(confirmed["baseline"], confirmed["candidate"])
    result = {"baseline": baseline, "selected": best, "confirmation": confirmed,
              "accepted": accepted, "confirmation_used_for_retuning": False,
              "production_changed": False}
    write(run / "evaluation.json", result)
    write(run / "decision.json", {"accepted": accepted, "candidate_sha256": digest(run / "candidate_profile.pkl"),
        "reason": "Development and frozen confirmation pass" if accepted else "Frozen confirmation rejected; retain active model"})
    assert protected == {str(p.relative_to(ROOT)): digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    print("CONFIRMATION", json.dumps(confirmed), flush=True)
    print("ACCEPTED FOR VERIFICATION", accepted, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=str(SOURCE))
    parser.add_argument("--run-dir", default=str(RUN))
    with threadpool_limits(limits=4):
        main(parser.parse_args())
