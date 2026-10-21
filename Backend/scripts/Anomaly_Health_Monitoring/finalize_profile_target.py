from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.advanced_profile_models import (
    ProfileRegressor, ColumnModel, commissioning_statistics,
)
from scripts.Anomaly_Health_Monitoring.search_profile_target import RUN, recipes
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import (
    ROOT, SOURCE, aggregate, digest, improves, score, subset_gate, write,
)
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
    BlendedStartupCorrection, SensorBlendStartupCorrection,
)
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope


def fit_weights(actual, baseline, candidates, penalty):
    variance = actual.var(axis=0)
    weights = np.zeros((len(candidates), actual.shape[1]))
    for sensor in np.flatnonzero(variance > 1e-8):
        differences = np.column_stack([item[:,sensor] - baseline[:,sensor] for item in candidates])
        target = actual[:,sensor] - baseline[:,sensor]
        hessian = differences.T @ differences / len(actual) / variance[sensor]
        linear = differences.T @ target / len(actual) / variance[sensor]
        solution = minimize(lambda w: w @ hessian @ w - 2*linear @ w + penalty*(w@w),
            np.zeros(len(candidates)), jac=lambda w:2*hessian@w - 2*linear + 2*penalty*w,
            bounds=[(0.,1.)]*len(candidates), constraints={"type":"ineq", "fun":lambda w:1.-w.sum(),
                "jac":lambda w:-np.ones_like(w)}, method="SLSQP", options={"ftol":1e-10, "maxiter":150})
        if not solution.success:
            raise ValueError(f"Mixture optimization failed: {solution.message}")
        weights[:,sensor] = np.maximum(solution.x, 0.)
    return weights


def mixture(baseline, candidates, weights):
    output = baseline.copy()
    for candidate, weight in zip(candidates, weights):
        output += (candidate - baseline) * weight
    return output


def main(args):
    run = Path(args.run_dir).resolve()
    final = run / "final"
    if (final / "model_lock.json").exists():
        raise ValueError("Candidate is already frozen; do not retune using its confirmation results")
    sources = [(run, False), (Path(str(run)+"_extended"), True)]
    families, total, best = {}, 0, None
    for source, extended in sources:
        state = json.loads((source / "search.json").read_text())
        if len(state["trials"]) != len(list(recipes(extended))):
            raise ValueError(f"Search is still running: {source.name}")
        total += len(state["trials"])
        if state["best"] and (best is None or state["best"]["metrics"]["r2"] > best["item"]["metrics"]["r2"]):
            best = {"source": source, "extended": extended, "item": state["best"]}
        for trial in state["trials"]:
            if "blends" not in trial:
                continue
            measured = max(trial["blends"], key=lambda item:item["r2"])

            family = (trial["spec"]["family"], extended)
            if family not in families or measured["r2"] > families[family]["score"]:
                families[family] = {"source": source, "extended": extended, "trial": trial, "score": measured["r2"]}
    final.mkdir(exist_ok=False)
    config = Config()
    config.mlops.log_to_file = config.mlops.enable_mlflow = False
    plan = load_training_split(config)
    baseline_bundle = joblib.load(config.paths.models / "startup_profile_twin.pkl")
    baseline_recipe = json.loads((config.paths.models / "startup_profile_recipe.json").read_text())
    fitting_ids, validation_ids = plan["engine_roles"]["fit"], plan["engine_roles"]["selection"]
    prepared = joblib.load(SOURCE / "prepared.joblib")["scaled"]
    frame = prepared.loc[prepared.unit.isin(fitting_ids + validation_ids)].copy()
    population = baseline_bundle["population"].predict(EnsembleTwin._profile_context(frame, baseline_bundle))
    x, ready = EnsembleTwin._profile_features(frame, population, baseline_bundle)
    extended_x, extended_ready = commissioning_statistics(frame, population, baseline_bundle)
    np.testing.assert_array_equal(ready, extended_ready)
    fitting = ready & frame.unit.isin(fitting_ids).to_numpy() & frame.RUL.ge(130).to_numpy()
    validation = ready & frame.unit.isin(validation_ids).to_numpy() & frame.RUL.ge(125).to_numpy()
    actual = frame[baseline_bundle["target_names"]].to_numpy(float)
    old_correction = EnsembleTwin._predict_profile_correction(baseline_bundle["correction"], x[validation])
    old_prediction = population.copy()
    old_prediction[validation] += old_correction
    baseline_metrics = score(frame, actual, old_prediction, validation)
    pool = sorted(families.values(), key=lambda item:item["score"], reverse=True)[:6]
    predictions = [population[validation]+np.load(item["source"]/f"prediction_{item['trial']['index']:03}.npz")["correction"] for item in pool]
    units = frame.loc[validation].groupby("unit").subset_id.first().sort_index()
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261017).split(units.index, units))
    trials = []
    selected_penalty = None
    for penalty in (.005, .05, .5, 5.):
        oof = old_prediction[validation].copy()
        for train_index, held_index in splits:
            fit_rows = frame.loc[validation].unit.isin(units.index[train_index]).to_numpy()
            held_rows = frame.loc[validation].unit.isin(units.index[held_index]).to_numpy()
            weights = fit_weights(actual[validation][fit_rows], old_prediction[validation][fit_rows],
                                  [p[fit_rows] for p in predictions], penalty)
            oof[held_rows] = mixture(old_prediction[validation][held_rows], [p[held_rows] for p in predictions], weights)
        result_prediction = population.copy()
        result_prediction[validation] = oof
        metrics = score(frame, actual, result_prediction, validation)
        eligible = improves(baseline_metrics, metrics) and subset_gate(baseline_metrics, metrics)
        trials.append({"penalty": penalty, "metrics": metrics, "eligible": eligible})
        print("CROSSFIT MIXTURE", penalty, metrics["r2"], metrics["rmse"], eligible, flush=True)
        if eligible and (selected_penalty is None or metrics["r2"] > selected_penalty["metrics"]["r2"]):
            selected_penalty = trials[-1]
    write(final / "mixture_crossfit.json", trials)
    chosen = dict(best["item"])
    candidate_bundle = joblib.load(best["source"] / "candidate_profile.pkl")
    recipe = json.loads((best["source"] / "selected_recipe.json").read_text())
    selected_as = "best_single_recipe"
    if selected_penalty is not None and selected_penalty["metrics"]["r2"] > chosen["metrics"]["r2"]:
        weights = fit_weights(actual[validation], old_prediction[validation], predictions, selected_penalty["penalty"])
        models, specifications = [], []
        cache = {}
        for member, sensor_weights in zip(pool, weights):
            if sensor_weights.max() <= 1e-8:
                continue
            spec = dict(member["trial"]["spec"])
            bins = spec.pop("bins")
            member_x = extended_x if member["extended"] else x
            cache_key = (member["extended"], bins)
            if cache_key not in cache:
                cache[cache_key] = aggregate(frame, member_x, actual-population, fitting, bins)
            xa, ya = cache[cache_key]
            model = ProfileRegressor(**spec).fit(xa, ya)
            models.append(ColumnModel(model, member_x.shape[1]))
            specifications.append({"spec": member["trial"]["spec"], "extended": member["extended"],
                                   "weights": sensor_weights.tolist()})
        mixture_weights = [item["weights"] for item in specifications]
        mixture_model = SensorBlendStartupCorrection(ColumnModel(baseline_bundle["correction"], 101), models, mixture_weights)
        extended_needed = any(item["extended"] for item in specifications)
        features = extended_x if extended_needed else x
        candidate_prediction = population.copy()
        candidate_prediction[validation] += mixture_model.predict(features[validation])
        metrics = score(frame, actual, candidate_prediction, validation)
        if improves(baseline_metrics, metrics) and subset_gate(baseline_metrics, metrics):
            candidate_bundle = dict(baseline_bundle)
            candidate_bundle.update(correction=mixture_model, correction_recipe="target_080_crossfit_sensor_mixture",
                                    validation_metrics=metrics, training_plan_id=plan["plan_id"], fit_unit_ids=fitting_ids)
            if extended_needed:
                candidate_bundle["feature_schema"] = "commissioning_statistics_v1"
            recipe = {"name": candidate_bundle["correction_recipe"], "family": "full_fit_sensor_mixture",
                "members": specifications, "penalty": selected_penalty["penalty"],
                "extended_commissioning_features": extended_needed,
                "training_plan_id": plan["plan_id"]}
            chosen = {"metrics": metrics, "crossfit_metrics": selected_penalty["metrics"]}
            selected_as = "crossfit_sensor_mixture"
    recipe["baseline_recipe"] = baseline_recipe
    recipe["report"] = str(final / "evaluation.json")
    joblib.dump(candidate_bundle, final / "candidate_profile.pkl", compress=3)
    write(final / "selected_recipe.json", recipe)
    protected = {str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    write(final / "protocol.json", {"protected_models": protected, "training_plan_id":plan["plan_id"],
        "fit_units":fitting_ids, "selection_units":validation_ids, "model_recipes_tested":total,
        "original_metric_retained": True, "target_r2": .8, "selection":selected_as,
        "model_selection_uses_confirmation_or_official":False})
    write(final / "model_lock.json", {"candidate_sha256":digest(final/"candidate_profile.pkl"),
        "locked_before_confirmation":True, "locked_before_official":True, "selection":chosen})
    confirmation = training_scope(prepared, config, ("confirmation",), required=True)
    cp = baseline_bundle["population"].predict(EnsembleTwin._profile_context(confirmation, baseline_bundle))
    cx, cr = EnsembleTwin._profile_features(confirmation, cp, baseline_bundle)
    if candidate_bundle.get("feature_schema"):
        candidate_x, _ = commissioning_statistics(confirmation, cp, baseline_bundle)
    else:
        candidate_x = cx
    mask = cr & confirmation.RUL.ge(125).to_numpy()
    cy = confirmation[baseline_bundle["target_names"]].to_numpy(float)
    old, new = cp.copy(), cp.copy()
    old[mask] += EnsembleTwin._predict_profile_correction(baseline_bundle["correction"], cx[mask])
    new[mask] += EnsembleTwin._predict_profile_correction(candidate_bundle["correction"], candidate_x[mask])
    confirmed = {"baseline": score(confirmation, cy, old, mask), "candidate": score(confirmation, cy, new, mask)}
    accepted = improves(confirmed["baseline"], confirmed["candidate"]) and subset_gate(confirmed["baseline"], confirmed["candidate"])
    result = {"baseline":{"selection":baseline_metrics}, "selected":chosen,
        "confirmation":confirmed, "accepted":accepted, "production_changed":False,
        "model_recipes_tested":total, "target_r2":.8,
        "target_met_on_confirmation":confirmed["candidate"]["r2"] >= .8,
        "confirmation_used_for_retuning":False}
    write(final / "evaluation.json", result)
    print("FROZEN CONFIRMATION", json.dumps(confirmed), flush=True)
    print("ACCEPTED FOR VERIFICATION", accepted, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(RUN))
    with threadpool_limits(limits=4):
        main(parser.parse_args())
