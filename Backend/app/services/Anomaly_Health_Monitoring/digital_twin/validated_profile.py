from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import (
    ProfileRegressor, ColumnModel, commissioning_statistics,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
    StartupCorrectionRegressor, BlendedStartupCorrection, SensorBlendStartupCorrection, fit_sensor_blend,
)
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope
from app.utils.Anomaly_Health_Monitoring.utils import engine_wise_split
from app.services.Anomaly_Health_Monitoring.digital_twin.forecast_policy import healthy_forecast_guard


def aggregate(frame, features, targets, mask, bins):
    keys = [frame.loc[mask, "unit"].reset_index(drop=True),
            frame.loc[mask, "operating_mode"].reset_index(drop=True),
            (frame.loc[mask, "cycle"] // bins).reset_index(drop=True)]
    return (pd.DataFrame(features[mask]).groupby(keys, sort=True).mean().to_numpy(),
            pd.DataFrame(targets[mask]).groupby(keys, sort=True).mean().to_numpy())


def measured(frame, actual, predictions, mask):
    result = BaseTwin._score(actual[mask], predictions[mask])
    result.update(rows=int(mask.sum()), engines=int(frame.loc[mask, "unit"].nunique()))
    result["per_subset"] = {str(name): BaseTwin._score(actual[mask & frame.subset_id.eq(name).to_numpy()],
        predictions[mask & frame.subset_id.eq(name).to_numpy()]) for name in frame.loc[mask, "subset_id"].unique()}
    return result


def joint_improvement(before, after):
    return (after["r2"] > before["r2"] + 1e-6 and after["rmse"] < before["rmse"] - 1e-6
            and after["mae"] < before["mae"] - 1e-6)


def refit_validated_profile(source, config, bundle, kernel_baseline, recipe):
    from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
    plan = load_training_split(config)
    if plan is None or recipe.get("training_plan_id") != plan["plan_id"]:
        raise ValueError("Validated profile recipe requires its original verified engine partition")
    frame = training_scope(source, config, ("fit", "selection"), required=True)
    fitting = training_scope(source, config, required=True)
    inner_train, inner_validation = engine_wise_split(fitting,
        validation_fraction=config.data.validation_engine_fraction, random_state=config.data.random_state)
    population = bundle["population"].predict(EnsembleTwin._profile_context(frame, bundle))
    compact_bundle = dict(bundle)
    compact_bundle.pop("feature_schema", None)
    compact, ready = EnsembleTwin._profile_features(frame, population, compact_bundle)
    extended, extended_ready = commissioning_statistics(frame, population, compact_bundle)
    np.testing.assert_array_equal(ready, extended_ready)
    actual = frame[bundle["target_names"]].to_numpy(float)
    healthy = ready & frame.RUL.ge(130).to_numpy()
    full_fit = healthy & frame.unit.isin(plan["engine_roles"]["fit"]).to_numpy()
    inner_fit = healthy & frame.unit.isin(inner_train.unit).to_numpy()
    inner_score = ready & frame.unit.isin(inner_validation.unit).to_numpy() & frame.RUL.ge(125).to_numpy()
    selection = ready & frame.unit.isin(plan["engine_roles"]["selection"]).to_numpy() & frame.RUL.ge(125).to_numpy()
    cache = {}
    history = {}

    def required_history(item):
        level = 2 if "causal" in item.get("family", "") else 0
        specifications = [item.get("spec", {})] + [member.get("spec", {}) for member in item.get("members", [])]
        for spec in specifications:
            if spec.get("history") == "trends": level = max(level,5)
            elif spec.get("history") == "fast_modes": level = max(level,4)
            elif spec.get("features") == "multimode": level = max(level,3)
        for member in item.get("members", []):
            if member.get("kind", member.get("source")) == "trends": level = max(level,5)
            elif member.get("source") == "independent": level = max(level,3)
            elif member.get("source") == "multitask": level = max(level,2)
        if "baseline_recipe" in item: level = max(level,required_history(item["baseline_recipe"]))
        return level

    history_ceiling = required_history(recipe)

    def features(level):
        nonlocal compact,extended
        if level >= 2:
            wanted = max(level,history_ceiling)
            if wanted not in history:
                if wanted == 5:
                    from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import trend_features
                    wide,history_ready = trend_features(frame,population,compact_bundle)
                elif wanted == 4:
                    from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features
                    wide,history_ready = fast_mode_features(frame,population,compact_bundle)
                else:
                    from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import prior_sensor_features
                    wide,history_ready = prior_sensor_features(frame,population,compact_bundle,multimode=wanted == 3)
                np.testing.assert_array_equal(ready, history_ready)


                history.clear()
                for available,width in ((2,427),(3,691),(4,952),(5,1141)):
                    if available <= wanted: history[available] = wide[:,:width]
                compact,extended = wide[:,:101],wide[:,:234]
            return history[level]
        return extended if level else compact

    def data(extended_features, bins, full):
        key = (extended_features, bins, full)
        if key not in cache:
            cache[key] = aggregate(frame, extended if extended_features else compact,
                                   actual-population, full_fit if full else inner_fit, bins)
        return cache[key]

    def construct(item):
        family = item.get("family", "legacy")
        if family == "legacy":
            spec = dict(item["spec"])
            bins = int(spec.pop("bins"))
            xa, ya = data(False, bins, False)
            member = (fit_sensor_blend(xa, ya, kernel_baseline, spec) if spec["family"] == "sensor_blend"
                      else StartupCorrectionRegressor(**spec).fit(xa, ya))
            model = BlendedStartupCorrection(kernel_baseline, member, float(item["weight"]))
            before = BaseTwin._score(actual[inner_score], population[inner_score] + kernel_baseline.predict(compact[inner_score]))
            after = BaseTwin._score(actual[inner_score], population[inner_score] + model.predict(compact[inner_score]))
            return (model if joint_improvement(before, after) else kernel_baseline), False
        previous, previous_extended = build(item["baseline_recipe"])
        baseline = ColumnModel(previous, features(previous_extended).shape[1])
        if family == "full_fit_context_sensor_mixture":
            from app.services.Anomaly_Health_Monitoring.digital_twin.context_mixture import ContextSensorMixture
            if not all(hasattr(previous, name) for name in ("baseline", "members")):
                raise ValueError("Context weights require the verified sensor-mixture component layout")
            fallback = previous.fallback_weights if isinstance(previous, ContextSensorMixture) else previous.weights
            model = ContextSensorMixture(previous.baseline, previous.members, item["context_weights"],
                                         fallback, item["grouping"], item.get("age_cut"))
            return model, previous_extended
        if family == "full_fit_forecast_target_mixture":
            members,weights,levels = [],[],[previous_extended]
            rows = ready & frame.unit.isin(plan["engine_roles"]["fit"]).to_numpy()
            for member in item["members"]:
                if member["source"] == "baseline_member":
                    model = previous.members[int(member["index"])]
                    level = previous_extended
                else:
                    spec = dict(member["spec"])
                    kind = member.get("kind",member["source"])
                    level = 5 if kind == "trends" else (4 if spec.get("history") == "fast_modes" else (3 if kind == "independent" else 2))
                    member_rows = rows.copy()
                    if member.get("scope", "all_cycles") == "healthy":
                        member_rows &= frame.RUL.ge(130).to_numpy()
                    elif member.get("scope", "all_cycles") != "all_cycles":
                        raise ValueError("Unknown forecast fitting scope")
                    if kind == "trends":


                        indices = np.flatnonzero(member_rows)
                        history_features = features(level)
                        values = np.empty((len(indices),history_features.shape[1]),dtype=np.float32)
                        for start in range(0,len(indices),2048):
                            batch = indices[start:start+2048]
                            values[start:start+len(batch)] = history_features[batch]
                    else:
                        values = features(level)[member_rows]
                    if kind == "independent":
                        from app.services.Anomaly_Health_Monitoring.digital_twin.independent_forecast import IndependentSensorForecast
                        fitted = IndependentSensorForecast(**spec).fit(values,(actual-population)[member_rows])
                    elif kind == "multitask":
                        from app.services.Anomaly_Health_Monitoring.digital_twin.multitask_forecast import MultitaskSensorForecast
                        fitted = MultitaskSensorForecast(**spec).fit(values,(actual-population)[member_rows],frame.loc[member_rows,"RUL"].to_numpy())
                    elif kind == "trends":
                        from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import TrendDecoder
                        fitted = TrendDecoder(**spec).fit(values,(actual-population)[member_rows])
                    else: raise ValueError("Unknown frozen forecast decoder")
                    model = ColumnModel(fitted,values.shape[1])
                members.append(model); weights.append(member["weights"]); levels.append(level)
            if not members: raise ValueError("Empty forecast target mixture")
            return SensorBlendStartupCorrection(baseline,members,weights),max(levels)
        if family in ("full_fit_causal_forecast", "full_fit_causal_sensor_mixture"):
            from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import CausalSensorRegressor
            def fit_forecast(specification):
                spec = dict(specification)
                scope = spec.pop("scope")
                level = 3 if spec.get("features") == "multimode" else 2
                history_features = features(level)
                if scope not in ("healthy", "all_cycles"):
                    raise ValueError("Unknown forecast fitting scope")
                rows = ready & frame.unit.isin(plan["engine_roles"]["fit"]).to_numpy()
                if scope == "healthy":
                    rows &= frame.RUL.ge(130).to_numpy()


                model = CausalSensorRegressor(**spec).fit(history_features[rows],(actual-population)[rows])
                return ColumnModel(model,history_features.shape[1]),level
            if family == "full_fit_causal_forecast":
                member,level = fit_forecast(item["spec"])
                return BlendedStartupCorrection(baseline,member,float(item["weight"])),max(previous_extended,level)
            fitted = [fit_forecast(member["spec"]) for member in item["members"]]
            return SensorBlendStartupCorrection(baseline,[model for model,level in fitted],
                [member["weights"] for member in item["members"]]),max(previous_extended,max(level for model,level in fitted))
        if family == "full_fit_profile":
            spec = dict(item["spec"])
            bins = int(spec.pop("bins"))
            is_extended = bool(item.get("extended_commissioning_features", False))
            xa, ya = data(is_extended, bins, True)
            member = ColumnModel(ProfileRegressor(**spec).fit(xa, ya), extended.shape[1] if is_extended else compact.shape[1])
            return BlendedStartupCorrection(baseline, member, float(item["weight"])), previous_extended or is_extended
        if family == "full_fit_sensor_mixture":
            members, weights = [], []
            any_extended = previous_extended
            for member in item["members"]:
                spec = dict(member["spec"])
                bins = int(spec.pop("bins"))
                is_extended = bool(member["extended"])
                xa, ya = data(is_extended, bins, True)
                model = ProfileRegressor(**spec).fit(xa, ya)
                members.append(ColumnModel(model, extended.shape[1] if is_extended else compact.shape[1]))
                weights.append(member["weights"])
                any_extended = max(any_extended, int(is_extended))
            return SensorBlendStartupCorrection(baseline, members, weights), any_extended
        raise ValueError("Unsupported validated profile recipe")

    model_cache = {}
    def build(item):
        key = id(item)
        if key not in model_cache:
            model_cache[key] = construct(item)
        return model_cache[key]

    previous, previous_extended = build(recipe["baseline_recipe"])
    model, use_extended = build(recipe)
    previous_prediction, candidate_prediction = population.copy(), population.copy()
    full_selection = ready & frame.unit.isin(plan["engine_roles"]["selection"]).to_numpy()
    full_objective = recipe.get("objective") == "full_trajectory_forecasts"
    scoring = full_selection if full_objective else selection
    previous_prediction[scoring] += previous.predict(features(previous_extended)[scoring])
    candidate_prediction[scoring] += model.predict(features(use_extended)[scoring])
    before = measured(frame, actual, previous_prediction, selection)
    after = measured(frame, actual, candidate_prediction, selection)
    accepted = joint_improvement(before, after) and all(
        after["per_subset"][name]["rmse"] <= value["rmse"] * 1.01
        and after["per_subset"][name]["r2"] >= value["r2"] - .01
        for name, value in before["per_subset"].items())
    full_before,full_after = None,None
    if full_objective:
        full_before = measured(frame,actual,previous_prediction,full_selection)
        full_after = measured(frame,actual,candidate_prediction,full_selection)
        accepted = (joint_improvement(full_before,full_after)
            and healthy_forecast_guard(before,after,recipe.get("healthy_guard"))
            and all(candidate["per_subset"][name]["rmse"]<=value["rmse"]*1.01
                    and candidate["per_subset"][name]["r2"]>=value["r2"]-.01
                for baseline_score,candidate in ((before,after),(full_before,full_after))
                for name,value in baseline_score["per_subset"].items()))
    selected_level = use_extended if accepted else previous_extended
    return {"accepted":accepted, "model":model if accepted else previous,
        "feature_schema":("causal_trend_history_v1" if selected_level == 5 else
                          "causal_fast_modes_v1" if selected_level == 4 else
                          "causal_multimode_history_v1" if selected_level == 3 else
                          "causal_sensor_history_v1" if selected_level == 2 else
                          "commissioning_statistics_v1" if selected_level else None),
        "prediction_role":"adaptive_sensor_forecast" if selected_level >= 2 else "frozen_commissioning_expectation",
        "baseline_metrics":before, "metrics":after if accepted else before,
        "full_trajectory_reference_metrics":full_before,
        "full_trajectory_metrics":full_after if accepted else full_before,
        "fit_unit_ids":sorted(map(int, frame.loc[full_fit,"unit"].unique())),
        "selection_unit_ids":plan["engine_roles"]["selection"]}
