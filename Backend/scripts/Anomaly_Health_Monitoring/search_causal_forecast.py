from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.causal_forecast_models import CausalSensorRegressor, prior_sensor_features
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT, SOURCE, digest, score, improves, subset_gate, write
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import ColumnModel
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import BlendedStartupCorrection
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

RUN=ROOT/"experiments/Anomaly_Health_Monitoring/run_20261016_causal_forecast"

def recipes(nonlinear=False,multimode=False):
    if multimode:
        for scope in ("healthy","all_cycles"):
            for grouping in ("subset_mode","subset"):
                for alpha in (1.,100.,10000.):
                    yield dict(scope=scope,grouping=grouping,features="multimode",family="ridge",alpha=alpha)
        return
    if nonlinear:
        for scope in ("healthy", "all_cycles"):
            for grouping in ("subset_mode", "subset"):
                for leaf in (40,120):
                    for depth in (10,None):
                        yield dict(scope=scope,grouping=grouping,features="combined",family="extra_trees",leaf=leaf,depth=depth)
            for gamma in (.001,.01):
                for alpha in (1.,100.):
                    yield dict(scope=scope,grouping="subset_mode",features="combined",family="kernel",gamma=gamma,alpha=alpha)
            for grouping in ("subset_mode","subset"):
                for alpha in (1.,100.,10000.):
                    yield dict(scope=scope,grouping=grouping,features="combined",family="ridge",alpha=alpha)
        return
    for scope in ("healthy", "all_cycles"):
        for grouping in ("subset_mode", "subset"):
            for features in ("all", "smooth", "global", "local"):
                for alpha in (1.,100.,10000.):
                    yield dict(scope=scope, grouping=grouping, features=features, alpha=alpha)

def assert_causal(frame,bundle,multimode=False):
    units=frame.groupby("subset_id").unit.first().tolist()
    sample=frame.loc[frame.unit.isin(units)&frame.cycle.le(60)].copy()
    pop=bundle["population"].predict(EnsembleTwin._profile_context(sample,bundle))
    features,ready=prior_sensor_features(sample,pop,bundle,multimode=multimode)
    changed=sample.copy()
    changed.loc[changed.cycle.ge(45),bundle["target_names"]]+=1000.
    changed_features,_=prior_sensor_features(changed,pop,bundle,multimode=multimode)
    np.testing.assert_allclose(features[sample.cycle.le(45)],changed_features[sample.cycle.le(45)],rtol=0,atol=1e-10)
    assert np.any(np.abs(features[sample.cycle.gt(45)]-changed_features[sample.cycle.gt(45)])>1.)
    shuffled=sample.sample(frac=1.,random_state=42)
    sp=bundle["population"].predict(EnsembleTwin._profile_context(shuffled,bundle))
    sx,sr=prior_sensor_features(shuffled,sp,bundle,multimode=multimode)
    positions=sample.index.get_indexer(shuffled.index)
    np.testing.assert_allclose(features[positions],sx,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(ready[positions],sr)

def main(args):
    run=Path(args.run_dir).resolve()
    if not run.is_relative_to(ROOT/"experiments"):
        raise ValueError("Keep experiments within Backend/experiments")
    run.mkdir(exist_ok=True)
    if (run/"final/model_lock.json").exists():
        raise ValueError("Candidate already locked; no confirmation retuning")
    config=Config()
    config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    fit_ids,selection_ids=plan["engine_roles"]["fit"],plan["engine_roles"]["selection"]
    bundle=joblib.load(config.paths.models/"startup_profile_twin.pkl")
    frame=joblib.load(SOURCE/"prepared.joblib")["scaled"]
    frame=frame.loc[frame.unit.isin(fit_ids+selection_ids)].copy()
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    protocol={"target_r2":.8,"prediction_role":"adaptive_sensor_forecast",
        "user_authorized_prior_readings":True,"training_plan_id":plan["plan_id"],
        "fit_units":fit_ids,"selection_units":selection_ids,"protected_models":protected,
        "recipes":list(recipes(args.nonlinear,args.multimode)),"score_rul_min":125,"commissioning_cycles":30,
        "individual_row_training":True,"raw_targets_and_metric_unchanged":True,
        "confirmation_and_official_closed":True,"monitoring_reference":"unchanged_frozen_healthy_expectation",
        "inputs":"current operating context; first30 commissioning statistics; strictly past sensor EWMs and lags"}
    if (run/"protocol.json").exists():
        assert json.loads((run/"protocol.json").read_text())["protected_models"]==protected
    else:
        write(run/"protocol.json",protocol)
    assert_causal(frame,bundle,args.multimode)
    population=bundle["population"].predict(EnsembleTwin._profile_context(frame,bundle))
    old_x,ready=EnsembleTwin._profile_features(frame,population,bundle)
    x,new_ready=prior_sensor_features(frame,population,bundle,multimode=args.multimode)
    np.testing.assert_array_equal(ready,new_ready)
    actual=frame[bundle["target_names"]].to_numpy(float)
    fitting=ready&frame.unit.isin(fit_ids).to_numpy()
    validation=ready&frame.unit.isin(selection_ids).to_numpy()&frame.RUL.ge(125).to_numpy()
    old=EnsembleTwin._predict_profile_correction(bundle["correction"],old_x[validation])
    baseline_prediction=population.copy()
    baseline_prediction[validation]+=old
    baseline=score(frame,actual,baseline_prediction,validation)
    write(run/"baseline.json",baseline)
    np.savez_compressed(run/"validation_data.npz",actual=actual[validation],population=population[validation],
        baseline_correction=old,units=frame.loc[validation,"unit"].to_numpy(),subsets=frame.loc[validation,"subset_id"].to_numpy(dtype=str))
    print("CAUSALITY CHECKS PASS; BASELINE",baseline["r2"],"FEATURES",x.shape,flush=True)
    path=run/"search.json"
    state=json.loads(path.read_text()) if path.exists() else {"trials":[],"best":None}
    completed={t["index"] for t in state["trials"]}
    for index,spec in enumerate(recipes(args.nonlinear,args.multimode)):
        if index in completed:
            continue
        started=time.perf_counter()
        parameters=dict(spec)
        scope=parameters.pop("scope")
        fit=fitting&(frame.RUL.ge(130).to_numpy() if scope=="healthy" else True)
        model=CausalSensorRegressor(**parameters).fit(x[fit],(actual-population)[fit])
        correction=model.predict(x[validation])
        predicted=population.copy()
        blends=[]
        for weight in (.25,.5,.75,1.):
            predicted[validation]=population[validation]+(1.-weight)*old+weight*correction
            metrics=score(frame,actual,predicted,validation)
            eligible=improves(baseline,metrics) and subset_gate(baseline,metrics)
            blends.append({"weight":weight,"metrics":metrics,"eligible":eligible})
            if eligible and (state["best"] is None or metrics["r2"]>state["best"]["metrics"]["r2"]):
                state["best"]={"index":index,"spec":spec,"weight":weight,"metrics":metrics}
        np.savez_compressed(run/f"prediction_{index:03}.npz",correction=correction)
        trial={"index":index,"spec":spec,"blends":blends,"seconds":time.perf_counter()-started}
        state["trials"].append(trial)
        write(path,state)
        winner=max(blends,key=lambda b:b["metrics"]["r2"])
        print(index,spec,"R2",round(winner["metrics"]["r2"],6),"eligible",any(b["eligible"] for b in blends),"seconds",round(trial["seconds"],1),flush=True)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    if state["best"]:
        best=state["best"]
        parameters=dict(best["spec"])
        scope=parameters.pop("scope")
        fit=fitting&(frame.RUL.ge(130).to_numpy() if scope=="healthy" else True)
        model=CausalSensorRegressor(**parameters).fit(x[fit],(actual-population)[fit])
        candidate=dict(bundle)
        candidate.update(correction=BlendedStartupCorrection(ColumnModel(bundle["correction"],old_x.shape[1]),model,best["weight"]),
            feature_schema="causal_multimode_history_v1" if args.multimode else "causal_sensor_history_v1",correction_recipe=f"causal_sensor_forecast_{best['index']}",
            prediction_role="adaptive_sensor_forecast",validation_metrics=best["metrics"],training_plan_id=plan["plan_id"])
        joblib.dump(candidate,run/"candidate_profile.pkl",compress=3)
        write(run/"selected_recipe.json",{"name":candidate["correction_recipe"],"family":"full_fit_causal_forecast",
            "spec":best["spec"],"weight":best["weight"],"training_plan_id":plan["plan_id"],"prediction_role":"adaptive_sensor_forecast",
            "feature_schema":candidate["feature_schema"],
            "baseline_recipe":json.loads((config.paths.models/"startup_profile_recipe.json").read_text())})
    print("BEST DEVELOPMENT",json.dumps(state["best"]),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",default=str(RUN))
    parser.add_argument("--nonlinear",action="store_true")
    parser.add_argument("--multimode",action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
