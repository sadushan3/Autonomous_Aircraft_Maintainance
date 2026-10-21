from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.Anomaly_Health_Monitoring.condition_profile_models import ConditionProfileRegressor,multimode_features
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,digest,aggregate,score,improves,subset_gate,write
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import ColumnModel
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import BlendedStartupCorrection
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

RUN=ROOT/"experiments/Anomaly_Health_Monitoring/run_20261016_condition_state"


def recipes(clusters=False):
    if clusters:
        for count in (2,3):
            for rank in (2,4,8):
                for noise_factor in (1.,4.):
                    for alpha in (1.,100.):
                        yield dict(rank=rank,noise_factor=noise_factor,decoder="ridge",alpha=alpha,
                                   age_features="quadratic",grouping="subset_mode",clusters=count)
        return
    for rank in (1,2,4,8):
        for noise_factor in (.25,1.,4.):
            for alpha in (1.,100.):
                for age_features in ("quadratic","spline"):
                    yield dict(rank=rank,noise_factor=noise_factor,decoder="ridge",alpha=alpha,
                               age_features=age_features,grouping="subset_mode")
    for rank in (2,4,8):
        for noise_factor in (.5,2.):
            for alpha in (.1,1.):
                for gamma in (.01,.1):
                    yield dict(rank=rank,noise_factor=noise_factor,decoder="kernel",alpha=alpha,
                               gamma=gamma,age_features="quadratic",grouping="subset_mode")
        for alpha in (1.,100.):
            yield dict(rank=rank,noise_factor=1.,decoder="ridge",alpha=alpha,age_features="spline",
                       grouping="subset_mode",robust=True)
            yield dict(rank=rank,noise_factor=1.,decoder="ridge",alpha=alpha,age_features="spline",
                       grouping="subset")


def main(args):
    run=Path(args.run_dir).resolve()
    if not run.is_relative_to(ROOT/"experiments"):
        raise ValueError("Keep experiments within Backend/experiments")
    run.mkdir(exist_ok=True)
    if (run/"model_lock.json").exists():
        raise ValueError("Candidate already frozen; confirmation may not be reused for tuning")
    config=Config()
    config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    fitting_ids=plan["engine_roles"]["fit"]
    selection_ids=plan["engine_roles"]["selection"]
    bundle=joblib.load(config.paths.models/"startup_profile_twin.pkl")
    source=joblib.load(SOURCE/"prepared.joblib")["scaled"]
    frame=source.loc[source.unit.isin(fitting_ids+selection_ids)].copy()
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    protocol={"target_r2":.8,"metric":"unchanged raw-sensor R2",
        "training_plan_id":plan["plan_id"],"fit_units":fitting_ids,"selection_units":selection_ids,
        "protected_models":protected,"recipes":list(recipes(args.clusters)),"commissioning_cycles":30,
        "fit_rul_min":130,"score_rul_min":125,"official_and_confirmation_closed":True,
        "encoder_uses":"All first-30-cycle mode means, variances and counts; no RUL, current or later sensor readings",
        "gate":"R2 increases and RMSE/MAE decrease; per-subset RMSE <=1% worse and R2 <=.01 worse"}
    if (run/"protocol.json").exists():
        assert json.loads((run/"protocol.json").read_text())["protected_models"]==protected
    else:
        write(run/"protocol.json",protocol)
    population=bundle["population"].predict(EnsembleTwin._profile_context(frame,bundle))
    original_features,ready=EnsembleTwin._profile_features(frame,population,bundle)
    features,new_ready=multimode_features(frame,population,bundle)
    np.testing.assert_array_equal(ready,new_ready)
    actual=frame[bundle["target_names"]].to_numpy(float)
    fit=ready&frame.unit.isin(fitting_ids).to_numpy()&frame.RUL.ge(130).to_numpy()
    validation=ready&frame.unit.isin(selection_ids).to_numpy()&frame.RUL.ge(125).to_numpy()
    baseline_correction=EnsembleTwin._predict_profile_correction(bundle["correction"],original_features[validation])
    baseline_prediction=population.copy()
    baseline_prediction[validation]+=baseline_correction
    baseline=score(frame,actual,baseline_prediction,validation)
    write(run/"baseline.json",baseline)
    print("BASELINE",{k:baseline[k] for k in ("r2","rmse","mae")},flush=True)
    xa,ya=aggregate(frame,features,actual-population,fit,20)
    np.savez_compressed(run/"validation_data.npz",actual=actual[validation],population=population[validation],
        baseline_correction=baseline_correction,units=frame.loc[validation,"unit"].to_numpy(),
        subsets=frame.loc[validation,"subset_id"].to_numpy(dtype=str))
    path=run/"search.json"
    state=json.loads(path.read_text()) if path.exists() else {"trials":[],"best":None}
    completed={trial["index"] for trial in state["trials"]}
    for index,spec in enumerate(recipes(args.clusters)):
        if index in completed:
            continue
        started=time.perf_counter()
        try:
            model=ConditionProfileRegressor(**spec).fit(xa,ya)
            correction=model.predict(features[validation])
            predicted=population.copy()
            blends=[]
            for weight in (.25,.5,.75,1.):
                predicted[validation]=population[validation]+(1.-weight)*baseline_correction+weight*correction
                metrics=score(frame,actual,predicted,validation)
                eligible=improves(baseline,metrics) and subset_gate(baseline,metrics)
                blends.append({"weight":weight,"eligible":eligible,"metrics":metrics})
                if eligible and (state["best"] is None or metrics["r2"]>state["best"]["metrics"]["r2"]):
                    state["best"]={"index":index,"spec":spec,"weight":weight,"metrics":metrics}
                    candidate=dict(bundle)
                    candidate.update(correction=BlendedStartupCorrection(ColumnModel(bundle["correction"],original_features.shape[1]),model,weight),
                        feature_schema="commissioning_multimode_v1",correction_recipe=f"condition_state_{index}",
                        validation_metrics=metrics,training_plan_id=plan["plan_id"])
                    joblib.dump(candidate,run/"candidate_profile.pkl",compress=3)
                    recipe={"name":candidate["correction_recipe"],"family":"full_fit_condition_profile",
                        "spec":dict(spec,bins=20),"weight":weight,"training_plan_id":plan["plan_id"],
                        "baseline_recipe":json.loads((config.paths.models/"startup_profile_recipe.json").read_text()),
                        "report":str(run/"evaluation.json")}
                    write(run/"selected_recipe.json",recipe)
            np.savez_compressed(run/f"prediction_{index:03}.npz",correction=correction)
            trial={"index":index,"spec":spec,"blends":blends,"seconds":time.perf_counter()-started}
            winner=max(blends,key=lambda item:item["metrics"]["r2"])
            print(index,spec,"R2",round(winner["metrics"]["r2"],6),"eligible",any(b["eligible"] for b in blends),
                  "seconds",round(trial["seconds"],1),flush=True)
        except Exception as exc:
            trial={"index":index,"spec":spec,"error":repr(exc),"seconds":time.perf_counter()-started}
            print("FAILED",index,repr(exc),flush=True)
        state["trials"].append(trial)
        write(path,state)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    print("BEST DEVELOPMENT",json.dumps(state["best"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",default=str(RUN))
    parser.add_argument("--clusters",action="store_true")
    with threadpool_limits(limits=4):
        main(parser.parse_args())
