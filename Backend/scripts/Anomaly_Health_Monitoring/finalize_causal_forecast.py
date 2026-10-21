from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.search_causal_forecast import RUN,recipes
from scripts.Anomaly_Health_Monitoring.finalize_profile_target import fit_weights,mixture
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,digest,score,improves,subset_gate,write
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import CausalSensorRegressor,prior_sensor_features
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import ColumnModel
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import SensorBlendStartupCorrection
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split,training_scope

def main(args):
    run=Path(args.run_dir).resolve()
    final=run/"final"
    if final.exists():
        raise ValueError("Finalization is single-use; never retune a frozen confirmation candidate")
    sources=[(run,False,False),(Path(str(run)+"_nonlinear"),True,False),(Path(str(run)+"_multimode"),False,True)]
    best=None
    families={}
    total=0
    for source,nonlinear,multimode in sources:
        state=json.loads((source/"search.json").read_text())
        expected=len(list(recipes(nonlinear,multimode)))
        if len(state["trials"])!=expected:
            raise ValueError(f"Search incomplete: {source.name} ({len(state['trials'])}/{expected})")
        total+=expected
        if state["best"] and (best is None or state["best"]["metrics"]["r2"]>best["item"]["metrics"]["r2"]):
            best={"source":source,"item":state["best"]}
        for trial in state["trials"]:
            metrics=max(trial["blends"],key=lambda b:b["metrics"]["r2"])["metrics"]
            family=(trial["spec"].get("family","ridge"),trial["spec"]["scope"],multimode)
            if family not in families or metrics["r2"]>families[family]["r2"]:
                families[family]={"source":source,"trial":trial,"r2":metrics["r2"]}
    if best is None:
        raise ValueError("No jointly improved development candidate")
    config=Config()
    config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    bundle=joblib.load(config.paths.models/"startup_profile_twin.pkl")
    baseline_recipe=json.loads((config.paths.models/"startup_profile_recipe.json").read_text())
    prepared=joblib.load(SOURCE/"prepared.joblib")["scaled"]
    fit_ids,selection_ids=plan["engine_roles"]["fit"],plan["engine_roles"]["selection"]
    frame=prepared.loc[prepared.unit.isin(fit_ids+selection_ids)].copy()
    population=bundle["population"].predict(EnsembleTwin._profile_context(frame,bundle))
    old_x,ready=EnsembleTwin._profile_features(frame,population,bundle)
    x,_=prior_sensor_features(frame,population,bundle)
    extended_x=None
    actual=frame[bundle["target_names"]].to_numpy(float)
    validation=ready&frame.unit.isin(selection_ids).to_numpy()&frame.RUL.ge(125).to_numpy()
    baseline=population.copy()
    baseline[validation]+=EnsembleTwin._predict_profile_correction(bundle["correction"],old_x[validation])
    baseline_metrics=score(frame,actual,baseline,validation)
    pool=sorted(families.values(),key=lambda item:item["r2"],reverse=True)[:6]
    predictions=[population[validation]+np.load(item["source"]/f"prediction_{item['trial']['index']:03}.npz")["correction"] for item in pool]
    units=frame.loc[validation].groupby("unit").subset_id.first().sort_index()
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20261018).split(units.index,units))
    trials=[]
    selected_penalty=None
    for penalty in (.005,.05,.5,5.):
        oof=baseline[validation].copy()
        for train_index,held_index in splits:
            train=frame.loc[validation].unit.isin(units.index[train_index]).to_numpy()
            held=frame.loc[validation].unit.isin(units.index[held_index]).to_numpy()
            weights=fit_weights(actual[validation][train],baseline[validation][train],[p[train] for p in predictions],penalty)
            oof[held]=mixture(baseline[validation][held],[p[held] for p in predictions],weights)
        output=population.copy()
        output[validation]=oof
        metrics=score(frame,actual,output,validation)
        eligible=improves(baseline_metrics,metrics) and subset_gate(baseline_metrics,metrics)
        trials.append({"penalty":penalty,"metrics":metrics,"eligible":eligible})
        print("CROSSFIT",penalty,metrics["r2"],metrics["rmse"],eligible,flush=True)
        if eligible and (selected_penalty is None or metrics["r2"]>selected_penalty["metrics"]["r2"]):
            selected_penalty=trials[-1]
    candidate=joblib.load(best["source"]/"candidate_profile.pkl")
    recipe=json.loads((best["source"]/"selected_recipe.json").read_text())
    chosen={"metrics":best["item"]["metrics"]}
    selected_as="best_single_forecast"
    if selected_penalty is not None and selected_penalty["metrics"]["r2"]>chosen["metrics"]["r2"]:
        weights=fit_weights(actual[validation],baseline[validation],predictions,selected_penalty["penalty"])
        members,specifications=[],[]
        for member,sensor_weights in zip(pool,weights):
            if sensor_weights.max()<=1e-8:
                continue
            spec=dict(member["trial"]["spec"])
            scope=spec.pop("scope")
            multimode=spec.get("features")=="multimode"
            if multimode and extended_x is None:
                extended_x,_=prior_sensor_features(frame,population,bundle,multimode=True)
            member_x=extended_x if multimode else x
            fit=ready&frame.unit.isin(fit_ids).to_numpy()
            if scope=="healthy":
                fit&=frame.RUL.ge(130).to_numpy()
            if member["source"] == best["source"] and member["trial"]["spec"] == best["item"]["spec"]:
                fitted=candidate["correction"].candidate
            else:
                fitted=CausalSensorRegressor(**spec).fit(member_x[fit],(actual-population)[fit])
            recorded=np.load(member["source"]/f"prediction_{member['trial']['index']:03}.npz")["correction"]
            np.testing.assert_allclose(fitted.predict(member_x[validation]),recorded,rtol=0,atol=1e-10)
            members.append(ColumnModel(fitted,member_x.shape[1]))
            specifications.append({"spec":member["trial"]["spec"],"weights":sensor_weights.tolist()})
        model=SensorBlendStartupCorrection(ColumnModel(bundle["correction"],old_x.shape[1]),members,[s["weights"] for s in specifications])
        multimode=any(s["spec"].get("features")=="multimode" for s in specifications)
        mixture_x=extended_x if multimode else x
        output=population.copy()
        output[validation]+=model.predict(mixture_x[validation])
        metrics=score(frame,actual,output,validation)
        if improves(baseline_metrics,metrics) and subset_gate(baseline_metrics,metrics):
            candidate=dict(bundle)
            candidate.update(correction=model,correction_recipe="causal_sensor_crossfit_mixture",
                feature_schema="causal_multimode_history_v1" if multimode else "causal_sensor_history_v1",
                prediction_role="adaptive_sensor_forecast",validation_metrics=metrics,training_plan_id=plan["plan_id"],fit_unit_ids=fit_ids)
            recipe={"name":candidate["correction_recipe"],"family":"full_fit_causal_sensor_mixture","members":specifications,
                "penalty":selected_penalty["penalty"],"training_plan_id":plan["plan_id"],"baseline_recipe":baseline_recipe,
                "prediction_role":"adaptive_sensor_forecast"}
            chosen={"metrics":metrics,"crossfit_metrics":selected_penalty["metrics"]}
            selected_as="crossfit_sensor_mixture"
    final.mkdir()
    recipe["report"]=str(final/"evaluation.json")
    joblib.dump(candidate,final/"candidate_profile.pkl",compress=3)
    write(final/"selected_recipe.json",recipe)
    write(final/"mixture_crossfit.json",trials)
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob("*") if p.is_file()}
    for source,_,_ in sources:
        assert protected==json.loads((source/"protocol.json").read_text())["protected_models"]
    write(final/"protocol.json",{"protected_models":protected,"training_plan_id":plan["plan_id"],"fit_units":fit_ids,
        "selection_units":selection_ids,"model_recipes_tested":total,"target_r2":.8,"selection":selected_as,
        "prediction_role":"adaptive_sensor_forecast","user_authorized_prior_readings":True,
        "raw_targets_and_metric_unchanged":True,"model_selection_uses_confirmation_or_official":False,
        "publication_requires_official_aggregate_improvement_without_retuning":True,
        "full_trajectory_scores":"descriptive only; primary healthy population unchanged",
        "forecast_inputs":"current operating context, frozen first30 statistics, strictly past sensor readings",
        "monitoring_reference":"unchanged_frozen_healthy_expectation","individual_row_training":True})
    write(final/"model_lock.json",{"candidate_sha256":digest(final/"candidate_profile.pkl"),
        "locked_before_confirmation":True,"locked_before_official":True,"selection":chosen})
    confirmation=training_scope(prepared,config,("confirmation",),required=True)
    cp=bundle["population"].predict(EnsembleTwin._profile_context(confirmation,bundle))
    cx,cr=EnsembleTwin._profile_features(confirmation,cp,bundle)
    nx,nr=EnsembleTwin._profile_features(confirmation,cp,candidate)
    np.testing.assert_array_equal(cr,nr)
    mask=cr&confirmation.RUL.ge(125).to_numpy()
    cy=confirmation[bundle["target_names"]].to_numpy(float)
    old,new=cp.copy(),cp.copy()
    old[mask]+=EnsembleTwin._predict_profile_correction(bundle["correction"],cx[mask])
    new[mask]+=EnsembleTwin._predict_profile_correction(candidate["correction"],nx[mask])
    confirmed={"baseline":score(confirmation,cy,old,mask),"candidate":score(confirmation,cy,new,mask)}
    accepted=improves(confirmed["baseline"],confirmed["candidate"]) and subset_gate(confirmed["baseline"],confirmed["candidate"])
    result={"baseline":{"selection":baseline_metrics},"selected":chosen,"confirmation":confirmed,"accepted":accepted,
        "production_changed":False,"model_recipes_tested":total,"target_r2":.8,"prediction_role":"adaptive_sensor_forecast",
        "target_met_on_confirmation":confirmed["candidate"]["r2"]>=.8,"confirmation_used_for_retuning":False,
        "baseline_prediction_role":"frozen_commissioning_expectation"}
    write(final/"evaluation.json",result)
    print("FROZEN CONFIRMATION",json.dumps(confirmed),flush=True)
    print("ACCEPTED FOR VERIFICATION",accepted,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",default=str(RUN))
    with threadpool_limits(limits=4):
        main(parser.parse_args())
