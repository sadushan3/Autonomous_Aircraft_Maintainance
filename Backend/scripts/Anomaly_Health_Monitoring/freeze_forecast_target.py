from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,score,improves,subset_gate,write,digest
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import ColumnModel
from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import SensorBlendStartupCorrection
from app.services.Anomaly_Health_Monitoring.digital_twin.independent_forecast import IndependentSensorForecast
from app.services.Anomaly_Health_Monitoring.digital_twin.multitask_forecast import MultitaskSensorForecast
from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features
from app.services.Anomaly_Health_Monitoring.digital_twin.forecast_policy import healthy_forecast_guard
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split,training_scope

def accepted(before,after,policy):
    return (improves(before['full'],after['full']) and healthy_forecast_guard(before['healthy'],after['healthy'],policy)
        and all(subset_gate(before[k],after[k]) for k in ('healthy','full')))

def main(args):
    run=Path(args.run_dir).resolve()
    if (run/'model_lock.json').exists(): raise ValueError('Already frozen; confirmation may not be used for retuning')
    selection=json.loads((run/'mixture_selection.json').read_text())
    if not selection.get('eligible'): raise ValueError('No eligible development mixture')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config); protocol=json.loads((run/'protocol.json').read_text())
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    assert protected==protocol['protected_models']
    bundle=joblib.load(config.paths.models/'startup_profile_twin.pkl')
    baseline_recipe=json.loads((config.paths.models/'startup_profile_recipe.json').read_text())
    prepared=joblib.load(SOURCE/'prepared.joblib')['scaled']
    fit=prepared.loc[prepared.unit.isin(plan['engine_roles']['fit'])].copy()
    population=bundle['population'].predict(EnsembleTwin._profile_context(fit,bundle))
    fast=any(member.get('spec',{}).get('history')=='fast_modes' and max(weights)>1e-8
        for member,weights in zip(selection['pool'],selection['weights']))
    x,ready=(fast_mode_features(fit,population,bundle) if fast else EnsembleTwin._profile_features(fit,population,bundle))
    y=fit[bundle['target_names']].to_numpy(float)-population; rul=fit.RUL.to_numpy()
    x,y,rul=x[ready],y[ready],rul[ready]
    del fit,population; gc.collect()
    data=np.load(run/'selection_data.npz')
    sx=np.load(run/'fast_modes/selection_features.npz')['x'] if fast else data['x']
    models,specifications=[],[]
    for member,sensor_weights in zip(selection['pool'],selection['weights']):
        if max(sensor_weights)<=1e-8: continue
        if member['source']=='baseline_member':
            model=bundle['correction'].members[member['index']]
            recorded=np.load(run/f"candidate_{member['index']:02}.npz")['correction']
        else:
            spec=member['spec']; kind=member.get('kind',member['source'])
            width=952 if spec.get('history')=='fast_modes' else (691 if kind=='independent' else 427)
            print('REFITTING SELECTED MEMBER',member['source'],member['index'],spec,flush=True)
            fitted=(IndependentSensorForecast(**spec).fit(x[:,:width],y) if kind=='independent'
                else MultitaskSensorForecast(**spec).fit(x[:,:width],y,rul))
            model=ColumnModel(fitted,width)
            recorded=np.load(run/member['source']/f"prediction_{member['index']:02}.npz")['correction']
        reproduced=EnsembleTwin._predict_profile_correction(model,sx)
        np.testing.assert_allclose(reproduced,recorded,rtol=0,atol=1e-10)
        models.append(model); specifications.append(dict(member,weights=sensor_weights))
    correction=SensorBlendStartupCorrection(ColumnModel(bundle['correction'],691),models,[m['weights'] for m in specifications])
    actual,sp,old=data['actual'],data['population'],data['baseline_correction']
    prediction=sp+EnsembleTwin._predict_profile_correction(correction,sx)
    frame=pd.DataFrame({'unit':data['units'],'subset_id':data['subsets']})
    masks={'healthy':data['healthy'],'full':np.ones(len(actual),bool)}
    measured={k:score(frame,actual,prediction,mask) for k,mask in masks.items()}
    if not accepted(selection['baseline'],measured,selection['healthy_guard']): raise ValueError('Refitted development mixture no longer passes')
    schema='causal_fast_modes_v1' if fast else 'causal_multimode_history_v1'
    recipe={'name':'forecast_target_080_constrained_mixture','family':'full_fit_forecast_target_mixture',
        'members':specifications,'baseline_recipe':baseline_recipe,'training_plan_id':plan['plan_id'],
        'feature_schema':schema,'objective':'full_trajectory_forecasts','healthy_guard':selection['healthy_guard'],
        'weight_selection':selection['best'],'report':str(run/'evaluation.json')}
    candidate=dict(bundle)
    candidate.update(correction=correction,feature_schema=schema,correction_recipe=recipe['name'],
        validation_metrics=measured['healthy'],reference_validation_metrics=selection['baseline']['healthy'],
        full_trajectory_validation_metrics=measured['full'],full_trajectory_reference_metrics=selection['baseline']['full'],
        validation_rows=measured['healthy']['rows'],validation_role='selection',prediction_role='adaptive_sensor_forecast',
        prediction_objective='full_trajectory_forecasts',validated_profile_recipe=recipe,
        training_plan_id=plan['plan_id'],fit_unit_ids=plan['engine_roles']['fit'],reference_preserved_for_monitoring=True)
    joblib.dump(candidate,run/'candidate_profile.pkl',compress=3)
    write(run/'selected_recipe.json',recipe)
    protocol.update(objective='full_trajectory_forecasts',healthy_guard=selection['healthy_guard'],
        selected_recipe=recipe['name'],model_recipes_tested=32,model_selection_uses_confirmation_or_official=False,
        publication_requires_official_full_metrics_improvement_and_healthy_guard=True)
    write(run/'protocol.json',protocol)
    write(run/'model_lock.json',{'candidate_sha256':digest(run/'candidate_profile.pkl'),
        'locked_before_confirmation':True,'locked_before_official':True,'selection':measured})
    print('LOCKED; OPENING CONFIRMATION',flush=True)
    confirmation=training_scope(prepared,config,('confirmation',),required=True)
    cp=bundle['population'].predict(EnsembleTwin._profile_context(confirmation,bundle))
    bx,br=EnsembleTwin._profile_features(confirmation,cp,bundle)
    cx,cr=EnsembleTwin._profile_features(confirmation,cp,candidate)
    np.testing.assert_array_equal(br,cr)
    cy=confirmation[bundle['target_names']].to_numpy(float)
    old,new=cp.copy(),cp.copy()
    old[br]+=EnsembleTwin._predict_profile_correction(bundle['correction'],bx[br])
    new[cr]+=EnsembleTwin._predict_profile_correction(correction,cx[cr])
    masks={'healthy':cr&confirmation.RUL.ge(125).to_numpy(),'full':cr}
    before={k:score(confirmation,cy,old,mask) for k,mask in masks.items()}
    after={k:score(confirmation,cy,new,mask) for k,mask in masks.items()}
    passed=accepted(before,after,selection['healthy_guard'])
    result={'baseline':{'selection':selection['baseline']['healthy']},
        'selected':{'metrics':measured['healthy'],'crossfit_metrics':selection['best']['metrics']['healthy']},
        'development_full_trajectories':{'baseline':selection['baseline']['full'],'candidate':measured['full'],
            'oof_candidate':selection['best']['metrics']['full']},
        'confirmation':{'baseline':before['healthy'],'candidate':after['healthy']},
        'confirmation_full_trajectories':{'baseline':before['full'],'candidate':after['full']},
        'accepted':passed,'production_changed':False,'target_r2':.8,'objective':'full_trajectory_forecasts',
        'healthy_guard':selection['healthy_guard'],'prediction_role':'adaptive_sensor_forecast',
        'baseline_prediction_role':'adaptive_sensor_forecast','model_recipes_tested':32,
        'target_met_on_confirmation':after['full']['r2']>=.8,'confirmation_used_for_retuning':False,
        'report_name':'forecast_target_080_20261016.md','metric_name':'forecast_target_080.json'}
    write(run/'evaluation.json',result)
    print('FROZEN CONFIRMATION',json.dumps({'before':before,'after':after,'accepted':passed}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN))
    with threadpool_limits(limits=4): main(parser.parse_args())
