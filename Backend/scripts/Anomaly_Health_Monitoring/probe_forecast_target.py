from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,score,improves,subset_gate,write,digest
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

RUN=ROOT/'experiments/Anomaly_Health_Monitoring/run_20261016_forecast_080'

def main(args):
    run=Path(args.run_dir).resolve(); run.mkdir(exist_ok=True)
    if (run/'model_lock.json').exists(): raise ValueError('Candidate is frozen; no confirmation retuning')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    write(run/'protocol.json',{'target_r2':.8,'objective':'full_trajectory_sensor_forecasts',
        'primary_original_healthy_benchmark_also_reported':True,'protected_models':protected,
        'training_plan_id':plan['plan_id'],'fit_units':plan['engine_roles']['fit'],
        'selection_units':plan['engine_roles']['selection'],'confirmation_and_official_closed':True,
        'raw_targets_formula_and_trajectory_rows_unchanged':True,
        'monitoring_reference':'unchanged_frozen_context_ensemble',
        'inputs':'operating context, commissioning statistics, strictly prior sensor readings',
        'selection':'full-trajectory joint R2/RMSE/MAE improvement plus healthy joint improvement and subset guards'})
    bundle=joblib.load(config.paths.models/'startup_profile_twin.pkl')
    recipe=json.loads((config.paths.models/'startup_profile_recipe.json').read_text())
    prepared=joblib.load(SOURCE/'prepared.joblib')['scaled']
    fitting=prepared.loc[prepared.unit.isin(plan['engine_roles']['fit'])&prepared.RUL.ge(130)]
    targets=bundle['target_names']
    groups=fitting.groupby(['unit','operating_mode'])[targets]
    counts=groups.size().to_numpy()-1
    variance=groups.var().fillna(0.).to_numpy()
    noise=np.sqrt(np.maximum(np.average(variance,axis=0,weights=np.maximum(counts,0)),.0025))
    frame=prepared.loc[prepared.unit.isin(plan['engine_roles']['selection'])].copy()
    p=bundle['population'].predict(EnsembleTwin._profile_context(frame,bundle))
    x,ready=EnsembleTwin._profile_features(frame,p,bundle)
    y=frame[targets].to_numpy(float)
    healthy=ready&frame.RUL.ge(125).to_numpy()
    baseline_correction=EnsembleTwin._predict_profile_correction(bundle['correction'],x[ready])
    baseline=p.copy(); baseline[ready]+=baseline_correction
    before={'healthy':score(frame,y,baseline,healthy),'full':score(frame,y,baseline,ready)}
    write(run/'baseline.json',before)
    print('BASELINE',json.dumps({k:{m:v[m] for m in ('r2','rmse','mae')} for k,v in before.items()}),flush=True)
    np.savez_compressed(run/'selection_data.npz',x=x[ready],actual=y[ready],population=p[ready],
        baseline_correction=baseline_correction,healthy=healthy[ready],units=frame.loc[ready,'unit'].to_numpy(),
        subsets=frame.loc[ready,'subset_id'].to_numpy(dtype=str),noise_scale=noise)
    results=[]; best=None
    for index,(model,spec) in enumerate(zip(bundle['correction'].members,recipe['members'])):
        correction=EnsembleTwin._predict_profile_correction(model,x[ready])
        np.savez_compressed(run/f'candidate_{index:02}.npz',correction=correction)
        predictions=p.copy(); predictions[ready]+=correction
        direct={'healthy':score(frame,y,predictions,healthy),'full':score(frame,y,predictions,ready)}
        print('MEMBER',index,spec['spec'],json.dumps({k:{m:v[m] for m in ('r2','rmse','mae')} for k,v in direct.items()}),flush=True)
        trials=[]
        for gate in ('fixed','per_sensor','shared'):
            starts=(0.,.05,.1,.25,.5) if gate!='fixed' else (0.,)
            for start in starts:
                ends=(.25,.5,1.,2.) if gate!='fixed' else (1.,)
                for end in ends:
                    if end<=start: continue
                    standardized=np.abs(correction-baseline_correction)/noise
                    if gate=='shared': standardized=np.mean(standardized,axis=1,keepdims=True)
                    response=np.clip((standardized-start)/(end-start),0.,1.) if gate!='fixed' else 1.
                    for weight in (.25,.5,.75,1.):
                        predictions[ready]=p[ready]+baseline_correction+weight*response*(correction-baseline_correction)
                        metrics={'healthy':score(frame,y,predictions,healthy),'full':score(frame,y,predictions,ready)}
                        eligible=all(improves(before[k],metrics[k]) and subset_gate(before[k],metrics[k]) for k in before)
                        trial={'member':index,'gate':gate,'start':start,'end':end,'weight':weight,
                            'eligible':eligible,'metrics':metrics}
                        trials.append(trial)
                        if eligible and (best is None or metrics['full']['r2']>best['metrics']['full']['r2']): best=trial
        results.append({'member':index,'spec':spec['spec'],'direct':direct,'trials':trials})
        write(run/'gate_search.json',{'members':results,'best':best})
        print('BEST SO FAR',json.dumps(best),flush=True)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN))
    with threadpool_limits(limits=4): main(parser.parse_args())
