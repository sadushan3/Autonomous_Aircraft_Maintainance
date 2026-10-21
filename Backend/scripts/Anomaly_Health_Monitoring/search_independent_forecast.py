from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.independent_forecast_models import IndependentSensorForecast
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,score,improves,subset_gate,write,digest
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

def recipes():
    for grouping in ('subset_mode','subset'):
        for features in ('own','shared'):
            for alpha in (10.,1000.):
                yield dict(family='ridge',grouping=grouping,features=features,alpha=alpha)
    for grouping in ('subset_mode','subset'):
        for features in ('own','shared'):
            for leaves in (7,31):
                yield dict(family='lightgbm',grouping=grouping,features=features,alpha=30.,leaves=leaves,leaf=60,n_estimators=240)

def main(args):
    run=Path(args.run_dir).resolve(); run.mkdir(exist_ok=True)
    if (run.parent/'model_lock.json').exists(): raise ValueError('Candidate locked; no confirmation retuning')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    protocol=json.loads((run.parent/'protocol.json').read_text())
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    assert protected==protocol['protected_models']
    write(run/'recipes.json',list(recipes()))
    bundle=joblib.load(config.paths.models/'startup_profile_twin.pkl')
    frame=joblib.load(SOURCE/'prepared.joblib')['scaled']
    fit_frame=frame.loc[frame.unit.isin(plan['engine_roles']['fit'])].copy()
    print('BUILDING FIT FEATURES',len(fit_frame),flush=True)
    population=bundle['population'].predict(EnsembleTwin._profile_context(fit_frame,bundle))
    x,ready=EnsembleTwin._profile_features(fit_frame,population,bundle)
    y=fit_frame[bundle['target_names']].to_numpy(float)-population
    data=np.load(run.parent/'selection_data.npz')
    sx,sy,sp,old=data['x'],data['actual'],data['population'],data['baseline_correction']
    full_frame=frame.loc[frame.unit.isin(plan['engine_roles']['selection'])].copy()
    _,selection_ready=EnsembleTwin._profile_features(full_frame,
        bundle['population'].predict(EnsembleTwin._profile_context(full_frame,bundle)),bundle)
    selection=full_frame.loc[selection_ready].copy()
    np.testing.assert_array_equal(sy,selection[bundle['target_names']].to_numpy(float))
    masks={'healthy':data['healthy'],'full':np.ones(len(selection),bool)}
    before={key:score(selection,sy,sp+old,mask) for key,mask in masks.items()}
    path=run/'search.json'; state=json.loads(path.read_text()) if path.exists() else {'trials':[],'best':None}
    completed={item['index'] for item in state['trials']}
    print('BASELINE',json.dumps({k:{m:v[m] for m in ('r2','rmse','mae')} for k,v in before.items()}),flush=True)
    for index,spec in enumerate(recipes()):
        if index in completed: continue
        started=time.perf_counter(); model=IndependentSensorForecast(**spec).fit(x[ready],y[ready])
        correction=model.predict(sx)
        np.savez_compressed(run/f'prediction_{index:02}.npz',correction=correction)
        trials=[]
        for weight in (.25,.5,.75,1.):
            prediction=sp+(1-weight)*old+weight*correction
            metrics={key:score(selection,sy,prediction,mask) for key,mask in masks.items()}
            eligible=all(improves(before[key],metrics[key]) and subset_gate(before[key],metrics[key]) for key in masks)
            trial={'weight':weight,'metrics':metrics,'eligible':eligible}; trials.append(trial)
            if eligible and (state['best'] is None or metrics['full']['r2']>state['best']['metrics']['full']['r2']):
                state['best']={'index':index,'spec':spec,**trial}
                joblib.dump(model,run/'best_model.pkl',compress=3)
        state['trials'].append({'index':index,'spec':spec,'blends':trials,'seconds':time.perf_counter()-started})
        write(path,state)
        print('TRIAL',index,spec,[(round(t['metrics']['full']['r2'],6),round(t['metrics']['healthy']['r2'],6),t['eligible']) for t in trials],
            'SECONDS',round(time.perf_counter()-started,1),flush=True)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    print('BEST',json.dumps(state['best']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN/'independent'))
    with threadpool_limits(limits=4): main(parser.parse_args())
