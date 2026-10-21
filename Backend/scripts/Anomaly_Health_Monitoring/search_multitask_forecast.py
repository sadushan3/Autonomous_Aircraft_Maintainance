from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.multitask_forecast_models import MultitaskSensorForecast
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,score,improves,subset_gate,write,digest
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import prior_sensor_features
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

def recipes():
    for grouping in ('subset','all'):
        for weight in (0.,1.,3.): yield dict(grouping=grouping,auxiliary_weight=weight,alpha=1.,max_iter=150)

def main(args):
    run=Path(args.run_dir).resolve(); run.mkdir(exist_ok=True)
    if (run.parent/'model_lock.json').exists(): raise ValueError('Candidate locked; no confirmation retuning')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    assert protected==json.loads((run.parent/'protocol.json').read_text())['protected_models']
    write(run/'recipes.json',{'recipes':list(recipes()),'labels_used_only_in_fit':['raw_sensors','capped_RUL'],
        'score_outputs':'21 original unsmoothed sensor targets; auxiliary RUL excluded',
        'early_stopping':'random rows within fitting engines only; selection engines are separate'})
    bundle=joblib.load(config.paths.models/'startup_profile_twin.pkl'); bundle.pop('correction'); gc.collect()
    frame=joblib.load(SOURCE/'prepared.joblib')['scaled']; frame=frame.loc[frame.unit.isin(plan['engine_roles']['fit'])].copy()
    print('BUILDING CAUSAL TRAINING FEATURES',len(frame),flush=True)
    population=bundle['population'].predict(EnsembleTwin._profile_context(frame,bundle))
    x,ready=prior_sensor_features(frame,population,bundle)
    y=frame[bundle['target_names']].to_numpy(float)-population
    data=np.load(run.parent/'selection_data.npz')
    sx,sy,sp,old=data['x'][:,:427],data['actual'],data['population'],data['baseline_correction']
    validation=pd.DataFrame({'unit':data['units'],'subset_id':data['subsets']})
    masks={'healthy':data['healthy'],'full':np.ones(len(sy),bool)}
    before={k:score(validation,sy,sp+old,mask) for k,mask in masks.items()}
    path=run/'search.json'; state=json.loads(path.read_text()) if path.exists() else {'trials':[],'best':None}
    completed={t['index'] for t in state['trials']}
    for index,spec in enumerate(recipes()):
        if index in completed: continue
        started=time.perf_counter()
        model=MultitaskSensorForecast(**spec).fit(x[ready],y[ready],frame.loc[ready,'RUL'].to_numpy())
        correction=model.predict(sx); np.savez_compressed(run/f'prediction_{index:02}.npz',correction=correction)
        blends=[]
        for weight in (.25,.5,.75,1.):
            predictions=sp+(1.-weight)*old+weight*correction
            metrics={k:score(validation,sy,predictions,mask) for k,mask in masks.items()}
            eligible=all(improves(before[k],metrics[k]) and subset_gate(before[k],metrics[k]) for k in masks)
            blend={'weight':weight,'metrics':metrics,'eligible':eligible}; blends.append(blend)
            if eligible and (state['best'] is None or metrics['full']['r2']>state['best']['metrics']['full']['r2']):
                state['best']={'index':index,'spec':spec,**blend}; joblib.dump(model,run/'best_model.pkl',compress=3)
        state['trials'].append({'index':index,'spec':spec,'blends':blends,'seconds':time.perf_counter()-started})
        write(path,state)
        print('TRIAL',index,spec,[(round(t['metrics']['full']['r2'],6),round(t['metrics']['healthy']['r2'],6),t['eligible']) for t in blends],
            'SECONDS',round(time.perf_counter()-started,1),flush=True)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    print('BEST',json.dumps(state['best']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN/'multitask'))
    with threadpool_limits(limits=4): main(parser.parse_args())
