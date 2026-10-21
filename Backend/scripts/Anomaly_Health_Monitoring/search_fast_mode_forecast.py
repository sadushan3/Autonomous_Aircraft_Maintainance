from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.fast_mode_features import fast_mode_features
from scripts.Anomaly_Health_Monitoring.independent_forecast_models import IndependentSensorForecast
from scripts.Anomaly_Health_Monitoring.multitask_forecast_models import MultitaskSensorForecast
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,score,improves,subset_gate,write,digest
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

def recipes():
    for grouping in ('subset_mode','subset'):
        for alpha in (10.,1000.):
            yield {'kind':'independent','spec':dict(family='ridge',grouping=grouping,features='shared',history='fast_modes',alpha=alpha)}
        for leaves in (7,31):
            yield {'kind':'independent','spec':dict(family='lightgbm',grouping=grouping,features='shared',history='fast_modes',alpha=30.,leaves=leaves,leaf=60,n_estimators=240)}
    for grouping in ('subset','all'):
        yield {'kind':'multitask','spec':dict(grouping=grouping,history='fast_modes',auxiliary_weight=1.,max_iter=150,alpha=1.)}

def causal_contract(frame,pop,bundle):
    sample=frame.loc[frame.unit.isin(frame.groupby('subset_id').unit.first())&frame.cycle.le(60)].copy()
    sp=bundle['population'].predict(EnsembleTwin._profile_context(sample,bundle))
    x,ready=fast_mode_features(sample,sp,bundle)
    changed=sample.copy(); changed.loc[changed.cycle.ge(45),bundle['target_names']]+=1000.
    changed_x,_=fast_mode_features(changed,sp,bundle)
    np.testing.assert_allclose(x[sample.cycle.le(45)],changed_x[sample.cycle.le(45)],rtol=0,atol=1e-10)
    assert np.any(np.abs(x[sample.cycle.gt(45)]-changed_x[sample.cycle.gt(45)])>1.)
    shuffled=sample.sample(frac=1.,random_state=42)
    shp=bundle['population'].predict(EnsembleTwin._profile_context(shuffled,bundle))
    sx,sr=fast_mode_features(shuffled,shp,bundle)
    positions=sample.index.get_indexer(shuffled.index)
    np.testing.assert_allclose(sx,x[positions],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(sr,ready[positions])

def main(args):
    run=Path(args.run_dir).resolve(); run.mkdir(exist_ok=True)
    if (run.parent/'model_lock.json').exists(): raise ValueError('Candidate locked; no confirmation retuning')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config)
    protected={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    assert protected==json.loads((run.parent/'protocol.json').read_text())['protected_models']
    write(run/'recipes.json',{'recipes':list(recipes()),'feature_schema':'causal_fast_modes_v1',
        'features_added':'strictly past mode EWM span3, last values, previous mode and operating settings'})
    bundle=joblib.load(config.paths.models/'startup_profile_twin.pkl'); bundle.pop('correction'); gc.collect()
    frame=joblib.load(SOURCE/'prepared.joblib')['scaled']
    fit=frame.loc[frame.unit.isin(plan['engine_roles']['fit'])].copy()
    selection=frame.loc[frame.unit.isin(plan['engine_roles']['selection'])].copy()
    print('BUILDING FAST HISTORY FEATURES',len(fit),flush=True)
    p=bundle['population'].predict(EnsembleTwin._profile_context(fit,bundle))
    causal_contract(fit,p,bundle)
    x,ready=fast_mode_features(fit,p,bundle)
    y=fit[bundle['target_names']].to_numpy(float)-p
    x,y,rul=x[ready],y[ready],fit.loc[ready,'RUL'].to_numpy()
    del fit,p; gc.collect()
    sp=bundle['population'].predict(EnsembleTwin._profile_context(selection,bundle))
    sx,sr=fast_mode_features(selection,sp,bundle); sx=sx[sr]
    np.savez_compressed(run/'selection_features.npz',x=sx)
    data=np.load(run.parent/'selection_data.npz'); actual,old,sp=data['actual'],data['baseline_correction'],data['population']
    np.testing.assert_array_equal(actual,selection.loc[sr,bundle['target_names']].to_numpy(float))
    validation=pd.DataFrame({'unit':data['units'],'subset_id':data['subsets']})
    masks={'healthy':data['healthy'],'full':np.ones(len(actual),bool)}
    before={k:score(validation,actual,sp+old,mask) for k,mask in masks.items()}
    path=run/'search.json'; state=json.loads(path.read_text()) if path.exists() else {'trials':[],'best':None}
    completed={t['index'] for t in state['trials']}
    for index,item in enumerate(recipes()):
        if index in completed: continue
        started=time.perf_counter()
        model=(IndependentSensorForecast(**item['spec']).fit(x,y) if item['kind']=='independent'
               else MultitaskSensorForecast(**item['spec']).fit(x,y,rul))
        correction=model.predict(sx); np.savez_compressed(run/f'prediction_{index:02}.npz',correction=correction)
        blends=[]
        for weight in (.25,.5,.75,1.):
            predictions=sp+(1.-weight)*old+weight*correction
            metrics={k:score(validation,actual,predictions,mask) for k,mask in masks.items()}
            eligible=all(improves(before[k],metrics[k]) and subset_gate(before[k],metrics[k]) for k in masks)
            blend={'weight':weight,'metrics':metrics,'eligible':eligible}; blends.append(blend)
            if eligible and (state['best'] is None or metrics['full']['r2']>state['best']['metrics']['full']['r2']):
                state['best']={'index':index,**item,**blend}; joblib.dump(model,run/'best_model.pkl',compress=3)
        state['trials'].append({'index':index,**item,'blends':blends,'seconds':time.perf_counter()-started})
        write(path,state)
        print('TRIAL',index,item,[(round(t['metrics']['full']['r2'],6),round(t['metrics']['healthy']['r2'],6),t['eligible']) for t in blends],
            'SECONDS',round(time.perf_counter()-started,1),flush=True)
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in config.paths.models.rglob('*') if p.is_file()}
    print('BEST',json.dumps(state['best']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN/'fast_modes'))
    with threadpool_limits(limits=4): main(parser.parse_args())
