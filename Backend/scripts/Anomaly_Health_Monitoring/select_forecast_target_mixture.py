from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import score,improves,subset_gate,write
from app.services.Anomaly_Health_Monitoring.digital_twin.forecast_policy import healthy_forecast_guard,DEFAULT_HEALTHY_GUARD

def fit_weights(actual,baseline,candidates,healthy,penalty,healthy_weight,budget=0.):
    weights=np.zeros((len(candidates),actual.shape[1]))
    full_var=actual.var(axis=0); healthy_var=actual[healthy].var(axis=0)
    for sensor in np.flatnonzero((full_var>1e-8)&(healthy_var>1e-8)):
        d=np.column_stack([p[:,sensor]-baseline[:,sensor] for p in candidates])
        residual=actual[:,sensor]-baseline[:,sensor]
        full_h=d.T@d/len(d)/full_var[sensor]; full_l=d.T@residual/len(d)/full_var[sensor]
        dh,rh=d[healthy],residual[healthy]
        health_h=dh.T@dh/len(dh)/healthy_var[sensor]; health_l=dh.T@rh/len(dh)/healthy_var[sensor]
        hessian=full_h+healthy_weight*health_h+penalty*np.eye(len(candidates))
        linear=full_l+healthy_weight*health_l
        constraints=[{'type':'ineq','fun':lambda w:1.-w.sum(),'jac':lambda w:-np.ones_like(w)},
            {'type':'ineq','fun':lambda w:budget+2*health_l@w-w@health_h@w,
             'jac':lambda w:2*health_l-2*health_h@w}]
        solved=minimize(lambda w:w@hessian@w-2*linear@w,np.zeros(len(candidates)),
            jac=lambda w:2*hessian@w-2*linear,bounds=[(0.,1.)]*len(candidates),
            constraints=constraints,method='SLSQP',options={'ftol':1e-10,'maxiter':250})
        w=np.maximum(solved.x,0.)
        if w.sum()>1.: w/=w.sum()
        if w@health_h@w-2*health_l@w>budget+1e-7: continue
        if not solved.success and np.linalg.norm(w)<1e-6: continue
        weights[:,sensor]=w
    return weights

def mixture(baseline,candidates,weights):
    output=baseline.copy()
    for candidate,weight in zip(candidates,weights): output+=(candidate-baseline)*weight
    return output

def select(args):
    run=Path(args.run_dir).resolve()
    if (run/'model_lock.json').exists(): raise ValueError('Candidate frozen; no confirmation retuning')
    state=json.loads((run/'independent/search.json').read_text())
    if len(state['trials'])!=16: raise ValueError('Independent search is not complete')
    pool=[]
    for family in ('ridge','lightgbm'):
        for features in ('own','shared'):
            trials=[trial for trial in state['trials'] if trial['spec']['family']==family and trial['spec']['features']==features]
            best=max(trials,key=lambda t:t['blends'][-1]['metrics']['full']['r2'])
            pool.append({'source':'independent','index':best['index'],'spec':best['spec']})
    pool.extend([{'source':'baseline_member','index':0},{'source':'baseline_member','index':2}])
    neural_path=run/'multitask/search.json'
    if neural_path.exists():
        neural=json.loads(neural_path.read_text())
        if len(neural['trials'])!=6: raise ValueError('Neural search is not complete')
        for auxiliary in (False,True):
            trials=[t for t in neural['trials'] if bool(t['spec']['auxiliary_weight'])==auxiliary]
            best=max(trials,key=lambda t:t['blends'][-1]['metrics']['full']['r2'])
            pool.append({'source':'multitask','index':best['index'],'spec':best['spec']})
    fast_path=run/'fast_modes/search.json'
    if fast_path.exists() and not args.without_fast:
        fast=json.loads(fast_path.read_text())
        if len(fast['trials'])!=10: raise ValueError('Fast history search is not complete')
        for kind,family in (('independent','ridge'),('independent','lightgbm'),('multitask',None)):
            trials=[t for t in fast['trials'] if t['kind']==kind and (family is None or t['spec']['family']==family)]
            best=max(trials,key=lambda t:t['blends'][-1]['metrics']['full']['r2'])
            pool.append({'source':'fast_modes','index':best['index'],'kind':kind,'spec':best['spec']})
    data=np.load(run/'selection_data.npz')
    actual,base=data['actual'],data['population']+data['baseline_correction']
    healthy=data['healthy']; frame=pd.DataFrame({'unit':data['units'],'subset_id':data['subsets']})
    predictions=[]
    for member in pool:
        path=(run/member['source']/f"prediction_{member['index']:02}.npz" if member['source'] in ('independent','multitask','fast_modes')
              else run/f"candidate_{member['index']:02}.npz")
        predictions.append(data['population']+np.load(path)['correction'])
    masks={'healthy':healthy,'full':np.ones(len(actual),bool)}
    before={k:score(frame,actual,base,mask) for k,mask in masks.items()}
    units=frame.groupby('unit').subset_id.first().sort_index()
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=20261018).split(units.index,units))
    trials=[]; best=None
    for budget in (0.,.003):
      for healthy_weight in (0.,.1,1.,10.):
        for penalty in (.001,.05):
            oof=base.copy()
            for train_ids,held_ids in splits:
                fit=frame.unit.isin(units.index[train_ids]).to_numpy()
                held=frame.unit.isin(units.index[held_ids]).to_numpy()
                weights=fit_weights(actual[fit],base[fit],[p[fit] for p in predictions],healthy[fit],penalty,healthy_weight,budget)
                oof[held]=mixture(base[held],[p[held] for p in predictions],weights)
            metrics={k:score(frame,actual,oof,mask) for k,mask in masks.items()}
            eligible=(improves(before['full'],metrics['full']) and healthy_forecast_guard(before['healthy'],metrics['healthy'])
                and all(subset_gate(before[k],metrics[k]) for k in masks))
            trial={'healthy_weight':healthy_weight,'penalty':penalty,'budget':budget,'metrics':metrics,'eligible':eligible}
            trials.append(trial)
            print('OOF',budget,healthy_weight,penalty,'full',metrics['full']['r2'],'healthy',metrics['healthy']['r2'],'eligible',eligible,flush=True)
            if eligible and (best is None or metrics['full']['r2']>best['metrics']['full']['r2']): best=trial
    result={'baseline':before,'pool':pool,'trials':trials,'best':best,'objective':'full_trajectory_forecasts',
        'healthy_guard':DEFAULT_HEALTHY_GUARD,
        'target_r2':.8,'raw_targets_and_rows_unchanged':True,'confirmation_or_official_used':False}
    if best:
        weights=fit_weights(actual,base,predictions,healthy,best['penalty'],best['healthy_weight'],best['budget'])
        final=mixture(base,predictions,weights)
        result['weights']=weights.tolist()
        result['metrics']={k:score(frame,actual,final,mask) for k,mask in masks.items()}
        result['eligible']=(improves(before['full'],result['metrics']['full'])
            and healthy_forecast_guard(before['healthy'],result['metrics']['healthy'])
            and all(subset_gate(before[k],result['metrics'][k]) for k in masks))
    write(run/args.output_name,result)
    print('SELECTED',json.dumps(best),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN))
    parser.add_argument('--output-name',default='mixture_selection.json')
    parser.add_argument('--without-fast',action='store_true')
    with threadpool_limits(limits=4): select(parser.parse_args())
