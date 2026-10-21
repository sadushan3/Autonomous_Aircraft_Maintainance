from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.Anomaly_Health_Monitoring.probe_forecast_target import RUN
from scripts.Anomaly_Health_Monitoring.optimize_startup_profile import ROOT,SOURCE,write
from scripts.Anomaly_Health_Monitoring.evaluate_linear_baseline import prepare
from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import OperatingModeDetector
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split

def main(args):
    run=Path(args.run_dir).resolve()
    if not (run/'model_lock.json').exists() or not (run/'official_comparison.json').exists():
        raise ValueError('Noise diagnosis is report-only after locking and official scoring')
    config=Config(); config.mlops.log_to_file=config.mlops.enable_mlflow=False
    plan=load_training_split(config); targets=[f'sensor_{i}' for i in range(1,22)]
    source=joblib.load(SOURCE/'prepared.joblib')['scaled']
    fitting=source.loc[source.unit.isin(plan['engine_roles']['fit'])&source.RUL.ge(130)&source.cycle.gt(30)].copy()
    fitting=fitting.sort_values(['unit','operating_mode','cycle']).reset_index(drop=True)
    context_bundle={'n_modes':6,'subsets':['FD001','FD002','FD003','FD004']}
    context=EnsembleTwin._profile_context(fitting,context_bundle)
    expected=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(context,fitting[targets]).predict(context)
    residual=pd.DataFrame(fitting[targets].to_numpy(float)-expected,columns=targets)
    group=[fitting.unit,fitting.operating_mode]
    first=residual.groupby(group,sort=False).diff()
    second=first.groupby(group,sort=False).diff()
    gaps=fitting.cycle.groupby(group,sort=False).diff()
    consecutive=gaps.eq(1)&gaps.groupby(group,sort=False).shift(1).eq(1)
    squared=second.loc[consecutive].pow(2)/6.
    observed=fitting.loc[consecutive,['unit','subset_id','operating_mode']].copy()
    squared['unit']=observed.unit; squared['subset_id']=observed.subset_id; squared['mode']=observed.operating_mode
    raw=CMAPSSReader(config).read_subsets(split='test')
    official=prepare(raw,config,OperatingModeDetector(config).load(),PerModeScaler(config).load_scalers())
    prefix_counts=official.loc[official.cycle.between(1,30)].groupby('unit').cycle.nunique()
    ready=official.unit.map(prefix_counts).fillna(0).eq(30)&official.cycle.gt(30)
    evaluation=official.loc[ready].copy(); actual=evaluation[targets].to_numpy(float)
    variance=actual.var(axis=0); informative=variance>1e-8
    units=sorted(plan['engine_roles']['fit'])
    rng=np.random.default_rng(20261019)
    weights=np.vstack([np.ones(len(units)),np.array([np.bincount(rng.integers(len(units),size=len(units)),minlength=len(units)) for _ in range(1000)])])
    estimated=np.zeros((len(weights),21)); group_details=[]
    for (subset,mode),rows in evaluation.groupby(['subset_id','operating_mode']):
        candidates=squared.loc[squared.subset_id.eq(subset)&squared['mode'].eq(mode)]
        fallback=candidates.empty
        if fallback: candidates=squared.loc[squared.subset_id.eq(subset)]
        if candidates.empty: raise ValueError(f'No fitting noise observations for {subset}/{mode}')
        sums=candidates.groupby('unit')[targets].sum().reindex(units,fill_value=0.).to_numpy()
        counts=candidates.groupby('unit').size().reindex(units,fill_value=0.).to_numpy()
        numerator=weights@sums; denominator=weights@counts
        noise=numerator/np.maximum(denominator[:,None],1.)
        noise[denominator==0]=sums.sum(axis=0)/max(float(counts.sum()),1.)
        estimated+=len(rows)/len(evaluation)*noise
        group_details.append({'subset':str(subset),'mode':int(mode),'fit_triples':len(candidates),
            'official_rows':len(rows),'subset_fallback':fallback})
    proxies=(1.-estimated[:,informative]/variance[informative]).mean(axis=1)
    result={'report_only_after_candidate_lock':True,'target_r2':.8,
        'estimator':'same-mode consecutive second differences / 6, after fitting-only context regression',
        'fit_engine_count':len(units),'fit_triples':int(consecutive.sum()),'official_rows':len(evaluation),
        'estimated_noise_limited_full_r2':float(proxies[0]),
        'engine_bootstrap_95_percent_interval':np.quantile(proxies[1:],[.025,.975]).tolist(),
        'bootstrap_replicates':1000,
        'assumptions':['Measurement errors are independent across adjacent same-mode observations.',
            'The latent healthy signal has small curvature over three consecutive cycles.',
            'Fitting-engine normalized noise variance transfers to official engines conditional on subset and mode.',
            'The fitted context regression removes predictable short-term operating changes.'],
        'limitations':'Approximate diagnostic, not a strict upper bound. Correlated errors, latent curvature, context misspecification or changing noise variance can bias it. The bootstrap fixes official target variance and reflects fitting-noise estimation only; it excludes assumption error and historical benchmark reuse.',
        'targets_smoothed_or_removed':False,'noise_groups':group_details,
        'per_sensor':{name:{'informative':bool(informative[i]),'estimated_noise_variance':float(estimated[0,i]),
            'official_target_variance':float(variance[i]),'proxy_r2':float(1.-estimated[0,i]/max(variance[i],1e-30))}
            for i,name in enumerate(targets)}}
    write(run/'noise_diagnostic.json',result)
    print('NOISE DIAGNOSTIC',json.dumps({k:result[k] for k in ('estimated_noise_limited_full_r2','engine_bootstrap_95_percent_interval','fit_triples','official_rows')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',default=str(RUN))
    with threadpool_limits(limits=4): main(parser.parse_args())
