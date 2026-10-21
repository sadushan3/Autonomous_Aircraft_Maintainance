from __future__ import annotations
import numpy as np,pandas as pd
from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import prior_sensor_features

def fast_mode_features(frame,population,bundle,base_features=None,ready=None):
    base,ready=prior_sensor_features(frame,population,bundle,base_features=base_features,ready=ready,multimode=True)
    order=np.lexsort((frame.cycle.to_numpy(),frame.unit.to_numpy()))
    ordered=frame.iloc[order].reset_index(drop=True)
    unit,mode=ordered.unit,ordered.operating_mode
    values=pd.DataFrame(ordered[bundle['target_names']].to_numpy(float)-population[order])
    inclusive=values.groupby([unit,mode],sort=False).ewm(span=3,adjust=False).mean().droplevel([0,1]).reindex(values.index)
    blocks=[]
    for state in (inclusive.to_numpy(),values.to_numpy()):
        wide=np.full((len(frame),126),np.nan)
        for label in range(6):
            rows=mode.eq(label).to_numpy(); wide[rows,label*21:(label+1)*21]=state[rows]
        past=pd.DataFrame(wide).groupby(unit,sort=False).ffill().groupby(unit,sort=False).shift(1)
        blocks.append(past.fillna(0.).to_numpy())
    previous_mode=mode.groupby(unit,sort=False).shift(1).fillna(-1).to_numpy(int)
    one_hot=np.zeros((len(frame),6)); valid=(previous_mode>=0)&(previous_mode<6)
    one_hot[np.arange(len(frame))[valid],previous_mode[valid]]=1.
    previous_ops=ordered[[f'op_setting_{i}' for i in (1,2,3)]].groupby(unit,sort=False).shift(1).fillna(0.).to_numpy()
    blocks.extend([one_hot,previous_ops])
    return np.column_stack([base,np.column_stack(blocks)[np.argsort(order)]]),ready
