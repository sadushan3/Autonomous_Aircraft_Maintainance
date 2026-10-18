from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_approximation import Nystroem


def prior_sensor_features(frame,population,bundle,base_features=None,ready=None,multimode=False):
    from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import commissioning_statistics
    base_bundle=dict(bundle)
    base_bundle.pop("feature_schema",None)
    base,ready=commissioning_statistics(frame,population,base_bundle,base_features=base_features,ready=ready)
    order=np.lexsort((frame.cycle.to_numpy(),frame.unit.to_numpy()))
    ordered=frame.iloc[order].reset_index(drop=True)
    if ordered.duplicated(["unit","cycle"]).any():
        raise ValueError("Causal forecast requires unique engine/cycle observations")
    values=pd.DataFrame(ordered[bundle["target_names"]].to_numpy(float)-population[order])
    unit,mode=ordered.unit,ordered.operating_mode
    global_last=values.groupby(unit,sort=False).shift(1)
    local_last=values.groupby([unit,mode],sort=False).shift(1)
    blocks=[]
    global_ewm={}
    for span in (3,10,30,100):
        inclusive=values.groupby(unit,sort=False).ewm(span=span,adjust=False).mean().droplevel(0)
        past=inclusive.groupby(unit,sort=False).shift(1).reindex(values.index)
        global_ewm[span]=past
        blocks.append(past.fillna(0.).to_numpy())
    for span in (3,10,30):
        inclusive=values.groupby([unit,mode],sort=False).ewm(span=span,adjust=False).mean().droplevel([0,1])
        past=inclusive.groupby([unit,mode],sort=False).shift(1).reindex(values.index)
        blocks.append(past.fillna(global_ewm[span]).fillna(0.).to_numpy())
    blocks.extend([global_last.fillna(0.).to_numpy(),local_last.fillna(global_last).fillna(0.).to_numpy()])
    previous_cycle=ordered.groupby("unit",sort=False).cycle.shift(1)
    previous_mode_cycle=ordered.groupby(["unit","operating_mode"],sort=False).cycle.shift(1)
    counts=ordered.groupby("unit",sort=False).cumcount().to_numpy()
    mode_counts=ordered.groupby(["unit","operating_mode"],sort=False).cumcount().to_numpy()
    blocks.append(np.column_stack([(ordered.cycle-previous_cycle).fillna(0).to_numpy(),
        (ordered.cycle-previous_mode_cycle).fillna(30).to_numpy(),np.minimum(counts,100),np.minimum(mode_counts,100)]))
    temporal=np.column_stack(blocks)
    if multimode:
        mode_blocks=[]
        for span in (10,30):
            inclusive=values.groupby([unit,mode],sort=False).ewm(span=span,adjust=False).mean().droplevel([0,1]).reindex(values.index).to_numpy()
            wide=np.full((len(values),126),np.nan)
            for mode_id in range(6):
                rows=mode.eq(mode_id).to_numpy()
                wide[rows,mode_id*21:(mode_id+1)*21]=inclusive[rows]


            state=pd.DataFrame(wide).groupby(unit,sort=False).ffill().groupby(unit,sort=False).shift(1)
            mode_blocks.append(state.fillna(0.).to_numpy())
        count_matrix=np.full((len(values),6),np.nan)
        time_matrix=np.full((len(values),6),np.nan)
        for mode_id in range(6):
            rows=mode.eq(mode_id).to_numpy()
            count_matrix[rows,mode_id]=mode_counts[rows]+1
            time_matrix[rows,mode_id]=ordered.loc[rows,"cycle"]
        counts_by_mode=pd.DataFrame(count_matrix).groupby(unit,sort=False).ffill().groupby(unit,sort=False).shift(1)
        last_times=pd.DataFrame(time_matrix).groupby(unit,sort=False).ffill().groupby(unit,sort=False).shift(1)
        gaps=last_times.rsub(ordered.cycle,axis=0).fillna(30.)
        mode_blocks.extend([counts_by_mode.fillna(0.).clip(upper=100).to_numpy(),gaps.clip(upper=100).to_numpy()])
        temporal=np.column_stack([temporal,*mode_blocks])
    inverse=np.argsort(order)
    return np.column_stack([base,temporal[inverse]]),ready


class CausalSensorRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,alpha=100.,grouping="subset_mode",features="all",family="ridge",leaf=100,depth=None,gamma=.001,components=256):
        self.alpha,self.grouping,self.features=alpha,grouping,features
        self.family,self.leaf,self.depth,self.gamma,self.components=family,leaf,depth,gamma,components

    def _transform(self,x):
        width=691 if self.features=="multimode" else 427
        if x.ndim!=2 or x.shape[1]!=width or not np.isfinite(x).all():
            raise ValueError(f"Causal sensor history requires {width} finite feature columns")
        history=x[:,234:]
        if self.features=="all":
            keep=history
        elif self.features=="combined":
            keep=np.column_stack([x[:,38:234],history])
        elif self.features=="multimode":
            keep=np.column_stack([x[:,38:80],history[:,:84],x[:,427:]])
        elif self.features=="smooth":
            keep=np.column_stack([history[:,42:84],history[:,105:147],history[:,-4:]])
        elif self.features=="global":
            keep=np.column_stack([history[:,:84],history[:,147:168],history[:,-4:]])
        elif self.features=="local":
            keep=np.column_stack([history[:,84:147],history[:,168:189],history[:,-4:]])
        else:
            raise ValueError("Unknown causal forecast feature set")
        return np.column_stack([x[:,:14],keep])

    def _labels(self,x):
        subset=np.argmax(x[:,10:14],axis=1)
        return subset*6+np.argmax(x[:,4:10],axis=1) if self.grouping=="subset_mode" else subset

    def fit(self,x,y):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if self.grouping not in ("subset_mode","subset") or self.family not in ("ridge","kernel","extra_trees"):
            raise ValueError("Unknown causal forecast model policy")
        if self.alpha<=0 or y.ndim!=2 or len(x)!=len(y) or not np.isfinite(y).all():
            raise ValueError("Invalid causal forecast fitting data")
        self.n_features_in_=x.shape[1]
        self.n_outputs_=y.shape[1]
        features=self._transform(x)
        self.fallback_=make_pipeline(StandardScaler(),Ridge(alpha=self.alpha)).fit(features,y)
        self.models_={}
        labels=self._labels(x)
        for label in np.unique(labels):
            rows=labels==label
            if getattr(self,"family","ridge")=="extra_trees":
                model=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=self.leaf,max_depth=self.depth,
                    max_features=.8,n_jobs=4,random_state=42)
            elif self.family=="kernel":
                model=make_pipeline(StandardScaler(),Nystroem(gamma=self.gamma,n_components=min(self.components,int(rows.sum())),random_state=42),Ridge(alpha=self.alpha))
            else:
                model=make_pipeline(StandardScaler(),Ridge(alpha=self.alpha))
            self.models_[int(label)]=model.fit(features[rows],y[rows])
        return self

    def predict(self,x):
        x=np.asarray(x,float)
        features=self._transform(x)
        output=self.fallback_.predict(features)
        labels=self._labels(x)
        for label,model in self.models_.items():
            rows=labels==label
            if rows.any():
                output[rows]=model.predict(features[rows])
        if not np.isfinite(output).all():
            raise ValueError("Causal forecast produced non-finite predictions")
        return output
