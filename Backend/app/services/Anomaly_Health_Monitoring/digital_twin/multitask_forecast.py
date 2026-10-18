from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class MultitaskSensorForecast(RegressorMixin,BaseEstimator):
    def __init__(self,auxiliary_weight=1.,grouping='subset',alpha=1.,max_iter=150,history='standard'):
        self.auxiliary_weight,self.grouping,self.alpha,self.max_iter=auxiliary_weight,grouping,alpha,max_iter
        self.history=history

    def features(self,x):
        x=np.asarray(x,float)
        fast=getattr(self,'history','standard')=='fast_modes'
        if x.ndim!=2 or x.shape[1]!=(952 if fast else 427) or not np.isfinite(x).all(): raise ValueError('Incorrect causal feature shape')
        blocks=[x[:,:14],x[:,38:80],x[:,234:427]]
        if fast: blocks.append(x[:,691:])
        return np.column_stack(blocks)

    def labels(self,x):
        return np.argmax(x[:,10:14],axis=1) if self.grouping=='subset' else np.zeros(len(x),int)

    def fit(self,x,y,rul=None):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if self.grouping not in ('subset','all') or self.auxiliary_weight<0:
            raise ValueError('Unknown multitask forecast policy')
        if getattr(self,'history','standard') not in ('standard','fast_modes'):
            raise ValueError('Unknown forecast history schema')
        if y.shape!=(len(x),21) or not np.isfinite(y).all(): raise ValueError('Expected 21 sensor targets')
        if self.auxiliary_weight and (rul is None or len(rul)!=len(x) or not np.isfinite(rul).all()):
            raise ValueError('Auxiliary lifetime labels are required only during fitting')
        f=self.features(x); self.selector_=VarianceThreshold(1e-10).fit(f)
        f=self.selector_.transform(f)
        self.fallback_=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(f,y)
        labels=self.labels(x); self.models_={}; self.n_features_in_,self.n_outputs_=x.shape[1],21
        for label in np.unique(labels):
            rows=labels==label
            scaler=StandardScaler().fit(f[rows]); values=np.clip(scaler.transform(f[rows]),-10.,10.)
            center=y[rows].mean(axis=0); scale=np.maximum(y[rows].std(axis=0),.05)
            targets=(y[rows]-center)/scale
            if self.auxiliary_weight:
                aux=np.minimum(np.maximum(np.asarray(rul)[rows],0.),130.)/130.
                aux=(aux-aux.mean())/max(float(aux.std()),.05)
                targets=np.column_stack([targets,aux*self.auxiliary_weight])
            model=MLPRegressor(hidden_layer_sizes=(64,32),activation='tanh',alpha=self.alpha,
                max_iter=self.max_iter,batch_size=256,learning_rate_init=.001,
                early_stopping=True,validation_fraction=.15,n_iter_no_change=15,random_state=20261016).fit(values,targets)
            self.models_[int(label)]=(scaler,center,scale,model)
        return self

    def predict(self,x):
        x=np.asarray(x,float); f=self.selector_.transform(self.features(x)); labels=self.labels(x)
        output=self.fallback_.predict(f)
        for label,(scaler,center,scale,model) in self.models_.items():
            rows=labels==label
            if rows.any():
                values=np.clip(scaler.transform(f[rows]),-10.,10.)
                output[rows]=model.predict(values)[:,:21]*scale+center
        if not np.isfinite(output).all(): raise ValueError('Non-finite neural sensor forecast')
        return output
