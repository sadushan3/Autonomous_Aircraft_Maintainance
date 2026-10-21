from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class IndependentSensorForecast(RegressorMixin,BaseEstimator):
    def __init__(self,family='ridge',grouping='subset_mode',features='own',alpha=100.,
                 n_estimators=200,leaves=15,leaf=50,learning_rate=.035,history='standard'):
        self.family,self.grouping,self.features,self.alpha=family,grouping,features,alpha
        self.n_estimators,self.leaves,self.leaf,self.learning_rate=n_estimators,leaves,leaf,learning_rate
        self.history=history

    def width(self):
        return 952 if getattr(self,'history','standard')=='fast_modes' else 691

    def columns(self,sensor):
        columns=list(range(14))+[start+sensor for start in (38,59,80,101,122,143,164,185,206)]
        columns+=list(range(227,234))+[234+21*i+sensor for i in range(9)]+list(range(423,427))
        columns+=[427+21*i+sensor for i in range(12)]+list(range(679,691))
        if self.features=='shared':
            columns+=list(range(38,80))+list(range(276,297))+list(range(339,360))
        elif self.features!='own': raise ValueError('Unknown independent forecast features')
        if getattr(self,'history','standard')=='fast_modes':
            columns+=[691+21*i+sensor for i in range(12)]+list(range(943,952))
        return np.unique(columns)

    def labels(self,x):
        subset=np.argmax(x[:,10:14],axis=1)
        return subset*6+np.argmax(x[:,4:10],axis=1) if self.grouping=='subset_mode' else subset

    def fit(self,x,y):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if self.grouping not in ('subset_mode','subset') or self.family not in ('ridge','lightgbm'):
            raise ValueError('Unknown independent forecast policy')
        if getattr(self,'history','standard') not in ('standard','fast_modes'):
            raise ValueError('Unknown forecast history schema')
        if x.ndim!=2 or x.shape[1]!=self.width() or y.shape!=(len(x),21): raise ValueError('Incorrect feature width or sensor target shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Non-finite forecast training data')
        self.n_features_in_,self.n_outputs_=self.width(),21
        self.columns_=[self.columns(sensor) for sensor in range(21)]
        self.fallback_columns_=np.unique(np.concatenate(self.columns_))
        self.fallback_=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(x[:,self.fallback_columns_],y)
        labels=self.labels(x); self.models_={}
        for label in np.unique(labels):
            rows=labels==label; members=[]
            group_x=x[rows]
            for sensor,columns in enumerate(self.columns_):
                target=y[rows,sensor]
                if target.var()<1e-8:
                    model=float(target.mean())
                elif self.family=='ridge':
                    model=make_pipeline(StandardScaler(),Ridge(alpha=self.alpha)).fit(group_x[:,columns],target)
                elif self.family=='lightgbm':
                    from lightgbm import LGBMRegressor
                    model=LGBMRegressor(n_estimators=self.n_estimators,num_leaves=self.leaves,
                        min_child_samples=self.leaf,learning_rate=self.learning_rate,reg_lambda=self.alpha,
                        n_jobs=4,verbosity=-1,random_state=20261016,deterministic=True,force_col_wise=True).fit(group_x[:,columns],target)
                else: raise ValueError('Unknown independent forecast family')
                members.append(model)
            self.models_[int(label)]=members
        return self

    def predict(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.width() or not np.isfinite(x).all(): raise ValueError('Incorrect forecast features')
        labels=self.labels(x)
        output=np.zeros((len(x),21))
        unknown=~np.isin(labels,list(self.models_))
        if unknown.any():
            columns=getattr(self,'fallback_columns_',np.unique(np.concatenate(self.columns_)))
            output[unknown]=self.fallback_.predict(x[unknown][:,columns])
        for label,members in self.models_.items():
            rows=labels==label
            if not rows.any(): continue
            group_x=x[rows]
            for sensor,(model,columns) in enumerate(zip(members,self.columns_)):
                output[rows,sensor]=model if isinstance(model,float) else model.predict(group_x[:,columns])
        if not np.isfinite(output).all(): raise ValueError('Non-finite sensor forecast')
        return output
