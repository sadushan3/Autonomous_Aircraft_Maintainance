from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def multimode_features(frame, population, bundle):
    from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import commissioning_statistics
    compact_bundle = dict(bundle)
    compact_bundle.pop("feature_schema", None)
    base, ready = commissioning_statistics(frame, population, compact_bundle)
    targets = bundle["target_names"]
    values = pd.DataFrame(frame[targets].to_numpy(float)-population)
    values["unit"], values["mode"] = frame.unit.to_numpy(), frame.operating_mode.to_numpy()
    values = values.loc[frame.cycle.between(1,bundle["commissioning_cycles"]).to_numpy()]
    columns = list(range(len(targets)))
    grouped = values.groupby(["unit","mode"])[columns]
    keys = pd.MultiIndex.from_product([frame.unit.unique(),range(6)],names=["unit","mode"])
    means = grouped.mean().reindex(keys).fillna(0.)
    std = grouped.std().reindex(keys).fillna(0.)
    units = frame.unit.unique()
    wide_means = pd.DataFrame(means.to_numpy().reshape(len(units),-1),index=units).reindex(frame.unit).to_numpy()
    wide_std = pd.DataFrame(std.to_numpy().reshape(len(units),-1),index=units).reindex(frame.unit).to_numpy()
    return np.column_stack([base,wide_means,wide_std]), ready


class ReliabilityFactorEncoder:


    def __init__(self, rank=2, noise_factor=1., iterations=25, robust=False):
        self.rank, self.noise_factor, self.iterations, self.robust = rank, noise_factor, iterations, robust

    def _unpack(self, static):
        means, std = static[:,:126], static[:,126:252]
        counts = np.repeat(static[:,252:258],21,axis=1)
        return means, std, counts

    def fit(self, static):
        means,std,counts = self._unpack(static)
        available=counts>0
        weights0=counts*available
        center=(weights0*means).sum(axis=0)/np.maximum(weights0.sum(axis=0),1.)
        variance=(weights0*(means-center)**2).sum(axis=0)/np.maximum(weights0.sum(axis=0),1.)
        self.columns_=np.flatnonzero((variance>1e-8)&(available.sum(axis=0)>=4))
        if len(self.columns_)<1:
            raise ValueError("No informative commissioning views")
        means,std,counts=means[:,self.columns_],std[:,self.columns_],counts[:,self.columns_]
        freedom=np.maximum(counts-1.,0.)
        noise=(std**2*freedom).sum(axis=0)/np.maximum(freedom.sum(axis=0),1.)
        if self.robust:
            noise=np.array([np.median(std[counts[:,j]>=2,j]**2) if (counts[:,j]>=2).any() else noise[j]
                            for j in range(len(noise))])
        self.noise_=np.maximum(noise*self.noise_factor,np.maximum(variance[self.columns_]*1e-4,1e-8))
        precision=counts/self.noise_
        self.mean_=(precision*means).sum(axis=0)/np.maximum(precision.sum(axis=0),1.)
        observed=np.where(counts>0,means-self.mean_,0.)
        rank=min(self.rank, len(self.columns_),len(static)-1)
        pca=PCA(n_components=rank,random_state=42).fit(observed)
        self.loadings_=pca.components_.T*np.sqrt(np.maximum(pca.explained_variance_,1e-5))
        for _ in range(self.iterations):
            posterior,covariance=self._posterior(means,counts)
            augmented=np.column_stack([np.ones(len(means)),posterior])
            moment=augmented[:,:,None]*augmented[:,None,:]
            moment[:,1:,1:]+=covariance
            normal=np.einsum("np,nij->pij",precision,moment,optimize=True)
            normal+=np.eye(rank+1)[None,:,:]*1e-5
            rhs=np.einsum("np,np,ni->pi",precision,means,augmented,optimize=True)
            coefficients=np.linalg.solve(normal,rhs[:,:,None])[:,:,0]
            self.mean_,self.loadings_=coefficients[:,0],coefficients[:,1:]
        return self

    def _posterior(self, means, counts):
        precision=counts/self.noise_
        rank=self.loadings_.shape[1]
        normal=np.eye(rank)[None,:,:]+np.einsum("np,pi,pj->nij",precision,self.loadings_,self.loadings_,optimize=True)
        rhs=np.einsum("np,np,pi->ni",precision,means-self.mean_,self.loadings_,optimize=True)
        covariance=np.linalg.inv(normal)
        posterior=np.einsum("nij,nj->ni",covariance,rhs,optimize=True)
        return posterior,covariance

    def transform(self, static):
        means,_,counts=self._unpack(static)
        return self._posterior(means[:,self.columns_],counts[:,self.columns_])[0]


class ConditionProfileRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,rank=2,noise_factor=1.,robust=False,decoder="ridge",alpha=10.,
                 gamma=.01,age_features="quadratic",grouping="subset_mode",iterations=25,clusters=1):
        self.rank=rank
        self.noise_factor=noise_factor
        self.robust=robust
        self.decoder=decoder
        self.alpha=alpha
        self.gamma=gamma
        self.age_features=age_features
        self.grouping=grouping
        self.iterations=iterations
        self.clusters=clusters

    @staticmethod
    def _static(x):


        return np.round(np.column_stack([x[:,234:486],x[:,228:234]]),10)

    def _encode(self,x):
        subset=np.argmax(x[:,10:14],axis=1)
        latent=np.zeros((len(x),self.rank))
        for label,encoder in self.encoders_.items():
            rows=np.flatnonzero(subset==label)
            if not len(rows):
                continue
            unique,inverse=np.unique(self._static(x[rows]),axis=0,return_inverse=True)
            encoded=encoder.transform(unique)[inverse]
            latent[rows,:encoded.shape[1]]=encoded
        age=x[:,3:4]
        blocks=[x[:,:14],latent,age*latent]
        if self.grouping=="subset":
            blocks.append((latent[:,:,None]*x[:,None,4:10]).reshape(len(x),-1))
        if self.age_features in ("quadratic","spline"):
            blocks.extend([age**2,age**2*latent,latent**2])
        if self.age_features=="spline":
            for knot in (.1,.2,.35,.5,.7):
                hinge=np.maximum(age-knot,0.)
                blocks.extend([hinge,hinge*latent])
        return np.column_stack(blocks)

    def _labels(self,x):
        subset=np.argmax(x[:,10:14],axis=1)
        return subset*6+np.argmax(x[:,4:10],axis=1) if self.grouping=="subset_mode" else subset

    def fit(self,x,y):
        x,y=np.asarray(x,float),np.asarray(y,float)
        if x.shape[1]!=486:
            raise ValueError("Shared condition model needs the frozen 486-column multimode profile")
        self.n_features_in_,self.n_outputs_=x.shape[1],y.shape[1]
        self.encoders_={}
        subset=np.argmax(x[:,10:14],axis=1)
        for label in np.unique(subset):
            static=np.unique(self._static(x[subset==label]),axis=0)
            self.encoders_[int(label)]=ReliabilityFactorEncoder(self.rank,self.noise_factor,self.iterations,self.robust).fit(static)
        features=self._encode(x)
        self.fallback_=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(features,y)
        self.models_={}
        labels=self._labels(x)
        for label in np.unique(labels):
            rows=labels==label
            scaler=StandardScaler().fit(features[rows])
            center=y[rows].mean(axis=0)
            model=(Ridge(alpha=self.alpha) if self.decoder=="ridge" else
                   KernelRidge(alpha=self.alpha,kernel="rbf",gamma=self.gamma))
            model.fit(scaler.transform(features[rows]),y[rows]-center)
            self.models_[int(label)]=(scaler,center,model)
        self.gates_,self.mixture_models_={},{}
        if self.clusters>1:
            for subset_label in np.unique(subset):
                rows=subset==subset_label
                latent=features[rows,14:14+self.rank]
                unique=np.unique(np.round(latent,10),axis=0)
                gate=make_pipeline(StandardScaler(),GaussianMixture(n_components=self.clusters,
                    covariance_type="full",reg_covar=1e-3,n_init=3,random_state=42)).fit(unique)
                self.gates_[int(subset_label)]=gate
            for label in np.unique(labels):
                rows=labels==label
                subset_label=int(subset[rows][0])
                probabilities=self.gates_[subset_label].predict_proba(features[rows,14:14+self.rank])
                scaler=StandardScaler().fit(features[rows])
                values=scaler.transform(features[rows])
                members=[]
                for component in range(self.clusters):
                    weights=np.maximum(probabilities[:,component],1e-6)
                    center=np.average(y[rows],axis=0,weights=weights)
                    model=(Ridge(alpha=self.alpha) if self.decoder=="ridge" else
                           KernelRidge(alpha=self.alpha,kernel="rbf",gamma=self.gamma))
                    model.fit(values,y[rows]-center,sample_weight=weights)
                    members.append((center,model))
                self.mixture_models_[int(label)]=(scaler,subset_label,members)
        return self

    def predict(self,x):
        x=np.asarray(x,float)
        features=self._encode(x)
        result=self.fallback_.predict(features)
        labels=self._labels(x)
        for label,(scaler,center,model) in self.models_.items():
            rows=np.flatnonzero(labels==label)
            for start in range(0,len(rows),512):
                batch=rows[start:start+512]
                result[batch]=model.predict(scaler.transform(features[batch]))+center
        for label,(scaler,subset_label,members) in getattr(self,"mixture_models_",{}).items():
            rows=np.flatnonzero(labels==label)
            for start in range(0,len(rows),512):
                batch=rows[start:start+512]
                if not len(batch):
                    continue
                values=scaler.transform(features[batch])
                probability=self.gates_[subset_label].predict_proba(features[batch,14:14+self.rank])
                result[batch]=sum(probability[:,i,None]*(member.predict(values)+center)
                                  for i,(center,member) in enumerate(members))
        if not np.isfinite(result).all():
            raise ValueError("Shared condition inference produced non-finite values")
        return result
