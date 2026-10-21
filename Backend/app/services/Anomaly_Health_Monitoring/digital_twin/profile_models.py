from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline


class ProfileRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, family="kernel", grouping="subset", features="all",
                 alpha=1., gamma=.001, components=12, depth=None, leaf=10,
                 n_estimators=200, age_weight=1., profile_weight=1., random_state=42):
        self.family = family
        self.grouping = grouping
        self.features = features
        self.alpha = alpha
        self.gamma = gamma
        self.components = components
        self.depth = depth
        self.leaf = leaf
        self.n_estimators = n_estimators
        self.age_weight = age_weight
        self.profile_weight = profile_weight
        self.random_state = random_state

    def _labels(self, x):
        subset = np.argmax(x[:, 10:14], axis=1)
        mode = np.argmax(x[:, 4:10], axis=1)
        if self.grouping == "subset_mode":
            return subset * 6 + mode
        if self.grouping == "mode":
            return mode
        if self.grouping == "subset":
            return subset
        if self.grouping == "none":
            return np.zeros(len(x), dtype=int)
        raise ValueError("Unknown profile grouping")

    def _features(self, x):
        age = x[:, 3:4]
        global_profile, mode_profile = x[:, 38:59], x[:, 59:80]

        channels = np.array([2,3,4,7,8,9,11,12,13,14,15,17,20,21]) - 1
        if self.features == "all":
            return x
        if self.features == "denoised":
            return np.column_stack([x[:, :38], global_profile[:, channels], mode_profile[:, channels],
                                    age * global_profile[:, channels], age * mode_profile[:, channels]])
        if self.features == "global":
            return np.column_stack([x[:, :14], global_profile, age * global_profile])
        if self.features == "local":
            return np.column_stack([x[:, :14], global_profile, mode_profile,
                                    age * global_profile, age * mode_profile])
        if self.features == "interactions":
            return np.column_stack([x, age * mode_profile,
                (global_profile[:,:,None] * x[:,None,4:10]).reshape(len(x), -1),
                (global_profile[:,:,None] * x[:,None,10:14]).reshape(len(x), -1)])
        if self.features == "spline":
            return np.column_stack([x, age * mode_profile] +
                [np.maximum(age-knot, 0.) * block
                 for knot in (.1, .2, .3, .5, .7) for block in (global_profile, mode_profile)])
        if self.features == "latent":
            return np.column_stack([x[:, :14], global_profile, mode_profile])
        raise ValueError("Unknown profile feature set")

    def _estimator(self, rows, columns):
        if self.family == "kernel":
            return KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)
        if self.family == "pca_kernel":
            return make_pipeline(PCA(n_components=min(self.components, columns, rows - 1), random_state=self.random_state),
                KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma))
        if self.family == "ridge":
            return Ridge(alpha=self.alpha)
        if self.family == "pls":
            return PLSRegression(n_components=min(self.components, columns, rows-1), scale=False)
        if self.family == "extra_trees":
            return ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.depth,
                min_samples_leaf=self.leaf, n_jobs=4, random_state=self.random_state)
        if self.family == "lightgbm":
            from lightgbm import LGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(LGBMRegressor(n_estimators=self.n_estimators,
                num_leaves=self.components, learning_rate=.025, min_child_samples=self.leaf,
                reg_lambda=self.alpha, n_jobs=4, verbosity=-1, random_state=self.random_state))
        if self.family == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=self.n_estimators, max_depth=self.depth or 3,
                learning_rate=.025, min_child_weight=self.leaf, reg_lambda=self.alpha,
                n_jobs=4, tree_method="hist", random_state=self.random_state)
        if self.family == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                alpha=self.alpha, max_iter=500, early_stopping=False, random_state=self.random_state)
        raise ValueError("Unknown profile model family")

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.n_features_in_, self.n_outputs_ = x.shape[1], y.shape[1]
        features = self._features(x)
        self.fallback_ = make_pipeline(StandardScaler(), Ridge(alpha=100.)).fit(features, y)
        self.models_ = {}
        labels = self._labels(x)
        for label in np.unique(labels):
            rows = labels == label
            scaler = StandardScaler().fit(features[rows])
            values = scaler.transform(features[rows])
            values[:, 3] *= self.age_weight
            if self.features == "all":
                values[:, 38:80] *= self.profile_weight
            center = y[rows].mean(axis=0)
            scale = np.maximum(y[rows].std(axis=0), .01)
            model = self._estimator(rows.sum(), features.shape[1]).fit(values, (y[rows]-center)/scale)
            self.models_[int(label)] = (scaler, center, scale, model)
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        features, labels = self._features(x), self._labels(x)
        prediction = self.fallback_.predict(features)
        for label, (scaler, center, scale, model) in self.models_.items():
            indices = np.flatnonzero(labels == label)
            for start in range(0, len(indices), 512):
                batch = indices[start:start+512]
                values = scaler.transform(features[batch])
                values[:, 3] *= self.age_weight
                if self.features == "all":
                    values[:, 38:80] *= self.profile_weight
                prediction[batch] = model.predict(values) * scale + center
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite profile prediction")
        return prediction


class ColumnModel(RegressorMixin, BaseEstimator):

    def __init__(self, model=None, columns=101):
        self.model = model
        self.columns = columns

    def predict(self, x):
        return self.model.predict(np.asarray(x)[:, :self.columns])


def commissioning_statistics(frame, population, bundle, base_features=None, ready=None):

    from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
    if base_features is None:
        compact_bundle = dict(bundle)
        compact_bundle.pop("feature_schema", None)
        base, ready = EnsembleTwin._profile_features(frame, population, compact_bundle)
    else:
        base = base_features
    sensor_count = len(bundle["target_names"])
    values = pd.DataFrame(frame[bundle["target_names"]].to_numpy(float) - population)
    values["unit"] = frame.unit.to_numpy()
    values["mode"] = frame.operating_mode.to_numpy()
    values["cycle"] = frame.cycle.to_numpy()
    values = values.loc[frame.cycle.between(1, bundle["commissioning_cycles"]).to_numpy()]
    columns = list(range(sensor_count))
    keys = pd.MultiIndex.from_arrays([frame.unit, frame.operating_mode], names=["unit", "mode"])
    global_group = values.groupby("unit")[columns]
    mode_group = values.groupby(["unit", "mode"])[columns]
    global_std = global_group.std().reindex(frame.unit).fillna(0.).to_numpy()
    mode_std = mode_group.std().reindex(keys).fillna(0.).to_numpy()
    global_median = global_group.median().reindex(frame.unit).fillna(0.).to_numpy()
    mode_median = mode_group.median().reindex(keys).fillna(0.).to_numpy()
    mode_count = mode_group.count()[0].reindex(keys).fillna(0.).to_numpy()[:, None]
    by_mode = values.groupby(["unit", "mode"]).size().unstack(fill_value=0).reindex(columns=range(6), fill_value=0)
    mode_counts = by_mode.reindex(frame.unit).fillna(0.).to_numpy()
    early = values.loc[values.cycle.le(10)].groupby("unit")[columns].mean()
    late = values.loc[values.cycle.gt(20)].groupby("unit")[columns].mean()
    change = (late - early).reindex(frame.unit).fillna(0.).to_numpy()

    centered = values.cycle - values.groupby("unit").cycle.transform("mean")
    numerator = values[columns].mul(centered, axis=0).groupby(values.unit).sum()
    denominator = centered.pow(2).groupby(values.unit).sum().clip(lower=1.)
    slope = numerator.div(denominator, axis=0).reindex(frame.unit).fillna(0.).to_numpy()
    return np.column_stack([base, global_std, mode_std, global_median, mode_median,
                            change, slope, mode_count, mode_counts]), ready
