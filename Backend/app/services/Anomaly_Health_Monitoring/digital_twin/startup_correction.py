"""Serializable regressors for a frozen commissioning profile.

Inputs are the existing compact profile, never observations after commissioning.
The model owns its feature transform so experiment and serving use the same code.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline


class StartupCorrectionRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, family="kernel", features="compact", grouped=False,
                 alpha=1., gamma=.001, depth=8, leaf=10, balanced=False,
                 profile_weight=1., mode_weight=1., age_weight=1.,
                 n_estimators=200, random_state=42):
        self.family = family
        self.features = features
        self.grouped = grouped
        self.alpha = alpha
        self.gamma = gamma
        self.depth = depth
        self.leaf = leaf
        self.balanced = balanced
        self.profile_weight = profile_weight
        self.mode_weight = mode_weight
        self.age_weight = age_weight
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 101:
            raise ValueError("Startup correction requires the 101-column compact profile.")
        if self.features not in ("compact", "quadratic", "spline", "interactions", "profile"):
            raise ValueError(f"Unknown startup feature transform: {self.features}")
        age, global_profile, mode_profile = x[:, 3:4], x[:, 38:59], x[:, 59:80]
        if self.features == "compact":
            return x
        blocks = [x]
        if self.features in ("quadratic", "spline", "interactions"):
            blocks += [age * mode_profile, age**2 * global_profile,
                       age**2 * mode_profile, age * x[:, 14:38]]
        if self.features == "spline":
            for knot in (.15, .3, .45, .6):
                hinge = np.maximum(age - knot, 0.)
                blocks += [hinge * global_profile, hinge * mode_profile,
                           hinge * x[:, 14:38]]
        if self.features == "interactions":
            for context in (x[:, 4:10], x[:, 10:14]):
                blocks += [(global_profile[:, :, None] * context[:, None, :]).reshape(len(x), -1)]
        if self.features == "profile":
            return np.column_stack([x[:, :14], global_profile, mode_profile])
        return np.column_stack(blocks)

    def _estimator(self):
        if self.family == "kernel":
            return KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)
        if self.family == "pca_kernel":
            from sklearn.decomposition import PCA
            return make_pipeline(PCA(n_components=self.depth, random_state=self.random_state),
                                 KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma))
        if self.family == "nystroem":
            from sklearn.kernel_approximation import Nystroem
            return make_pipeline(Nystroem(kernel="rbf", gamma=self.gamma, n_components=512,
                                          random_state=self.random_state), Ridge(alpha=self.alpha))
        if self.family == "ridge":
            return Ridge(alpha=self.alpha)
        if self.family in ("extra_trees", "random_forest"):
            cls = ExtraTreesRegressor if self.family == "extra_trees" else RandomForestRegressor
            return cls(n_estimators=self.n_estimators, max_depth=self.depth,
                       min_samples_leaf=self.leaf, n_jobs=4, random_state=self.random_state)
        if self.family == "lightgbm":
            from lightgbm import LGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor
            return MultiOutputRegressor(LGBMRegressor(
                n_estimators=self.n_estimators, num_leaves=2**self.depth - 1,
                learning_rate=.035, min_child_samples=self.leaf, reg_lambda=self.alpha,
                n_jobs=4, verbosity=-1, random_state=self.random_state))
        if self.family == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=self.n_estimators, max_depth=self.depth,
                learning_rate=.035, min_child_weight=self.leaf, reg_lambda=self.alpha,
                subsample=.9, colsample_bytree=.9, tree_method="hist", n_jobs=4,
                random_state=self.random_state)
        if self.family == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                alpha=self.alpha, max_iter=400, early_stopping=True,
                random_state=self.random_state)
        if self.family == "pls":
            from sklearn.cross_decomposition import PLSRegression
            return PLSRegression(n_components=self.depth)
        raise ValueError(f"Unknown correction family: {self.family}")

    def _weighted(self, transformed, scaler):
        values = scaler.transform(transformed)
        # Feature scaling changes the kernel distance, without using any targets.
        values[:, 3:4] *= self.age_weight
        values[:, 38:59] *= self.profile_weight
        values[:, 59:80] *= self.mode_weight
        values[:, 80:101] *= self.age_weight
        return values

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        transformed = self._transform(x)
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1]
        self.models_ = {}
        # A global ridge supplies an explicit fallback if a subset had no fit rows.
        self.fallback_ = make_pipeline(StandardScaler(), Ridge(alpha=100.)).fit(transformed, y)
        labels = np.argmax(x[:, 10:14], axis=1) if self.grouped else np.zeros(len(x), int)
        for label in np.unique(labels):
            rows = labels == label
            scaler = StandardScaler().fit(transformed[rows])
            scale = np.maximum(np.std(y[rows], axis=0), .05) if self.balanced else np.ones(y.shape[1])
            model = self._estimator().fit(self._weighted(transformed[rows], scaler), y[rows] / scale)
            self.models_[int(label)] = (scaler, scale, model)
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        transformed = self._transform(x)
        prediction = self.fallback_.predict(transformed)
        labels = np.argmax(x[:, 10:14], axis=1) if self.grouped else np.zeros(len(x), int)
        for label, (scaler, scale, model) in self.models_.items():
            rows = np.flatnonzero(labels == label)
            for start in range(0, len(rows), 1024):
                batch = rows[start:start + 1024]
                prediction[batch] = model.predict(self._weighted(transformed[batch], scaler)) * scale
        if not np.isfinite(prediction).all():
            raise ValueError("Startup correction produced non-finite predictions.")
        return prediction


class BlendedStartupCorrection(RegressorMixin, BaseEstimator):
    """Fixed convex blend, with weights selected on training-engine validation."""
    def __init__(self, baseline=None, candidate=None, weight=.5):
        self.baseline = baseline
        self.candidate = candidate
        self.weight = weight

    def predict(self, x):
        if self.weight == 1.:
            return self.candidate.predict(x)
        if self.weight == 0.:
            return self.baseline.predict(x)
        return (1. - self.weight) * self.baseline.predict(x) + self.weight * self.candidate.predict(x)


class SensorBlendStartupCorrection(RegressorMixin, BaseEstimator):
    """A convex mixture with fixed validation-selected weights per sensor."""
    def __init__(self, baseline=None, members=None, weights=None):
        self.baseline = baseline
        self.members = members
        self.weights = weights

    def predict(self, x):
        baseline = self.baseline.predict(x)
        weights = np.asarray(self.weights, float)
        if weights.shape != (len(self.members), baseline.shape[1]):
            raise ValueError("Sensor mixture weights do not match models/targets")
        if not np.isfinite(weights).all() or (weights < 0).any() or (weights.sum(axis=0) > 1. + 1e-8).any():
            raise ValueError("Sensor mixture must be convex")
        prediction = baseline.copy()
        for model, weight in zip(self.members, weights):
            prediction += (model.predict(x) - baseline) * weight
        return prediction


def fit_sensor_blend(x, y, baseline, spec):
    models = [StartupCorrectionRegressor(**member).fit(x, y) for member in spec["members"]]
    return SensorBlendStartupCorrection(baseline, models, spec["weights"])
