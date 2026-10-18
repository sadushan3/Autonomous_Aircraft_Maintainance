from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features

WIDTH = 1141
_ENGINE_BATCH_SIZE = 24


def _trend_features(frame, population, bundle, base_features=None, ready=None):
    base, ready = fast_mode_features(frame, population, bundle, base_features, ready)
    order = np.lexsort((frame.cycle.to_numpy(), frame.unit.to_numpy()))
    ordered = frame.iloc[order].reset_index(drop=True)
    unit, mode = ordered.unit, ordered.operating_mode
    values = pd.DataFrame(ordered[bundle['target_names']].to_numpy(float) - population[order])
    time = ordered.cycle.astype(float).to_frame()
    global_keys, local_keys = unit, [unit, mode]

    def past_ewm(matrix, keys, span):
        levels = [0, 1] if isinstance(keys, list) else 0
        inclusive = matrix.groupby(keys, sort=False).ewm(span=span, adjust=False).mean().droplevel(levels)
        return inclusive.reindex(values.index).groupby(keys, sort=False).shift(1)

    def prior_mean(keys):
        count = values.groupby(keys, sort=False).cumcount().to_numpy(float)
        total = values.groupby(keys, sort=False).cumsum().groupby(keys, sort=False).shift(1)
        return total.div(np.maximum(count, 1), axis=0)

    global_mean = prior_mean(global_keys).fillna(0.)
    local_mean = prior_mean(local_keys).fillna(global_mean)
    blocks = [global_mean.to_numpy(), local_mean.to_numpy(),
              past_ewm(values, local_keys, 100).fillna(global_mean).fillna(0.).to_numpy()]
    global_trends = {}
    for local in (False, True):
        keys = local_keys if local else global_keys
        for span in (10, 30, 100):
            mean = past_ewm(values, keys, span).to_numpy()
            mt = past_ewm(time, keys, span).to_numpy()
            mtt = past_ewm(time ** 2, keys, span).to_numpy()
            mty = past_ewm(values.mul(time.iloc[:, 0], axis=0), keys, span).to_numpy()
            variance = np.maximum(mtt - mt ** 2, 0.)
            slope = np.divide(mty - mean * mt, variance,
                              out=np.zeros_like(mean), where=variance > 1e-6)
            estimate = mean + np.clip(slope, -1., 1.) * (time.to_numpy() - mt)
            if local:
                estimate = np.where(np.isfinite(estimate), estimate, global_trends[span])
            else:
                estimate = np.nan_to_num(estimate)
                global_trends[span] = estimate
            blocks.append(np.clip(estimate, -12., 12.))
    return np.column_stack([base, np.column_stack(blocks)[np.argsort(order)]]), ready


def engine_batches(frame, batch_size=_ENGINE_BATCH_SIZE):

    order = np.argsort(frame.unit.to_numpy(), kind='stable')
    units = frame.unit.to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(units[1:] != units[:-1]) + 1, len(frame)]
    for group in range(0, len(starts) - 1, batch_size):
        yield order[starts[group]:starts[min(group + batch_size, len(starts) - 1)]]


def trend_features(frame, population, bundle, base_features=None, ready=None):


    if frame.unit.nunique() <= _ENGINE_BATCH_SIZE:
        return _trend_features(frame, population, bundle, base_features, ready)
    features = np.empty((len(frame), WIDTH), dtype=float)
    result_ready = np.empty(len(frame), dtype=bool)
    for rows in engine_batches(frame):
        features[rows], result_ready[rows] = _trend_features(
            frame.iloc[rows].reset_index(drop=True), population[rows], bundle,
            None if base_features is None else base_features[rows],
            None if ready is None else ready[rows])
    return features, result_ready


class TrendDecoder(RegressorMixin, BaseEstimator):


    def __init__(self, family='ridge', features='smooth', alpha=1000., components=16,
                 leaves=15, n_estimators=400, leaf=80, history='trends'):
        self.family, self.features, self.alpha, self.components = family, features, alpha, components
        self.leaves, self.n_estimators, self.leaf, self.history = leaves, n_estimators, leaf, history

    def transform(self, x):

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != WIDTH or not np.isfinite(x).all():
            raise ValueError('Expected 1141 finite causal history columns')
        if self.features == 'all':
            values = x.astype(float)
        elif self.features in ('smooth', 'smooth_ops'):
            values = np.column_stack([x[:, :14], x[:, 38:80], x[:, 234:381],
                                      x[:, 423:427], x[:, 427:679], x[:, 952:]]).astype(float)
        else:
            raise ValueError('Unknown trend decoder feature policy')
        if self.features == 'smooth_ops':
            values = np.column_stack([values, PolynomialFeatures(3, include_bias=False).fit_transform(x[:, :3].astype(float))])
        return values

    def fit(self, x, y):
        if self.family not in ('ridge', 'pls', 'kernel', 'lightgbm', 'extra_trees') or self.history != 'trends':
            raise ValueError('Unknown trend decoder policy')
        x, y = np.asarray(x), np.asarray(y, float)
        values = self.transform(x)
        if y.shape != (len(x), 21) or not np.isfinite(y).all():
            raise ValueError('Expected finite raw sensor labels')
        labels = np.argmax(x[:, 10:14], axis=1) * 6 + np.argmax(x[:, 4:10], axis=1)
        self.n_features_in_, self.n_outputs_ = WIDTH, 21
        self.fallback_ = make_pipeline(VarianceThreshold(1e-10), StandardScaler(), Ridge(alpha=1000.)).fit(values, y)
        self.models_ = {}
        for label in np.unique(labels):
            rows = labels == label
            selector = VarianceThreshold(1e-10).fit(values[rows])
            scaler = StandardScaler().fit(selector.transform(values[rows]))
            features = np.clip(scaler.transform(selector.transform(values[rows])), -12., 12.)
            center, scale = y[rows].mean(axis=0), np.maximum(y[rows].std(axis=0), .05)
            target = (y[rows] - center) / scale
            if self.family == 'ridge':
                model = Ridge(alpha=self.alpha).fit(features, target)
            elif self.family == 'pls':
                model = PLSRegression(n_components=min(self.components, features.shape[1], len(features)-1),
                                      scale=False, max_iter=250).fit(features, target)
            elif self.family == 'kernel':
                model = make_pipeline(Nystroem(gamma=.001, n_components=min(256, len(features)), random_state=20261018),
                                      Ridge(alpha=self.alpha)).fit(features, target)
            elif self.family == 'extra_trees':
                from sklearn.ensemble import ExtraTreesRegressor
                model = ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.leaf,
                    max_features=.6, n_jobs=2, random_state=20261018).fit(features, target)
            else:
                from lightgbm import LGBMRegressor
                model = []
                for sensor in range(21):
                    if target[:, sensor].var() < 1e-8:
                        model.append(float(target[:, sensor].mean()))
                    else:
                        model.append(LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.leaves,
                            min_child_samples=self.leaf, reg_lambda=self.alpha, learning_rate=.035,
                            n_jobs=4, random_state=20261018, verbosity=-1, deterministic=True,
                            force_col_wise=True).fit(features, target[:, sensor]))
            self.models_[int(label)] = selector, scaler, center, scale, model
        return self

    def predict(self, x):
        x = np.asarray(x)
        values = self.transform(x)
        labels = np.argmax(x[:, 10:14], axis=1) * 6 + np.argmax(x[:, 4:10], axis=1)
        result = np.zeros((len(x), 21))
        unknown = ~np.isin(labels, list(self.models_))
        if unknown.any():
            result[unknown] = self.fallback_.predict(values[unknown])
        for label, (selector, scaler, center, scale, model) in self.models_.items():
            rows = labels == label
            if not rows.any():
                continue
            features = np.clip(scaler.transform(selector.transform(values[rows])), -12., 12.)
            prediction = (np.column_stack([np.full(rows.sum(), m) if isinstance(m, float) else m.predict(features)
                                          for m in model]) if isinstance(model, list) else model.predict(features))
            result[rows] = prediction * scale + center
        if not np.isfinite(result).all():
            raise ValueError('Non-finite trend forecasts')
        return result
