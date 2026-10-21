from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ContextSensorMixture(RegressorMixin, BaseEstimator):
    def __init__(self, baseline=None, members=None, context_weights=None,
                 fallback_weights=None, grouping='subset', age_cut=None):
        self.baseline, self.members = baseline, members
        self.context_weights, self.fallback_weights = context_weights, fallback_weights
        self.grouping, self.age_cut = grouping, age_cut

    def labels(self, x):
        subset = np.argmax(x[:,10:14], axis=1)
        if self.grouping == 'subset': return subset
        if self.grouping == 'subset_mode': return subset*6 + np.argmax(x[:,4:10], axis=1)
        if self.grouping == 'subset_age' and self.age_cut is not None and 0 < self.age_cut < 1:
            return subset*2 + (x[:,3] > self.age_cut).astype(int)
        raise ValueError('Invalid context mixture policy')

    def predict(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] < 14 or not np.isfinite(x).all():
            raise ValueError('Context mixture requires finite context/history features')
        labels = self.labels(x)
        base = np.asarray(self.baseline.predict(x), float)
        fallback = np.asarray(self.fallback_weights, float)
        contexts = {int(label): np.asarray(weights, float) for label, weights in self.context_weights.items()}
        for weights in [fallback, *contexts.values()]:
            if (weights.shape != (len(self.members), base.shape[1]) or not np.isfinite(weights).all()
                    or (weights < 0).any() or (weights.sum(axis=0) > 1.+1e-8).any()):
                raise ValueError('Context sensor weights must form a finite convex mixture')
        output = base.copy()
        for member_index, member in enumerate(self.members):
            row_weights = np.stack([contexts.get(int(label), fallback)[member_index] for label in labels])
            selected = np.any(row_weights != 0., axis=1)
            if selected.any():
                output[selected] += (member.predict(x[selected]) - base[selected]) * row_weights[selected]
        if not np.isfinite(output).all(): raise ValueError('Non-finite context forecasts')
        return output
