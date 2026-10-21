from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError


def representative_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:

    if frame.empty or limit <= 0:
        return frame.iloc[:0].copy()
    groups = [c for c in ("subset_id", "operating_mode", "unit") if c in frame]
    shuffled = frame.sample(frac=1, random_state=seed).copy()
    shuffled["_sample_round"] = shuffled.groupby(groups, dropna=False).cumcount()
    return shuffled.sort_values("_sample_round", kind="stable").head(limit).drop(columns="_sample_round")


class MonitoringReferenceExplainer:


    feature_names = ["operating_context", "cycle_position"]

    def __init__(self, ensemble: Any, config: Any) -> None:
        self.ensemble = ensemble
        self.config = config

    def _raw_prediction(self, matrix: np.ndarray) -> np.ndarray:
        predictions = []
        for twin in self.ensemble.twins:
            if twin.target_names != self.ensemble.target_names:
                raise AnomalyHealthError("Twin target ordering differs within the monitoring ensemble.")
            batches = [np.asarray(twin._predict_estimator(twin.model, matrix[i:i + 4096]), dtype=float)
                       for i in range(0, len(matrix), 4096)]
            predictions.append(np.concatenate(batches))
        weights = np.array([self.ensemble.weights.get(t.name, 0.) for t in self.ensemble.twins])
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1. / len(weights))
        return np.tensordot(weights, np.stack(predictions), axes=(0, 0))

    def explain(self, scaled: pd.DataFrame, events: pd.DataFrame,
                predictions: pd.DataFrame) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
        import shap

        keys = ["unit", "cycle"]
        for label, data in (("scaled features", scaled), ("events", events), ("predictions", predictions)):
            if data.duplicated(keys).any():
                raise AnomalyHealthError(f"Duplicate engine/cycle identities in {label}.")
        if scaled.empty or events.empty or predictions.empty:
            raise AnomalyHealthError("Local SHAP requires matching context, events and monitoring predictions.")
        template = self.ensemble.twins[0]
        matrix, names = template.build_design_matrix(scaled, n_modes=template.n_modes)
        for twin in self.ensemble.twins[1:]:
            other, other_names = twin.build_design_matrix(scaled, n_modes=twin.n_modes)
            if other_names != names or not np.array_equal(other, matrix):
                raise AnomalyHealthError("Twins use different context layouts; grouped explanation is unavailable.")
        raw = self._raw_prediction(matrix)
        weights = np.array([self.ensemble.weights.get(t.name, 0.) for t in self.ensemble.twins])
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1. / len(weights))


        offsets = []
        for twin in self.ensemble.twins:
            p = np.asarray(twin._predict_estimator(twin.model, matrix), dtype=float)
            offsets.append(twin.calibrate_predictions(scaled, p) - p)
        offset = np.tensordot(weights, np.stack(offsets), axes=(0, 0))
        position = pd.Series(np.arange(len(scaled)), index=pd.MultiIndex.from_frame(scaled[keys]))
        event_keys = pd.MultiIndex.from_frame(events[keys])
        if not event_keys.isin(position.index).all():
            raise AnomalyHealthError("Some alert identities are absent from the supplied context history.")
        rows = position.reindex(event_keys).to_numpy(int)
        targets = [self.ensemble.target_names.index(str(s)) for s in events["shap_sensor"]]
        saved = predictions.set_index(keys).reindex(event_keys)
        expected = []
        for i, sensor in enumerate(events.shap_sensor):
            column = f"reference_expected_{sensor}"
            if column not in saved:
                column = f"expected_{sensor}"
            if column not in saved or pd.isna(saved.iloc[i][column]):
                raise AnomalyHealthError(f"Missing monitoring expectation for {sensor}.")
            expected.append(float(saved.iloc[i][column]))
        expected = np.asarray(expected)
        actual_prediction = raw[rows, targets] + offset[rows, targets]
        fidelity = np.abs(actual_prediction - expected)
        tolerance = float(self.config.explainability.fidelity_tolerance)
        if np.any(fidelity > tolerance):
            raise AnomalyHealthError(
                f"Saved monitoring predictions do not match active ensemble/history (max error {fidelity.max():.6g}). "
                "Refresh inference outputs before attaching SHAP."
            )


        healthy = scaled[scaled.cycle.between(1, max(int(self.config.twin.commissioning_cycles), 1))]
        background_frame = representative_sample(healthy, self.config.explainability.reference_background_rows,
                                                   self.config.data.random_state)
        if background_frame.empty:
            raise AnomalyHealthError("No commissioning-prefix background is available; SHAP was withheld.")
        bg_rows = position.reindex(pd.MultiIndex.from_frame(background_frame[keys])).to_numpy(int)
        background = matrix[bg_rows]
        age_columns = [names.index("cycle"), names.index("cycle_norm")]
        context_columns = [i for i in range(len(names)) if i not in age_columns]
        coalition = np.zeros((4, len(events)))
        coalition[0] = self._raw_prediction(background).mean(axis=0)[targets]
        coalition[3] = raw[rows, targets]


        for mask, columns in ((1, context_columns), (2, age_columns)):
            mixed = np.tile(background, (len(events), 1))
            for i, row in enumerate(rows):
                mixed[i * len(background):(i + 1) * len(background), columns] = matrix[row, columns]
            result = self._raw_prediction(mixed).reshape(len(events), len(background), -1).mean(axis=1)
            coalition[mask] = result[np.arange(len(events)), targets]


        def coalition_value(flags: np.ndarray) -> np.ndarray:
            indices = flags[:, 0].astype(int) + 2 * flags[:, 1].astype(int)
            return coalition[indices]

        explanation = shap.Explainer(coalition_value, np.zeros((1, 2)), algorithm="exact",
                                     feature_names=self.feature_names)(np.ones((1, 2)))
        values = np.asarray(explanation.values).reshape(1, 2, len(events))[0].T
        bases = np.asarray(explanation.base_values).reshape(-1) + offset[rows, targets]
        errors = np.abs(bases + values.sum(axis=1) - actual_prediction)
        if not np.isfinite(values).all() or np.any(errors > tolerance):
            raise AnomalyHealthError("Grouped SHAP failed prediction reconstruction.")
        records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for i, event in enumerate(events.itertuples()):
            ranking = [{"feature": self.feature_names[j], "shap_value": float(values[i, j]),
                        "abs_shap_value": float(abs(values[i, j])), "rank": rank}
                       for rank, j in enumerate(np.argsort(-np.abs(values[i])))]
            records[(int(event.unit), int(event.cycle))] = {
                "shap_status": "computed", "shap_sensor": event.shap_sensor,
                "shap_explained_output": "monitoring_reference_expected_sensor",
                "shap_top_features": ranking, "shap_base_value": float(bases[i]),
                "shap_prediction": float(actual_prediction[i]),
                "shap_commissioning_offset": float(offset[rows[i], targets[i]]),
                "shap_additivity_error": float(errors[i]), "shap_reference_error": float(fidelity[i]),
                "shap_background_rows": len(background),
            }


        global_by_sensor = {}
        for sensor in events.shap_sensor.unique():
            selected = (events.shap_sensor == sensor).to_numpy()
            global_by_sensor[sensor] = {
                "rows": int(selected.sum()),
                "features": [{"feature": name, "mean_abs_shap": float(np.abs(values[selected, j]).mean())}
                             for j, name in enumerate(self.feature_names)],
            }
        return records, {
            "explainer_used": "exact_grouped", "shap_version": shap.__version__,
            "background_policy": "representative commissioning-prefix rows",
            "background_rows": len(background), "global_by_sensor": global_by_sensor,
            "model_weights": dict(zip([t.name for t in self.ensemble.twins], weights.tolist())),
            "max_additivity_error": float(errors.max()), "max_reference_error": float(fidelity.max()),
            "scope": "Expected sensor value; commissioning history held fixed. Not causal fault attribution.",
            "feature_groups": {"operating_context": [names[i] for i in context_columns],
                               "cycle_position": [names[i] for i in age_columns]},
        }
