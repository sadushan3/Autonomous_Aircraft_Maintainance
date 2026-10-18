from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class ShapExplainer:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.kind_used: str = "none"
        self.base_values: Optional[np.ndarray] = None


    def explain_tree_model(
        self,
        model: Any,
        x: np.ndarray,
        feature_names: List[str],
        output_index: int = 0,
    ) -> np.ndarray:


        try:
            import shap
        except ImportError as exc:
            raise AnomalyHealthError(
                "shap is required for explainability. Install it with 'pip install shap'."
            ) from exc

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x)
        if isinstance(values, list):
            values = values[output_index]
        elif np.asarray(values).ndim == 3:
            values = values[:, :, output_index]
        expected = np.asarray(explainer.expected_value).reshape(-1)
        self.base_values = np.full(len(x), float(expected[min(output_index, len(expected) - 1)]))
        self.kind_used = "tree"
        return np.asarray(values, dtype=float)


    def explain_generic_model(
        self,
        predict_fn: Any,
        x: np.ndarray,
        background: np.ndarray,
        output_index: int = 0,
    ) -> np.ndarray:


        try:
            import shap
        except ImportError as exc:
            raise AnomalyHealthError(
                "shap is required for explainability. Install it with 'pip install shap'."
            ) from exc

        if not len(background) or not len(x):
            raise AnomalyHealthError("SHAP requires non-empty observations and background data.")


        summarized_background = shap.utils.sample(
            np.asarray(background, dtype=float),
            min(int(self.config.explainability.background_sample_size), len(background)),
            random_state=self.config.data.random_state,
        )

        def _single_output(data: np.ndarray) -> np.ndarray:
            predictions = np.asarray(predict_fn(data), dtype=float)
            if predictions.ndim == 1:
                return predictions
            return predictions[:, output_index]

        masker = shap.maskers.Independent(summarized_background)
        explainer = shap.PermutationExplainer(_single_output, masker)
        explanation = explainer(
            x, max_evals=max(int(self.config.explainability.permutation_max_evals), 2 * x.shape[1] + 1)
        )
        self.kind_used = "permutation"
        self.base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
        return np.asarray(explanation.values, dtype=float)


    def rank_features(
        self, shap_values: np.ndarray, feature_names: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:


        if shap_values.size == 0:
            return []
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        mean_signed = np.mean(shap_values, axis=0)
        order = np.argsort(mean_abs)[::-1]
        limit = int(top_k) if top_k else len(order)

        return [
            {
                "feature": feature_names[i],
                "shap_value": float(mean_signed[i]),
                "abs_shap_value": float(mean_abs[i]),
                "rank": rank,
            }
            for rank, i in enumerate(order[:limit])
        ]

    def explain_row(
        self, shap_values: np.ndarray, feature_names: List[str], row_index: int, top_k: int = 8
    ) -> List[Dict[str, Any]]:


        if row_index >= shap_values.shape[0]:
            return []
        row_values = shap_values[row_index]
        order = np.argsort(np.abs(row_values))[::-1][:top_k]
        return [
            {
                "feature": feature_names[i],
                "shap_value": float(row_values[i]),
                "abs_shap_value": float(abs(row_values[i])),
                "rank": rank,
            }
            for rank, i in enumerate(order)
        ]


    def explain_twin(
        self,
        twin: Any,
        frame: pd.DataFrame,
        sensor: str,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:


        if sensor not in twin.target_names:
            raise AnomalyHealthError(f"Twin does not model sensor '{sensor}'.")

        limit = int(max_rows or self.config.explainability.max_explain_rows)
        from app.services.Anomaly_Health_Monitoring.explainability.monitoring_reference_explainer import representative_sample
        sample = representative_sample(frame, limit, self.config.data.random_state)
        matrix, feature_names = twin.build_design_matrix(sample, n_modes=twin.n_modes)
        output_index = twin.target_names.index(sensor)

        kind = str(self.config.explainability.explainer_kind).lower()
        shap_values: Optional[np.ndarray] = None

        if kind in ("auto", "tree") and hasattr(twin, "per_sensor_models"):
            models = twin.per_sensor_models()
            model = models.get(sensor)
            if model is not None:
                try:
                    shap_values = self.explain_tree_model(model, matrix, feature_names)
                except Exception as exc:
                    self.logger.warning("Tree explainer failed for %s: %s", sensor, exc)

        if shap_values is None and kind in ("auto", "tree") and type(twin).__name__ == "RandomForestTwin":
            try:
                shap_values = self.explain_tree_model(twin.model, matrix, feature_names, output_index)
            except Exception as exc:
                self.logger.warning("Tree explainer failed on random forest: %s", exc)

        if shap_values is None:
            reference_rows = frame[frame.cycle.between(1, max(self.config.twin.commissioning_cycles, 1))]
            background_frame = representative_sample(reference_rows, self.config.explainability.background_sample_size,
                                                     self.config.data.random_state)
            background, _ = twin.build_design_matrix(background_frame, n_modes=twin.n_modes)

            def _predict(data: np.ndarray) -> np.ndarray:
                return twin._predict_estimator(twin.model, data)

            shap_values = self.explain_generic_model(_predict, matrix, background, output_index)

        ranking = self.rank_features(
            shap_values, feature_names, top_k=self.config.explainability.top_k_features
        )
        predicted = np.asarray(twin._predict_estimator(twin.model, matrix), dtype=float)
        predicted = predicted if predicted.ndim == 1 else predicted[:, output_index]
        errors = np.abs(self.base_values + shap_values.sum(axis=1) - predicted)
        return {
            "sensor": sensor,
            "kind_used": self.kind_used,
            "feature_names": feature_names,
            "n_rows_explained": int(matrix.shape[0]),
            "global_ranking": ranking,
            "shap_values": shap_values.tolist(),
            "base_values": self.base_values.tolist() if self.base_values is not None else [],
            "explained_output": "raw_twin_prediction",
            "prediction": predicted.tolist(), "max_additivity_error": float(errors.max()),
            "identities": sample[["unit", "cycle"]].to_dict("records"),
        }


def main() -> None:

    from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import (
        RandomForestTwin,
    )
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.scaled_features_csv),
        required_columns=get_identity_columns(),
        stage="shap_explainer",
    )
    twin = RandomForestTwin(config).load()
    explainer = ShapExplainer(config)
    sensor = frame[twin.target_names].var().idxmax()
    result = explainer.explain_twin(twin, frame, sensor=sensor, max_rows=200)
    logger.info("Explainer used: %s", result["kind_used"])
    logger.info("Global ranking for %s: %s", result["sensor"], result["global_ranking"][:5])


if __name__ == "__main__":
    main()
