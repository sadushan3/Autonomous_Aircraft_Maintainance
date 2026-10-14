"""
Digital Twin ensemble.

Four twins are trained independently on the same healthy region and then fused
into one expected-sensor prediction per row.  Two products come out of this:

**The fused expectation.** Weighted by inverse validation RMSE, so a twin that
models healthy behaviour more accurately contributes more.  This is what the
residual is computed against.

**The disagreement spread.** The per-sensor standard deviation across the four
twin predictions.  This is not a by-product -- it is the component's primary
uncertainty signal.  Where the twins agree, the expected value is well
determined and a large residual really does mean the engine deviated.  Where
they disagree, the expectation itself is in doubt, and the confidence estimator
discounts the resulting anomaly accordingly.

A twin that fails to train (missing optional dependency, degenerate data) is
logged and excluded rather than aborting the stage; the ensemble proceeds with
whatever trained successfully, and the weights renormalise.

Output: ``twin_predictions.csv`` carrying ``unit``, ``cycle``,
``expected_sensor_*``, ``twin_spread_*`` and ``twin_spread_mean``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.lightgbm_twin import LightGBMTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import (
    RandomForestTwin,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.tensorflow_twin import (
    TensorFlowTwin,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.xgboost_twin import XGBoostTwin
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_expected_columns,
    get_identity_columns,
    get_twin_spread_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)

#: Mean per-sensor twin disagreement, used directly by the uncertainty layer.
TWIN_SPREAD_MEAN_COLUMN = "twin_spread_mean"

#: Registry of available twin variants.
TWIN_REGISTRY: Dict[str, type] = {
    "random_forest": RandomForestTwin,
    "xgboost": XGBoostTwin,
    "lightgbm": LightGBMTwin,
    "tensorflow": TensorFlowTwin,
}


class EnsembleTwin:
    """Trains, fuses and persists the family of Digital Twins."""

    def __init__(
        self,
        config: Optional[Config] = None,
        twins: Optional[List[BaseTwin]] = None,
    ) -> None:
        """Initialise the ensemble.

        Args:
            config: Configuration supplying which twins are enabled.
            twins: Optional explicit twin instances, injected for testing.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.twins: List[BaseTwin] = twins if twins is not None else self._build_twins()
        self.weights: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.target_names: List[str] = []

    def _build_twins(self) -> List[BaseTwin]:
        """Instantiate every enabled twin variant.

        Returns:
            The list of twin instances.

        Raises:
            AnomalyHealthError: If no known twin is enabled.
        """
        built: List[BaseTwin] = []
        for name in self.config.twin.enabled_twins:
            twin_class = TWIN_REGISTRY.get(str(name).lower().strip())
            if twin_class is None:
                self.logger.warning("Unknown twin variant '%s' ignored", name)
                continue
            built.append(twin_class(self.config))
        if not built:
            raise AnomalyHealthError(
                f"No valid twins enabled. Choose from {sorted(TWIN_REGISTRY)}."
            )
        self.logger.info("Ensemble twins: %s", [t.name for t in built])
        return built

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train every twin, tolerating individual failures.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            A mapping of twin name to its metrics.

        Raises:
            AnomalyHealthError: If no twin trained successfully.
        """
        self.metrics = {}
        trained: List[BaseTwin] = []

        for twin in self.twins:
            try:
                metrics = twin.train(frame)
                twin.save()
                self.metrics[twin.name] = metrics
                trained.append(twin)
                if not self.target_names:
                    self.target_names = list(twin.target_names)
            except Exception as exc:  # noqa: BLE001 - one twin must not kill the stage
                self.logger.error(
                    "Twin '%s' failed to train and is excluded from the ensemble: %s",
                    twin.name,
                    exc,
                )

        if not trained:
            raise AnomalyHealthError(
                "Every digital twin failed to train; the pipeline cannot continue."
            )

        self.twins = trained
        self.weights = self.compute_weights()
        self._persist_weights()
        return self.metrics

    def compute_weights(self) -> Dict[str, float]:
        """Derive fusion weights from validation accuracy.

        Strategies:

        ``uniform``
            Equal weight to every twin.
        ``inverse_rmse``
            Influence proportional to ``1 / RMSE``.
        ``r2``
            Influence proportional to mean per-target R-squared, clipped at
            zero so a twin that fails to beat the target mean contributes
            nothing.

        Why ``r2`` exists: pooled RMSE and mean R-squared rank these twins in
        *opposite* order, because RMSE across 21 targets is dominated by the
        high-variance sensors while R-squared weights every sensor equally. On
        the 2026-08-24 run the TensorFlow twin had the best pooled RMSE
        (0.5762) and by far the worst mean R-squared (0.0862), so
        ``inverse_rmse`` handed it the *largest* share (0.2527) of the
        ensemble. RMSE also barely separates the twins at all -- a 0.0092
        spread against an R-squared spread of 0.1448 -- so inverse-RMSE
        weighting is numerically indistinguishable from ``uniform`` (all four
        weights land in 0.2487-0.2527) while being mildly anti-correlated with
        per-target skill.

        A twin with a degenerate (zero or non-finite) score is given no weight
        rather than an infinite one; if every score is degenerate the method
        falls back to uniform.

        Returns:
            Normalised weights summing to 1, keyed by twin name.
        """
        strategy = str(self.config.twin.ensemble_weighting).lower().strip()
        names = [twin.name for twin in self.twins]

        if strategy == "uniform" or not self.metrics:
            weight = 1.0 / max(len(names), 1)
            return {name: weight for name in names}

        raw: Dict[str, float] = {}
        if strategy == "r2":
            for name in names:
                r2 = float(self.metrics.get(name, {}).get("r2", 0.0) or 0.0)
                raw[name] = max(r2, 0.0) if np.isfinite(r2) else 0.0
        else:
            for name in names:
                rmse = float(self.metrics.get(name, {}).get("rmse", 0.0) or 0.0)
                raw[name] = 1.0 / rmse if np.isfinite(rmse) and rmse > 1e-12 else 0.0

        total = sum(raw.values())
        if total <= 0:
            weight = 1.0 / max(len(names), 1)
            self.logger.warning(
                "All twin %s scores were degenerate; using uniform weights.",
                "R-squared" if strategy == "r2" else "RMSE",
            )
            return {name: weight for name in names}

        weights = {name: value / total for name, value in raw.items()}
        self.logger.info(
            "Ensemble weights (%s): %s",
            "mean per-target R-squared" if strategy == "r2" else "inverse RMSE",
            {k: round(v, 4) for k, v in weights.items()},
        )
        return weights

    def _persist_weights(self) -> Path:
        """Save the fusion weights and per-twin metrics.

        Returns:
            The path written.
        """
        payload = {
            "weighting_strategy": self.config.twin.ensemble_weighting,
            "weights": self.weights,
            "metrics": self.metrics,
            "target_names": self.target_names,
            "twins": [twin.name for twin in self.twins],
        }
        return atomic_write_json(
            payload, self.config.paths.model_path(self.config.paths.twin_weights_json)
        )

    def load(self) -> "EnsembleTwin":
        """Reload every twin and the persisted fusion weights.

        Returns:
            This instance, ready to predict.

        Raises:
            AnomalyHealthError: If no twin could be reloaded.
        """
        payload = read_json(
            self.config.paths.model_path(self.config.paths.twin_weights_json),
            default=None,
        )
        loaded: List[BaseTwin] = []
        for twin in self.twins:
            try:
                twin.load()
                loaded.append(twin)
                if not self.target_names:
                    self.target_names = list(twin.target_names)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Could not reload twin '%s': %s", twin.name, exc)

        if not loaded:
            raise AnomalyHealthError(
                "No saved digital twins could be loaded. Train the twin stage first."
            )
        self.twins = loaded

        if payload:
            self.weights = {
                name: float(value)
                for name, value in (payload.get("weights") or {}).items()
                if name in {t.name for t in loaded}
            }
            self.metrics = payload.get("metrics", {})
        if not self.weights:
            self.weights = {twin.name: 1.0 / len(loaded) for twin in loaded}

        # Renormalise in case some twins failed to reload.
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        return self

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Produce the fused expectation and the disagreement spread.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            A frame of identity keys, ``expected_sensor_*``, ``twin_spread_*``
            and ``twin_spread_mean``.

        Raises:
            AnomalyHealthError: If no twin produced a prediction.
        """
        if not self.twins:
            raise AnomalyHealthError("Ensemble has no trained twins.")

        stacked: List[np.ndarray] = []
        used_weights: List[float] = []
        used_names: List[str] = []

        for twin in self.twins:
            try:
                prediction = twin.predict(frame)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Twin '%s' failed during inference: %s", twin.name, exc)
                continue
            expected_columns = get_expected_columns(twin.target_names)
            stacked.append(prediction[expected_columns].to_numpy(dtype=float))
            used_weights.append(float(self.weights.get(twin.name, 0.0)))
            used_names.append(twin.name)
            if not self.target_names:
                self.target_names = list(twin.target_names)

        if not stacked:
            raise AnomalyHealthError("No twin produced predictions.")

        # Guard against every surviving twin having zero weight.
        weight_array = np.asarray(used_weights, dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones_like(weight_array) / len(weight_array)
        else:
            weight_array = weight_array / weight_array.sum()

        tensor = np.stack(stacked, axis=0)  # (n_twins, n_rows, n_sensors)
        fused = np.tensordot(weight_array, tensor, axes=(0, 0))
        spread = (
            np.std(tensor, axis=0)
            if tensor.shape[0] > 1
            else np.zeros_like(fused)
        )

        self.logger.info(
            "Fused %d twin(s) %s with weights %s",
            len(used_names),
            used_names,
            np.round(weight_array, 4).tolist(),
        )

        result = frame[get_identity_columns()].copy().reset_index(drop=True)
        for index, column in enumerate(get_expected_columns(self.target_names)):
            result[column] = fused[:, index]
        for index, column in enumerate(get_twin_spread_columns(self.target_names)):
            result[column] = spread[:, index]
        result[TWIN_SPREAD_MEAN_COLUMN] = spread.mean(axis=1)

        self.logger.info(
            "Twin disagreement: mean spread %.5f, max %.5f",
            float(result[TWIN_SPREAD_MEAN_COLUMN].mean()),
            float(result[TWIN_SPREAD_MEAN_COLUMN].max()),
        )
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``twin_predictions.csv`` atomically.

        Args:
            frame: Fused predictions.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.twin_predictions_csv
        )
        verify_identity_columns(frame, context="twin_predictions output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``twin_predictions.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.twin_predictions_csv
        )
        return read_csv(
            target, required_columns=get_identity_columns(), stage="ensemble_twin"
        )

    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the Digital Twin stage.

        Args:
            frame: Optional in-memory input; otherwise ``scaled_features.csv``
                is read from disk.
            retrain: Train fresh twins, or reload the saved ones.

        Returns:
            A ``(predictions, report)`` tuple.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.scaled_features_csv),
                required_columns=get_identity_columns(),
                stage="ensemble_twin",
            )

        if retrain:
            self.train(source)
        else:
            self.load()

        predictions = self.predict(source)
        output = self.save(predictions)

        ensemble_rmse = float(
            np.average(
                [self.metrics.get(t.name, {}).get("rmse", 0.0) for t in self.twins],
                weights=[max(self.weights.get(t.name, 0.0), 1e-9) for t in self.twins],
            )
        ) if self.metrics else 0.0

        report: Dict[str, Any] = {
            "rows": int(len(predictions)),
            "engines": int(predictions["unit"].nunique()),
            "twins": [t.name for t in self.twins],
            "weights": self.weights,
            "per_twin_metrics": self.metrics,
            "weighted_mean_rmse": round(ensemble_rmse, 6),
            "mean_twin_spread": float(predictions[TWIN_SPREAD_MEAN_COLUMN].mean()),
            "n_sensors_modelled": len(self.target_names),
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "digital_twin.json")
        self.logger.info(
            "Digital twin stage complete: %d rows, %d twin(s), weighted RMSE %.5f",
            report["rows"],
            len(self.twins),
            ensemble_rmse,
        )
        return predictions, report


def main() -> None:
    """Train the full twin ensemble standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    ensemble = EnsembleTwin(config)
    predictions, report = ensemble.run()
    logger.info("Ensemble report: %s", {k: v for k, v in report.items() if k != "per_twin_metrics"})
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
