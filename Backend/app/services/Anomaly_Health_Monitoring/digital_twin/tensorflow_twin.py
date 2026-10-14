"""
Keras MLP Digital Twin.

Architecture: ``Input -> [Dense + BatchNormalization + Dropout] x N -> Dense``
with a linear multi-output head predicting all active sensors at once.

The neural twin earns its place in the ensemble by failing differently from the
tree twins.  Forests and boosters are piecewise-constant: outside the training
envelope they clamp to the nearest leaf value.  The MLP is smooth and will
extrapolate a trend.  When a degraded engine drifts away from healthy operating
context, the trees hold their last healthy estimate while the network follows
the drift, and the *gap between those two behaviours* is what the uncertainty
layer reads as twin disagreement.

Three callbacks are used, all required by the specification:

* ``EarlyStopping`` -- restores the best weights rather than the last ones.
* ``ModelCheckpoint`` -- writes the best epoch to disk during training.
* ``ReduceLROnPlateau`` -- drops the learning rate when validation loss stalls.

Models are saved in the native ``.keras`` format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class TensorFlowTwin(BaseTwin):
    """Digital Twin backed by a multi-output Keras MLP."""

    name = "tensorflow"
    artifact_suffix = ".keras"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the twin and record its training history.

        Args:
            config: Configuration supplying the network hyper-parameters.
        """
        super().__init__(config)
        self.history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ #
    # Architecture
    # ------------------------------------------------------------------ #
    def _build_estimator(self, n_outputs: int) -> Any:
        """Build and compile the MLP.

        Args:
            n_outputs: Number of sensor channels in the output head.

        Returns:
            A compiled Keras model.

        Raises:
            AnomalyHealthError: If TensorFlow is not installed.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AnomalyHealthError(
                "TensorFlow is required for the Keras twin. Install it with "
                "'pip install tensorflow-cpu'."
            ) from exc

        params = dict(self.config.twin.tensorflow)
        seed = int(params.get("random_state", 42))
        tf.keras.utils.set_random_seed(seed)

        n_features = len(self.feature_names) or 1
        hidden_units = list(params.get("hidden_units", [256, 128, 64]))
        dropout = float(params.get("dropout", 0.2))

        inputs = keras.Input(shape=(n_features,), name="context_input")
        x = inputs
        for index, units in enumerate(hidden_units):
            x = keras.layers.Dense(
                int(units), activation="relu", name=f"dense_{index}"
            )(x)
            x = keras.layers.BatchNormalization(name=f"batchnorm_{index}")(x)
            x = keras.layers.Dropout(dropout, name=f"dropout_{index}")(x)

        outputs = keras.layers.Dense(
            int(n_outputs), activation="linear", name="expected_sensors"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="digital_twin_mlp")
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=float(params.get("learning_rate", 1e-3))
            ),
            loss="mse",
            metrics=["mae"],
        )
        self.logger.info(
            "Built Keras twin: %d input(s) -> %s -> %d output(s), dropout=%.2f",
            n_features,
            hidden_units,
            n_outputs,
            dropout,
        )
        return model

    def _callbacks(self) -> List[Any]:
        """Assemble the three required training callbacks.

        Returns:
            A list of configured Keras callbacks.
        """
        from tensorflow import keras

        params = dict(self.config.twin.tensorflow)
        checkpoint_path = self.config.paths.model_path(
            f"{self.name}_twin_best{self.artifact_suffix}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(params.get("early_stopping_patience", 8)),
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(params.get("reduce_lr_factor", 0.5)),
                patience=int(params.get("reduce_lr_patience", 4)),
                min_lr=float(params.get("min_lr", 1e-5)),
                verbose=0,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def _fit_estimator(
        self,
        estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Any:
        """Train the network with early stopping on held-out engines.

        The validation data comes from the engine-wise split rather than Keras'
        ``validation_split``, which would slice off the tail of the training
        array and mix cycles from the same engine across both sides.

        Args:
            estimator: The compiled model.
            x_train: Training design matrix.
            y_train: Training targets.
            x_val: Validation design matrix.
            y_val: Validation targets.

        Returns:
            The trained model.
        """
        params = dict(self.config.twin.tensorflow)

        if x_val is not None and y_val is not None and len(x_val):
            validation_data = (x_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
            validation_split = float(params.get("validation_split", 0.15))

        history = estimator.fit(
            x_train,
            y_train,
            batch_size=int(params.get("batch_size", 256)),
            epochs=int(params.get("epochs", 60)),
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=self._callbacks(),
            verbose=0,
        )
        self.history = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }
        epochs_run = len(self.history.get("loss", []))
        best_val = min(self.history.get("val_loss", [float("nan")]))
        self.logger.info(
            "Keras twin trained for %d epoch(s); best val_loss=%.6f", epochs_run, best_val
        )
        return estimator

    def _predict_estimator(self, estimator: Any, x: np.ndarray) -> np.ndarray:
        """Predict expected sensor values.

        Args:
            estimator: The trained model.
            x: Design matrix.

        Returns:
            An ``(n_rows, n_targets)`` prediction array.
        """
        predictions = estimator.predict(x, verbose=0)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return np.asarray(predictions, dtype=float)

    def train(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Fit the network, ensuring feature names exist before building it.

        The Keras input layer needs the feature count up front, but the base
        class builds the design matrix inside :meth:`BaseTwin.train`.  Deriving
        the column names first resolves that ordering.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            The metrics dictionary.
        """
        if not self.feature_names:
            from app.utils.Anomaly_Health_Monitoring.model_utils import (
                get_operating_mode_column,
            )

            mode_column = get_operating_mode_column()
            self.n_modes = (
                int(frame[mode_column].max()) + 1
                if mode_column in frame.columns
                else 1
            )
            _, self.feature_names = self.build_design_matrix(frame, n_modes=self.n_modes)

        metrics = super().train(frame)
        metrics["epochs_trained"] = len(self.history.get("loss", []))
        return metrics

    def monte_carlo_predict(
        self, frame: pd.DataFrame, n_samples: int = 20
    ) -> np.ndarray:
        """Estimate predictive uncertainty via Monte-Carlo dropout.

        Dropout is kept active at inference and the network is sampled
        repeatedly; the spread across samples is an epistemic uncertainty
        estimate from this single model, complementing the cross-twin spread.

        Args:
            frame: Scaled, context-labelled data.
            n_samples: Number of stochastic forward passes.

        Returns:
            An ``(n_rows, n_targets)`` array of per-sensor standard deviations.

        Raises:
            AnomalyHealthError: If the model has not been trained or loaded.
        """
        if self.model is None:
            raise AnomalyHealthError("Keras twin used before train/load.")

        matrix, _ = self.build_design_matrix(frame, n_modes=self.n_modes)
        samples = [
            np.asarray(self.model(matrix, training=True), dtype=float)
            for _ in range(max(int(n_samples), 2))
        ]
        return np.std(np.stack(samples, axis=0), axis=0)

    def summary_text(self) -> str:
        """Return the Keras architecture summary as a string.

        Returns:
            The formatted summary, or a placeholder when untrained.
        """
        if self.model is None:
            return "Keras twin not built yet."
        lines: List[str] = []
        self.model.summary(print_fn=lines.append)
        return "\n".join(lines)


def main() -> None:
    """Train the Keras twin standalone and report accuracy."""
    config = get_config()
    logger = get_logger(__name__, config)
    twin = TensorFlowTwin(config)
    predictions, metrics = twin.run()
    logger.info("Keras twin metrics: %s", metrics)
    logger.info("Architecture:\n%s", twin.summary_text())
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
