from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class TensorFlowTwin(BaseTwin):


    name = "tensorflow"
    artifact_suffix = ".keras"

    def __init__(self, config: Optional[Config] = None) -> None:


        super().__init__(config)
        self.history: Dict[str, List[float]] = {}


    def _build_estimator(self, n_outputs: int) -> Any:


        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError as exc:
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


    def _fit_estimator(
        self,
        estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Any:


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


        predictions = estimator.predict(x, verbose=0)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return np.asarray(predictions, dtype=float)

    def train(self, frame: pd.DataFrame) -> Dict[str, Any]:


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


        if self.model is None:
            raise AnomalyHealthError("Keras twin used before train/load.")

        matrix, _ = self.build_design_matrix(frame, n_modes=self.n_modes)
        samples = [
            np.asarray(self.model(matrix, training=True), dtype=float)
            for _ in range(max(int(n_samples), 2))
        ]
        return np.std(np.stack(samples, axis=0), axis=0)

    def summary_text(self) -> str:


        if self.model is None:
            return "Keras twin not built yet."
        lines: List[str] = []
        self.model.summary(print_fn=lines.append)
        return "\n".join(lines)


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    twin = TensorFlowTwin(config)
    predictions, metrics = twin.run()
    logger.info("Keras twin metrics: %s", metrics)
    logger.info("Architecture:\n%s", twin.summary_text())
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
