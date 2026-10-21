from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.sequence_generator import (
    SequenceGenerator,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_identity_columns,
    present_residual_z_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_json,
    empirical_percentile_normalize,
    get_logger,
    load_model,
    percentile_threshold,
    read_json,
)


SCORE_COLUMN = "lstm_ae_score"


class LSTMAutoencoderDetector:


    ARTIFACT_NAME = "lstm_autoencoder.keras"
    META_NAME = "lstm_autoencoder_meta.json"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.sequence_generator = SequenceGenerator(self.config)
        self.model: Any = None
        self.feature_columns: List[str] = []
        self.sequence_length: int = int(self.config.features.sequence_length)
        self.threshold: float = 1.0
        self._healthy_errors_sorted: np.ndarray = np.array([], dtype=float)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:


        settings = self.config.healthy_region
        if RUL_COLUMN in frame.columns:
            rul_ok = frame[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=frame.index)
        max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = frame["cycle"].astype(float) <= (float(settings.cycle_fraction) * max_cycle)
        return rul_ok & cycle_ok


    def _build_model(self, n_features: int) -> Any:


        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError as exc:
            raise AnomalyHealthError(
                "TensorFlow is required for the LSTM auto-encoder. Install it "
                "with 'pip install tensorflow-cpu'."
            ) from exc

        params = dict(self.config.anomaly.lstm_autoencoder)
        tf.keras.utils.set_random_seed(int(params.get("random_state", 42)))

        encoder_units = int(params.get("encoder_units", 64))
        latent_units = int(params.get("latent_units", 32))
        dropout = float(params.get("dropout", 0.1))

        inputs = keras.Input(shape=(self.sequence_length, n_features), name="residual_window")
        encoded = keras.layers.LSTM(
            encoder_units, activation="tanh", return_sequences=True, name="encoder_lstm_1"
        )(inputs)
        encoded = keras.layers.Dropout(dropout)(encoded)
        encoded = keras.layers.LSTM(
            latent_units, activation="tanh", return_sequences=False, name="encoder_latent"
        )(encoded)

        repeated = keras.layers.RepeatVector(self.sequence_length, name="repeat")(encoded)
        decoded = keras.layers.LSTM(
            latent_units, activation="tanh", return_sequences=True, name="decoder_lstm_1"
        )(repeated)
        decoded = keras.layers.Dropout(dropout)(decoded)
        decoded = keras.layers.LSTM(
            encoder_units, activation="tanh", return_sequences=True, name="decoder_lstm_2"
        )(decoded)
        outputs = keras.layers.TimeDistributed(
            keras.layers.Dense(n_features, activation="linear"), name="reconstruction"
        )(decoded)

        model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(params.get("learning_rate", 1e-3))),
            loss="mse",
        )
        self.logger.info(
            "Built LSTM auto-encoder: window=%d, features=%d, encoder_units=%d, latent=%d",
            self.sequence_length,
            n_features,
            encoder_units,
            latent_units,
        )
        return model


    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "LSTMAutoencoderDetector":


        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for the LSTM auto-encoder.")
        self.feature_columns = present

        healthy = frame.loc[self._healthy_mask(frame)]
        if healthy.empty:
            healthy = frame

        training_windows = self.sequence_generator.build_training_windows(
            healthy, present, sequence_length=self.sequence_length
        )

        self.model = self._build_model(n_features=len(present))
        params = dict(self.config.anomaly.lstm_autoencoder)

        from tensorflow import keras

        callbacks = [
            keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.logger.info(
                "LSTM auto-encoder epoch %d: loss=%.6f, val_loss=%.6f",
                epoch + 1, logs.get("loss", float("nan")), logs.get("val_loss", float("nan")))),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(params.get("early_stopping_patience", 6)),
                restore_best_weights=True,
                verbose=0,
            )
        ]
        self.model.fit(
            training_windows,
            training_windows,
            batch_size=int(params.get("batch_size", 128)),
            epochs=int(params.get("epochs", 40)),
            validation_split=float(params.get("validation_split", 0.15)),
            callbacks=callbacks,
            verbose=0,
        )

        healthy_windows, _ = self.sequence_generator.build(
            healthy, present, sequence_length=self.sequence_length
        )
        healthy_errors = self._reconstruction_error(healthy_windows)
        self._healthy_errors_sorted = np.sort(healthy_errors)
        normalized_healthy = empirical_percentile_normalize(
            self._healthy_errors_sorted, healthy_errors
        )
        self.threshold = percentile_threshold(
            normalized_healthy, self.config.anomaly.lstm_ae_threshold_percentile
        )

        self.logger.info(
            "LSTM auto-encoder fitted on %d training window(s); threshold=%.4f (p%.0f)",
            training_windows.shape[0],
            self.threshold,
            self.config.anomaly.lstm_ae_threshold_percentile,
        )
        return self


    def _reconstruction_error(self, windows: np.ndarray) -> np.ndarray:


        if windows.shape[0] == 0:
            return np.zeros(0, dtype=float)
        reconstructed = self.model.predict(windows, verbose=0)
        return np.mean((windows - reconstructed) ** 2, axis=(1, 2))

    def score(self, frame: pd.DataFrame) -> np.ndarray:


        if self.model is None:
            raise AnomalyHealthError("LSTMAutoencoderDetector used before fit/load.")

        windows, keys = self.sequence_generator.build(
            frame, self.feature_columns, sequence_length=self.sequence_length
        )
        errors = self._reconstruction_error(windows)
        normalized = empirical_percentile_normalize(self._healthy_errors_sorted, errors)


        score_frame = keys.copy()
        score_frame[SCORE_COLUMN] = normalized
        ordered = frame[get_identity_columns()].merge(
            score_frame, on=get_identity_columns(), how="left"
        )
        return ordered[SCORE_COLUMN].fillna(0.0).to_numpy()

    def predict_flags(self, frame: pd.DataFrame) -> np.ndarray:


        return self.score(frame) > self.threshold


    def save(self) -> Path:


        if self.model is None:
            raise AnomalyHealthError("Nothing to save; fit the detector first.")
        from app.utils.Anomaly_Health_Monitoring.utils import save_model

        path = save_model(
            self.model, self.config.paths.model_path(self.ARTIFACT_NAME), config=self.config
        )
        atomic_write_json(
            {
                "feature_columns": self.feature_columns,
                "sequence_length": self.sequence_length,
                "threshold": self.threshold,
                "healthy_errors_sorted": self._healthy_errors_sorted.tolist(),
            },
            self.config.paths.model_path(self.META_NAME),
        )
        return path

    def load(self) -> "LSTMAutoencoderDetector":


        meta = read_json(self.config.paths.model_path(self.META_NAME), default=None)
        if not meta:
            raise AnomalyHealthError("No LSTM auto-encoder metadata found; train it first.")
        self.model = load_model(self.config.paths.model_path(self.ARTIFACT_NAME))
        self.feature_columns = list(meta["feature_columns"])
        self.sequence_length = int(meta["sequence_length"])
        self.threshold = float(meta["threshold"])
        self._healthy_errors_sorted = np.asarray(meta["healthy_errors_sorted"], dtype=float)
        return self


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="lstm_autoencoder",
    )
    feature_columns = present_residual_z_columns(frame)
    detector = LSTMAutoencoderDetector(config).fit(frame, feature_columns)
    detector.save()
    scores = detector.score(frame)
    logger.info(
        "Mean score=%.4f, flagged=%d/%d", float(scores.mean()),
        int((scores > detector.threshold).sum()), len(frame),
    )


if __name__ == "__main__":
    main()
