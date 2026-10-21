"""
supervisor_model.py

Abstract Base Class & Interface for Machine Learning Models in the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Defines a unified scikit-learn compatible interface `BaseSupervisorModel` for all
classifier models implemented in the Supervisor Agent:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

It enforces consistent method signatures (`fit`, `predict`, `predict_proba`, `save`, `load`),
metadata management, probability normalization, and class-label mapping.

Execution
---------
Imported by concrete model subclasses in `app.services.Maintenance_Supervisor.models`.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from app.config.Maintenance_Supervisor.supervisor_config import (
    DECISION_CLASSES,
    DECISION_TO_SEVERITY,
    SEVERITY_TO_DECISION,
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()
_CFG = SupervisorConfig()


class BaseSupervisorModel(abc.ABC):
    """Abstract Base Class for Supervisor Decision Classifiers."""

    def __init__(self, model_name: str, config: SupervisorConfig | None = None):
        self.model_name = model_name
        self.cfg = config or _CFG
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array(DECISION_CLASSES)
        self.model: Any = None

    @abc.abstractmethod
    def _build_model(self) -> Any:
        """Instantiate the underlying ML estimator."""
        pass

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> BaseSupervisorModel:
        """Fit model on feature matrix X and target array y."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        # Convert string targets to numeric severities if necessary
        if y_arr.dtype.kind in ("U", "S", "O"):
            y_arr = np.array([DECISION_TO_SEVERITY.get(str(v).strip().lower(), 0) for v in y_arr])

        logger.info("Fitting %s model on shape X=%s...", self.model_name, X_mat.shape)

        if self.model is None:
            self.model = self._build_model()

        if sample_weight is not None:
            self.model.fit(X_mat, y_arr, sample_weight=sample_weight)
        else:
            self.model.fit(X_mat, y_arr)

        self.is_fitted = True
        logger.info("%s model fitting complete.", self.model_name)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before predict_proba.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        probs = self.model.predict_proba(X_mat)

        # Normalize to ensure rows sum strictly to 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict string decision classes for X."""
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return np.array([SEVERITY_TO_DECISION.get(idx, "continue_operation") for idx in class_indices])

    def save(self, filepath: Path | str) -> Path:
        """Serialize model artifact to joblib file."""
        if not self.is_fitted:
            raise RuntimeError(f"Cannot save unfitted model '{self.model_name}'.")

        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
            "model": self.model,
        }
        joblib.dump(state, filepath)
        logger.info("Saved %s artifact to: %s", self.model_name, filepath)
        return filepath

    def load(self, filepath: Path | str) -> BaseSupervisorModel:
        """Deserialize model artifact from joblib file."""
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found: {filepath}")

        state = joblib.load(filepath)
        self.model_name = state["model_name"]
        self.feature_names = state["feature_names"]
        self.classes_ = state["classes_"]
        self.is_fitted = state["is_fitted"]
        self.model = state["model"]

        logger.info("Loaded %s artifact from: %s", self.model_name, filepath)
        return self
