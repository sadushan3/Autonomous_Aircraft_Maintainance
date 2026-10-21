"""
model_registry.py

Model Factory & Registry for the Autonomous Maintenance Supervisor Agent.

Purpose
-------
Provides a centralized registry for creating, loading, listing, and saving
supervisor decision model instances dynamically by name.

Supported Models:
- "logistic_regression"
- "random_forest"
- "xgboost"
- "lightgbm"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Type

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel
from app.services.Maintenance_Supervisor.models.logistic_regression_supervisor import LogisticRegressionSupervisorModel
from app.services.Maintenance_Supervisor.models.random_forest_supervisor import RandomForestSupervisorModel
from app.services.Maintenance_Supervisor.models.xgboost_supervisor import XGBoostSupervisorModel
from app.services.Maintenance_Supervisor.models.lightgbm_supervisor import LightGBMSupervisorModel
from app.services.Maintenance_Supervisor.models.ppo_supervisor import PPOSupervisorModel
from app.services.Maintenance_Supervisor.models.dqn_supervisor import DQNSupervisorModel
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()

_MODEL_MAP: Final[dict[str, Type[BaseSupervisorModel]]] = {
    "logistic_regression": LogisticRegressionSupervisorModel,
    "random_forest": RandomForestSupervisorModel,
    "xgboost": XGBoostSupervisorModel,
    "lightgbm": LightGBMSupervisorModel,
    "ppo": PPOSupervisorModel,
    "dqn": DQNSupervisorModel,
}


def get_available_models() -> list[str]:
    """Return list of supported model names."""
    return list(_MODEL_MAP.keys())


def create_model(
    model_name: str,
    config: SupervisorConfig | None = None,
) -> BaseSupervisorModel:
    """Instantiate a supervisor model by name."""
    name_clean = model_name.strip().lower()

    if name_clean not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available models: {get_available_models()}"
        )

    model_cls = _MODEL_MAP[name_clean]
    logger.info("Creating model instance for '%s'", name_clean)
    return model_cls(config=config)


def load_model(filepath: Path | str) -> BaseSupervisorModel:
    """Load a model artifact from disk and instantiate its corresponding wrapper."""
    filepath = Path(filepath).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact not found: {filepath}")

    import joblib
    import torch

    try:
        state = joblib.load(filepath)
    except Exception:
        # weights_only defaults to True since PyTorch 2.6, which refuses to
        # unpickle the numpy arrays (classes_, etc.) our own checkpoints
        # carry alongside the tensor state dict. These are artifacts this
        # codebase trained and saved itself, never loaded from an external
        # source, so disabling it here is safe.
        state = torch.load(filepath, map_location="cpu", weights_only=False)

    model_name = state.get("model_name", "random_forest")
    model_instance = create_model(model_name)
    model_instance.load(filepath)
    return model_instance

