"""
lightgbm_supervisor.py

Light Gradient Boosting Machine (LightGBM) Classifier for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Implements LightGBM multi-class classifier providing fast, memory-efficient GBDT training
with leaf-wise tree growth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel


class LightGBMSupervisorModel(BaseSupervisorModel):
    """LightGBM GBDT Supervisor Classifier."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="lightgbm", config=config)

    def _build_model(self) -> Any:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM package is not installed. Run 'pip install lightgbm' to use this model.")
        return lgb.LGBMClassifier(
            n_estimators=self.cfg.lightgbm_n_estimators,
            learning_rate=self.cfg.lightgbm_learning_rate,
            num_leaves=self.cfg.lightgbm_num_leaves,
            max_depth=self.cfg.lightgbm_max_depth,
            subsample=self.cfg.lightgbm_subsample,
            colsample_bytree=self.cfg.lightgbm_colsample_bytree,
            reg_alpha=self.cfg.lightgbm_reg_alpha,
            reg_lambda=self.cfg.lightgbm_reg_lambda,
            random_state=self.cfg.random_seed,
            class_weight="balanced",
            objective="multiclass",
            n_jobs=-1,
            verbose=-1,
        )
