"""
xgboost_supervisor.py

Gradient Boosted Decision Tree (XGBoost) Classifier for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Implements XGBoost multi-class classifier optimized for high-dimensional feature spaces
and fine-grained probability calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import xgboost as xgb

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel


class XGBoostSupervisorModel(BaseSupervisorModel):
    """XGBoost Gradient Boosting Supervisor Classifier."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="xgboost", config=config)

    def _build_model(self) -> Any:
        return xgb.XGBClassifier(
            n_estimators=self.cfg.xgboost_n_estimators,
            max_depth=self.cfg.xgboost_max_depth,
            learning_rate=self.cfg.xgboost_learning_rate,
            subsample=self.cfg.xgboost_subsample,
            colsample_bytree=self.cfg.xgboost_colsample_bytree,
            min_child_weight=self.cfg.xgboost_min_child_weight,
            reg_alpha=self.cfg.xgboost_reg_alpha,
            reg_lambda=self.cfg.xgboost_reg_lambda,
            random_state=self.cfg.random_seed,
            eval_metric="mlogloss",
            objective="multi:softprob",
            n_jobs=-1,
        )
