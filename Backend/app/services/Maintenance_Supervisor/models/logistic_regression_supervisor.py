"""
logistic_regression_supervisor.py

Multinomial Logistic Regression Classifier for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Implements L2-regularized multinomial logistic regression for supervisor decision
classification. Provides a linear baseline model with probability calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

from sklearn.linear_model import LogisticRegression

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel


class LogisticRegressionSupervisorModel(BaseSupervisorModel):
    """Multinomial Logistic Regression Supervisor Classifier."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="logistic_regression", config=config)

    def _build_model(self) -> Any:
        return LogisticRegression(
            solver=self.cfg.logistic_solver,
            max_iter=self.cfg.logistic_max_iter,
            random_state=self.cfg.random_seed,
            class_weight="balanced",
            multi_class="multinomial",
        )
