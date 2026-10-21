"""
random_forest_supervisor.py

Balanced Random Forest Classifier for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Implements an ensemble Random Forest classifier optimized for non-linear decision
boundaries and imbalanced maintenance decision classes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

from sklearn.ensemble import RandomForestClassifier

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel


class RandomForestSupervisorModel(BaseSupervisorModel):
    """Random Forest Ensemble Supervisor Classifier."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="random_forest", config=config)

    def _build_model(self) -> Any:
        return RandomForestClassifier(
            n_estimators=self.cfg.random_forest_n_estimators,
            max_depth=self.cfg.random_forest_max_depth,
            min_samples_split=self.cfg.random_forest_min_samples_split,
            min_samples_leaf=self.cfg.random_forest_min_samples_leaf,
            random_state=self.cfg.random_seed,
            n_jobs=self.cfg.random_forest_n_jobs,
            class_weight="balanced",
        )
