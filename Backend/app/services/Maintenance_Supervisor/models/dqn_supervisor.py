"""
dqn_supervisor.py

Deep Q-Network (DQN) Deep Reinforcement Learning Agent for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Provides `DQNSupervisorModel`, a Q-value-shaped classifier over the five discrete
maintenance actions, inheriting from `BaseSupervisorModel`.

What this is NOT
----------------
Despite the name, this is **not** reinforcement learning, and its outputs must
not be described as a learned policy:

* There is no environment. `TurbofanMaintenanceEnv` (``rl_agent/rl_environment.py``)
  is never imported here (it is used by ``PPOSupervisorModel``, which trains via
  genuine environment rollouts -- see ``ppo_supervisor.py``).
* There are no transitions. ``fit`` receives a static ``(X, y)`` matrix of
  states and *labelled* actions; ``max_next_q`` is evaluated on the **same**
  state, not a successor state, so the Bellman backup has no next-state term.
* There is no reward signal. The "reward" is ``r = float(action) * 10.0`` --
  a pure function of the ground-truth label, not of any outcome.
* There is no replay buffer, no exploration policy, and no episode structure.

In effect this fits Q(s, a_label) toward ``10 * a_label + gamma * max_a Q(s, a)``
by full-batch gradient descent for a fixed 15 epochs, and softmaxes the result
into class probabilities. It is a supervised classifier with an unusual loss.
Treat its metrics as classification metrics, and see
``artifacts/Maintenance_Supervisor/champion_model_manifest.json`` for why those
metrics are near-1.0 on this target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    DECISION_CLASSES,
    DECISION_TO_SEVERITY,
    SEVERITY_TO_DECISION,
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.models.supervisor_model import BaseSupervisorModel
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()
_CFG = SupervisorConfig()


class QNetwork(nn.Module):
    """Deep Q-Network architecture mapping state vector to Q-values per action."""

    def __init__(self, state_dim: int, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNSupervisorModel(BaseSupervisorModel):
    """DQN Deep Reinforcement Learning Supervisor Model."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="dqn", config=config)
        self.q_net: QNetwork | None = None
        self.target_net: QNetwork | None = None
        self.optimizer: optim.Adam | None = None
        self.device = torch.device("cpu")

    def _build_model(self) -> Any:
        state_dim = len(self.feature_names) if self.feature_names else 30
        self.q_net = QNetwork(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.target_net = QNetwork(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.rl_learning_rate)
        return self.q_net

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> BaseSupervisorModel:
        """Fit the Q-network on labelled (state, action) pairs.

        Not a Bellman optimisation: the bootstrap term is taken from the same
        state rather than a successor, and the reward is derived from the label.
        See the module docstring.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            X_mat = np.asarray(X)

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = np.asarray(y)

        if y_arr.dtype.kind in ("U", "S", "O"):
            y_arr = np.array([DECISION_TO_SEVERITY.get(str(v).strip().lower(), 0) for v in y_arr])

        X_mat = np.nan_to_num(X_mat.astype(np.float32))
        y_arr = y_arr.astype(np.int64)

        if self.q_net is None:
            self._build_model()

        states_t = torch.tensor(X_mat, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(y_arr, dtype=torch.long, device=self.device)

        logger.info("Fitting DQN RL Model over %d samples...", len(X_mat))

        self.q_net.train()
        criterion = nn.SmoothL1Loss()

        for epoch in range(15):
            self.optimizer.zero_grad()
            q_values = self.q_net(states_t)

            # Target Q-values constructed from target network + action rewards
            with torch.no_grad():
                max_next_q = self.target_net(states_t).max(1)[0]

            targets = q_values.clone()
            for i in range(len(actions_t)):
                act = actions_t[i]
                r = float(act) * 10.0
                targets[i, act] = r + self.cfg.rl_gamma * max_next_q[i]

            loss = criterion(q_values, targets)
            loss.backward()
            self.optimizer.step()

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.is_fitted = True
        logger.info("DQN model fitting completed successfully.")
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict softmax probabilities over Q-values for X."""
        if not self.is_fitted or self.q_net is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before predict_proba.")

        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                X_df = X.reindex(columns=self.feature_names, fill_value=0.0)
                X_mat = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            else:
                X_mat = X.select_dtypes(include=[np.number]).values
        else:
            X_mat = np.asarray(X)

        X_mat = np.nan_to_num(X_mat.astype(np.float32))
        states_t = torch.tensor(X_mat, dtype=torch.float32, device=self.device)

        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(states_t)
            probs = torch.softmax(q_values, dim=1).cpu().numpy()

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def save(self, filepath: Path | str) -> Path:
        """Serialize DQN weights and metadata."""
        if not self.is_fitted or self.q_net is None:
            raise RuntimeError(f"Cannot save unfitted model '{self.model_name}'.")

        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
            "q_net_state_dict": self.q_net.state_dict(),
        }
        torch.save(state, filepath)
        logger.info("Saved %s RL model artifact to: %s", self.model_name, filepath)
        return filepath

    def load(self, filepath: Path | str) -> BaseSupervisorModel:
        """Deserialize DQN weights and metadata."""
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found: {filepath}")

        # weights_only=False: our own checkpoint, never loaded externally
        # (see ppo_supervisor.py's load() for why the PyTorch 2.6 default breaks this).
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model_name = state["model_name"]
        self.feature_names = state["feature_names"]
        self.classes_ = state["classes_"]
        self.is_fitted = state["is_fitted"]

        state_dim = len(self.feature_names) if self.feature_names else 30
        self.q_net = QNetwork(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.target_net = QNetwork(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.q_net.load_state_dict(state["q_net_state_dict"])
        self.target_net.load_state_dict(state["q_net_state_dict"])
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.rl_learning_rate)

        logger.info("Loaded %s RL model artifact from: %s", self.model_name, filepath)
        return self
