"""
ppo_supervisor.py

Proximal Policy Optimization (PPO) Deep Reinforcement Learning Agent for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Provides `PPOSupervisorModel`, an actor-critic policy over the five discrete
maintenance actions, trained by genuine on-policy PPO against
``rl_agent.rl_environment.TurbofanMaintenanceEnv``: rollout collection under
the current stochastic policy, Generalized Advantage Estimation (GAE), a
clipped surrogate policy objective, a value-function loss, and an entropy
bonus -- not supervised fitting on labelled actions.

``fit`` accepts the full trajectory dataframe (engine_id, cycle, and the raw
RUL/risk/anomaly signals the environment steps over) and trains via
environment rollouts. It also accepts a plain ``(X, y)`` feature/label pair
for the one remaining caller that does not have trajectory structure
(`federated/federated_supervisor.py`, itself out of project scope per the
federated-learning scope decision) -- in that case it falls back to
behaviour cloning and the returned model must not be described as a trained
policy. The dispatch is automatic: `fit` inspects whether `X` carries the
engine/cycle columns the environment needs.

Feature normalization is owned by the model, not the caller
--------------------------------------------------------------
The live serving path (``supervisor_bridge.run_supervisor_bridge`` ->
``real_output_merger``) feeds the champion RAW physical-unit values
straight from the upstream exchange CSVs -- it applies no scaler.
``TurbofanMaintenanceEnv`` likewise deals in raw units (see its "Fixed
reward-scale bug" docstring note). A network trained on arbitrarily-scaled
input and then fed raw values (or vice versa) sees inputs far outside its
training distribution and saturates to one constant output regardless of
the actual input -- this was confirmed directly: even after fixing the
reward-scale bug and restoring real episode-return variance,
``raw_model_decision`` stayed constant across every row of a real
33,991-row live run, including its 20 highest-risk rows.

``_fit_via_environment`` computes a fixed per-feature mean/std from the raw
training data once, stores it as ``feature_mean_``/``feature_std_`` (saved
and loaded with the model), and every state->tensor conversion in this
class routes through ``_normalize()`` before reaching the network -- during
training rollouts, during ``evaluate_policy``, and during live
``predict_proba``. Callers (the environment, ``train_ppo.py``,
``decision_fusion_engine.py``) always deal in raw physical units; only this
class's internal tensors are normalized.
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
from torch.distributions import Categorical

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
from app.services.Maintenance_Supervisor.rl_agent.rl_environment import TurbofanMaintenanceEnv
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()
_CFG = SupervisorConfig()

# The environment's reward costs are in currency units (cost_unplanned_failure
# = 5000, accumulated over up to ~200 steps per episode), which puts raw
# returns in the thousands. Feeding that directly into the value-function
# target makes value_loss dominate policy_loss by several orders of
# magnitude once both go through the same gradient-norm clip, effectively
# drowning out the policy learning signal. Scaling rewards down before GAE
# and the value loss keeps training numerically well-behaved without
# changing the (interpretable, currency-unit) returns reported in logs and
# evaluate_policy, which sum the raw, unscaled environment reward.
_REWARD_SCALE: Final[float] = 1000.0


class ActorCriticNet(nn.Module):
    """Shared-trunk actor-critic network over the 5 maintenance actions."""

    def __init__(self, state_dim: int, action_dim: int = 5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value


class PPOSupervisorModel(BaseSupervisorModel):
    """PPO deep reinforcement-learning supervisor policy."""

    def __init__(self, config: SupervisorConfig | None = None):
        super().__init__(model_name="ppo", config=config)
        self.net: ActorCriticNet | None = None
        self.optimizer: optim.Adam | None = None
        self.device = torch.device("cpu")
        self.training_history: list[dict[str, float]] = []
        self.trained_via_environment: bool = False
        # Fixed per-feature mean/std, computed once from the raw training
        # data at fit time and saved with the model -- see the module
        # docstring on why the network needs normalized input while the
        # live serving path and TurbofanMaintenanceEnv both deal in raw
        # physical units. None until _fit_via_environment sets them; the
        # behaviour-cloning fallback path leaves them unset (that caller's
        # X is not necessarily in the same raw units).
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        # Set by _fit_via_environment when val_df is supplied -- which
        # epoch's checkpoint was actually kept, and its held-out metrics.
        self.best_checkpoint_epoch: int | None = None
        self.best_checkpoint_val_metrics: dict[str, float] | None = None

    def _build_model(self) -> Any:
        state_dim = len(self.feature_names) if self.feature_names else 30
        self.net = ActorCriticNet(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.rl_learning_rate)
        return self.net

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Apply the fixed training-time mean/std normalization, if set.

        Works on a single state vector (1D) or a batch (2D, rows = samples);
        broadcasting handles both. A no-op when feature_mean_/feature_std_
        were never computed (behaviour-cloning fallback path).
        """
        if self.feature_mean_ is None or self.feature_std_ is None:
            return state
        return (state - self.feature_mean_) / self.feature_std_

    # ------------------------------------------------------------------
    # Public entry point -- dispatches to environment-based PPO training
    # when trajectory structure is available, else behaviour cloning.
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        val_df: pd.DataFrame | None = None,
    ) -> BaseSupervisorModel:
        has_trajectory_columns = (
            isinstance(X, pd.DataFrame)
            and self.cfg.engine_id_column in X.columns
            and self.cfg.cycle_column in X.columns
        )
        if has_trajectory_columns:
            return self._fit_via_environment(X, val_df=val_df)

        logger.warning(
            "PPOSupervisorModel.fit called without engine/cycle trajectory "
            "columns; falling back to behaviour cloning on labelled actions. "
            "This path exists only for callers without trajectory structure "
            "(e.g. federated_supervisor.py) -- its output is NOT a trained "
            "RL policy and must not be reported as one."
        )
        return self._fit_behaviour_cloning(X, y, sample_weight)

    # ------------------------------------------------------------------
    # Genuine PPO: rollout collection + GAE + clipped surrogate objective
    # ------------------------------------------------------------------

    def _fit_via_environment(
        self, df_trajectories: pd.DataFrame, val_df: pd.DataFrame | None = None,
    ) -> BaseSupervisorModel:
        cfg = self.cfg
        env = TurbofanMaintenanceEnv(df_trajectories, config=cfg)
        self.feature_names = list(env.feature_cols)

        # Fixed normalization computed once from the raw training data
        # (physical units -- see module docstring) and applied at every
        # state->tensor conversion below, so the network trains on
        # well-conditioned input without needing df_trajectories itself
        # (or the live serving path) to be pre-scaled.
        feat_matrix = (
            env.df_raw[self.feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float64)
            .values
        )
        self.feature_mean_ = np.nanmean(feat_matrix, axis=0)
        feature_std = np.nanstd(feat_matrix, axis=0)
        self.feature_std_ = np.where(feature_std < 1e-6, 1.0, feature_std)

        if self.net is None:
            self._build_model()

        logger.info(
            "Training PPO via environment rollouts: %d engines, state_dim=%d, "
            "epochs=%d, gamma=%.3f, gae_lambda=%.3f, clip=%.3f.",
            len(env.engine_ids), env.state_dim, cfg.rl_epochs, cfg.rl_gamma,
            cfg.ppo_gae_lambda, cfg.ppo_clip_ratio,
        )

        self.training_history = []
        rng = np.random.default_rng(cfg.random_seed)

        # Best-checkpoint-by-validation selection: fixes a real generalization
        # gap observed in production (catastrophic_failure_rate 0.0 on
        # training engines vs. 0.84 on held-out ones for the final-epoch
        # policy). Training return/failure-rate swing widely epoch to epoch
        # (on-policy PPO is not monotonically improving), so blindly keeping
        # whatever the last epoch converged to is a gamble. Every
        # _VAL_EVAL_INTERVAL epochs (and always the last), the current greedy
        # policy is rolled out against val_df's held-out engines and the
        # best-scoring snapshot is restored at the end -- standard
        # early-stopping-by-validation practice, applied to an RL policy.
        # Safety dominates the score lexicographically: a policy with a lower
        # catastrophic_failure_rate always wins regardless of return, since
        # that is the metric that actually matters for this system.
        best_val_score = float("-inf")
        best_val_epoch: int | None = None
        best_val_metrics: dict[str, float] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        _VAL_EVAL_INTERVAL: Final[int] = max(1, cfg.rl_epochs // 10)

        for epoch in range(cfg.rl_epochs):
            engine_order = rng.permutation(env.engine_ids)

            batch_states: list[np.ndarray] = []
            batch_actions: list[int] = []
            batch_logprobs: list[float] = []
            batch_values: list[float] = []
            batch_rewards: list[float] = []
            batch_dones: list[bool] = []
            episode_returns: list[float] = []
            episode_events: list[str] = []

            self.net.eval()
            with torch.no_grad():
                for engine_id in engine_order:
                    state = env.reset(engine_id=engine_id)
                    done = False
                    episode_return = 0.0
                    last_event = "max_steps_reached"

                    while not done:
                        state_t = torch.as_tensor(
                            self._normalize(state), dtype=torch.float32, device=self.device,
                        ).unsqueeze(0)
                        logits, value = self.net(state_t)
                        dist = Categorical(logits=logits)
                        action_t = dist.sample()
                        action = int(action_t.item())
                        logprob = float(dist.log_prob(action_t).item())

                        next_state, reward, done, info = env.step(action)

                        batch_states.append(state)
                        batch_actions.append(action)
                        batch_logprobs.append(logprob)
                        batch_values.append(float(value.item()))
                        batch_rewards.append(reward / _REWARD_SCALE)
                        batch_dones.append(done)

                        episode_return += reward
                        last_event = info.get("event", last_event)
                        state = next_state

                    episode_returns.append(episode_return)
                    episode_events.append(last_event)

            # -------------------------------------------------------------
            # GAE advantage + return computation (per-episode, using dones
            # to reset the bootstrap at each episode boundary).
            # -------------------------------------------------------------
            rewards_arr = np.asarray(batch_rewards, dtype=np.float32)
            values_arr = np.asarray(batch_values, dtype=np.float32)
            dones_arr = np.asarray(batch_dones, dtype=np.float32)

            advantages = np.zeros_like(rewards_arr)
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(rewards_arr))):
                next_non_terminal = 1.0 - dones_arr[t]
                delta = rewards_arr[t] + cfg.rl_gamma * next_value * next_non_terminal - values_arr[t]
                gae = delta + cfg.rl_gamma * cfg.ppo_gae_lambda * next_non_terminal * gae
                advantages[t] = gae
                next_value = values_arr[t] if next_non_terminal else 0.0
            returns_arr = advantages + values_arr

            adv_std = advantages.std()
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            states_t = torch.as_tensor(
                self._normalize(np.stack(batch_states)), dtype=torch.float32, device=self.device,
            )
            actions_t = torch.as_tensor(batch_actions, dtype=torch.long, device=self.device)
            old_logprobs_t = torch.as_tensor(batch_logprobs, dtype=torch.float32, device=self.device)
            returns_t = torch.as_tensor(returns_arr, dtype=torch.float32, device=self.device)
            advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

            n_samples = states_t.shape[0]
            minibatch_size = min(cfg.rl_batch_size, n_samples)

            self.net.train()
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            n_updates = 0

            for _ in range(cfg.ppo_update_epochs):
                perm = torch.randperm(n_samples)
                for start in range(0, n_samples, minibatch_size):
                    idx = perm[start:start + minibatch_size]
                    if idx.numel() == 0:
                        continue

                    logits, values = self.net(states_t[idx])
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(actions_t[idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - old_logprobs_t[idx])
                    surr1 = ratio * advantages_t[idx]
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.ppo_clip_ratio, 1.0 + cfg.ppo_clip_ratio
                    ) * advantages_t[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = nn.functional.mse_loss(values.squeeze(-1), returns_t[idx])

                    total_loss = (
                        policy_loss
                        + cfg.ppo_value_coef * value_loss
                        - cfg.ppo_entropy_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    epoch_policy_loss += float(policy_loss.item())
                    epoch_value_loss += float(value_loss.item())
                    epoch_entropy += float(entropy.item())
                    n_updates += 1

            mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            failure_rate = float(
                np.mean([e in ("unplanned_catastrophic_failure",
                                "unplanned_failure_during_monitoring",
                                "unplanned_failure_during_inspection") for e in episode_events])
            ) if episode_events else 0.0

            history_entry = {
                "epoch": epoch,
                "mean_episode_return": round(mean_return, 3),
                "catastrophic_failure_rate": round(failure_rate, 4),
                "policy_loss": round(epoch_policy_loss / max(n_updates, 1), 5),
                "value_loss": round(epoch_value_loss / max(n_updates, 1), 5),
                "entropy": round(epoch_entropy / max(n_updates, 1), 5),
            }
            is_eval_epoch = epoch % _VAL_EVAL_INTERVAL == 0 or epoch == cfg.rl_epochs - 1

            if val_df is not None and is_eval_epoch:
                # Mark fitted just long enough for evaluate_policy's own
                # is_fitted check; this doesn't publish the model anywhere,
                # it only runs greedy rollouts against held-out engines.
                self.is_fitted = True
                val_metrics = self.evaluate_policy(val_df, greedy=True)
                history_entry["val_mean_episode_return"] = val_metrics["mean_episode_return"]
                history_entry["val_catastrophic_failure_rate"] = val_metrics["catastrophic_failure_rate"]

                # Lexicographic score: minimizing catastrophic_failure_rate
                # dominates: multiplying it by a constant far larger than any
                # plausible return spread guarantees a safer policy always
                # outranks a riskier one with a merely higher return.
                val_score = (
                    -val_metrics["catastrophic_failure_rate"] * 1_000_000.0
                    + val_metrics["mean_episode_return"]
                )
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_epoch = epoch
                    best_val_metrics = val_metrics
                    best_state_dict = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    logger.info(
                        "PPO epoch %d/%d: new best validation checkpoint "
                        "(val_catastrophic_failure_rate=%.3f, val_mean_return=%.2f).",
                        epoch + 1, cfg.rl_epochs,
                        val_metrics["catastrophic_failure_rate"], val_metrics["mean_episode_return"],
                    )

            self.training_history.append(history_entry)

            if epoch % max(1, cfg.rl_epochs // 10) == 0 or epoch == cfg.rl_epochs - 1:
                logger.info(
                    "PPO epoch %d/%d: mean_episode_return=%.2f, "
                    "catastrophic_failure_rate=%.3f, policy_loss=%.4f, entropy=%.4f",
                    epoch + 1, cfg.rl_epochs, mean_return, failure_rate,
                    history_entry["policy_loss"], history_entry["entropy"],
                )

        self.is_fitted = True
        self.trained_via_environment = True

        if best_state_dict is not None:
            final_val_metrics = self.evaluate_policy(val_df, greedy=True) if val_df is not None else None
            self.net.load_state_dict(best_state_dict)
            logger.info(
                "Restoring best validation checkpoint from epoch %d/%d "
                "(val_catastrophic_failure_rate=%.3f, val_mean_return=%.2f) over the final "
                "epoch's policy%s.",
                best_val_epoch + 1, cfg.rl_epochs,
                best_val_metrics["catastrophic_failure_rate"], best_val_metrics["mean_episode_return"],
                (
                    f" (final epoch scored val_catastrophic_failure_rate="
                    f"{final_val_metrics['catastrophic_failure_rate']:.3f})"
                    if final_val_metrics is not None else ""
                ),
            )
            self.best_checkpoint_epoch = best_val_epoch
            self.best_checkpoint_val_metrics = best_val_metrics
        else:
            self.best_checkpoint_epoch = None
            self.best_checkpoint_val_metrics = None

        logger.info("PPO environment training complete over %d epochs.", cfg.rl_epochs)
        return self

    def evaluate_policy(self, df_trajectories: pd.DataFrame, greedy: bool = True) -> dict[str, float]:
        """Roll the current policy out over every engine trajectory in `df_trajectories`.

        Returns real episode-based metrics (mean return, success rate,
        catastrophic-failure rate) -- the primary evidence for a trained RL
        policy, as opposed to a label-agreement classification metric.
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before evaluate_policy.")

        env = TurbofanMaintenanceEnv(df_trajectories, feature_columns=self.feature_names, config=self.cfg)

        episode_returns: list[float] = []
        episode_events: list[str] = []

        self.net.eval()
        with torch.no_grad():
            for engine_id in env.engine_ids:
                state = env.reset(engine_id=engine_id)
                done = False
                episode_return = 0.0
                last_event = "max_steps_reached"

                while not done:
                    state_t = torch.as_tensor(
                        self._normalize(state), dtype=torch.float32, device=self.device,
                    ).unsqueeze(0)
                    logits, _ = self.net(state_t)
                    if greedy:
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action = int(Categorical(logits=logits).sample().item())

                    state, reward, done, info = env.step(action)
                    episode_return += reward
                    last_event = info.get("event", last_event)

                episode_returns.append(episode_return)
                episode_events.append(last_event)

        n = max(len(episode_events), 1)
        catastrophic = sum(
            1 for e in episode_events
            if e in ("unplanned_catastrophic_failure", "unplanned_failure_during_monitoring",
                      "unplanned_failure_during_inspection")
        )
        successful_maintenance = sum(
            1 for e in episode_events
            if e in ("successful_scheduled_maintenance", "emergency_shutdown_executed")
        )

        return {
            "episodes_evaluated": len(episode_events),
            "mean_episode_return": round(float(np.mean(episode_returns)), 3) if episode_returns else 0.0,
            "std_episode_return": round(float(np.std(episode_returns)), 3) if episode_returns else 0.0,
            "catastrophic_failure_rate": round(catastrophic / n, 4),
            "successful_maintenance_rate": round(successful_maintenance / n, 4),
        }

    # ------------------------------------------------------------------
    # Legacy behaviour-cloning fallback (no trajectory structure available)
    # ------------------------------------------------------------------

    def _fit_behaviour_cloning(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> BaseSupervisorModel:
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

        if self.net is None:
            self._build_model()

        states_t = torch.tensor(X_mat, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(y_arr, dtype=torch.long, device=self.device)

        self.net.train()
        criterion_ce = nn.CrossEntropyLoss()

        for _ in range(self.cfg.ppo_update_epochs * 5):
            self.optimizer.zero_grad()
            logits, values = self.net(states_t)
            loss_policy = criterion_ce(logits, actions_t)
            value_targets = actions_t.float().unsqueeze(1) / 4.0
            loss_value = nn.MSELoss()(values, value_targets)
            total_loss = loss_policy + self.cfg.ppo_value_coef * loss_value
            total_loss.backward()
            self.optimizer.step()

        self.is_fitted = True
        self.trained_via_environment = False
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict softmax action probabilities pi(a|s) for X."""
        if not self.is_fitted or self.net is None:
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
        # Normalize with the fixed training-time mean/std before the forward
        # pass -- X_mat arrives in raw physical units (from the live
        # serving path, or from any raw dataframe a caller supplies), same
        # as during training. See _fit_via_environment / the module docstring.
        states_t = torch.tensor(self._normalize(X_mat), dtype=torch.float32, device=self.device)

        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(states_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def save(self, filepath: Path | str) -> Path:
        """Serialize PPO weights and metadata."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"Cannot save unfitted model '{self.model_name}'.")

        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "is_fitted": self.is_fitted,
            "net_state_dict": self.net.state_dict(),
            "trained_via_environment": self.trained_via_environment,
            "training_history": self.training_history,
            "feature_mean_": self.feature_mean_,
            "feature_std_": self.feature_std_,
            "best_checkpoint_epoch": self.best_checkpoint_epoch,
            "best_checkpoint_val_metrics": self.best_checkpoint_val_metrics,
        }
        torch.save(state, filepath)
        logger.info("Saved %s RL model artifact to: %s", self.model_name, filepath)
        return filepath

    def load(self, filepath: Path | str) -> BaseSupervisorModel:
        """Deserialize PPO weights and metadata."""
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found: {filepath}")

        # weights_only=False: this checkpoint is our own (classes_ etc. are
        # numpy, which weights_only=True, the default since PyTorch 2.6,
        # refuses to unpickle), never loaded from an external source.
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model_name = state["model_name"]
        self.feature_names = state["feature_names"]
        self.classes_ = state["classes_"]
        self.is_fitted = state["is_fitted"]
        self.trained_via_environment = state.get("trained_via_environment", False)
        self.training_history = state.get("training_history", [])
        self.feature_mean_ = state.get("feature_mean_")
        self.feature_std_ = state.get("feature_std_")
        self.best_checkpoint_epoch = state.get("best_checkpoint_epoch")
        self.best_checkpoint_val_metrics = state.get("best_checkpoint_val_metrics")

        state_dim = len(self.feature_names) if self.feature_names else 30
        self.net = ActorCriticNet(state_dim=state_dim, action_dim=len(DECISION_CLASSES)).to(self.device)
        self.net.load_state_dict(state["net_state_dict"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.rl_learning_rate)

        logger.info("Loaded %s RL model artifact from: %s", self.model_name, filepath)
        return self
