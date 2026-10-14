"""
rl_environment.py

Sequential Markov Decision Process (MDP) Maintenance Environment for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Provides a gym-compatible sequential environment (`TurbofanMaintenanceEnv`) that simulates
engine degradation trajectories based on multivariate sensor outputs, multi-horizon failure
risks, RUL estimates, and anomaly scores.

Key Components
--------------
1. State Space: Unified machine health state vector (features derived from RUL, Risk, & Anomaly agents).
2. Action Space: Discrete(5)
    0: continue_operation
    1: monitor_closely
    2: schedule_inspection
    3: schedule_maintenance
    4: immediate_maintenance
3. Reward Function: Multi-objective cost-aware trade-off function balancing downtime, maintenance costs,
   and catastrophic failure prevention.

Status: wired into PPOSupervisorModel
-------------------------------------
``PPOSupervisorModel.fit`` (``models/ppo_supervisor.py``) builds this
environment from the training trajectories and trains against it by rollout
collection + GAE + a clipped surrogate objective -- genuine on-policy PPO,
not supervised fitting on labelled actions. ``DQNSupervisorModel`` is
intentionally left on supervised fitting: it is retained only as a labelled
baseline for comparison, per the project's PPO-over-DQN preference.

Design note on episode termination: ``step`` sets ``done=True`` on any
maintenance action (3 or 4) and leaves the episode running otherwise. This is
an optimal-stopping formulation on purpose -- "when, if ever, should I
intervene in this trajectory" -- not a bug: once a maintenance action is
taken the trajectory's future no longer describes an in-service engine, so
there is nothing further to control.

Fixed modelling caveat: reaching ``max_steps`` without a maintenance action
used to be charged as ``cost_unplanned_failure`` unconditionally, which
mislabels censored (test-unit) trajectories -- ones where the data simply
stops before failure -- as catastrophic. There is no ground-truth failure
flag in the processed dataset (`true_rul` etc. are excluded upstream as
forbidden features), so ``_ends_in_failure`` uses the best available proxy:
a trajectory whose final row's ``predicted_rul`` is at or below
``CENSORED_RUL_THRESHOLD`` is treated as having actually run to failure;
one that ends with a materially higher predicted RUL is treated as censored
and reaching the end of it is reward-neutral rather than penalised.

Fixed modelling bug (scale mismatch): the reward thresholds in ``step()``
(e.g. ``predicted_rul > 60.0``, ``risk_10 > 0.85``) are written in raw
physical units -- cycle counts and 0-1 probabilities. But the training
data this environment is actually built from
(``processed/Maintenance_Supervisor/supervisor_scaled_dataset.csv``) has
already been through ``RobustScaler`` upstream, so ``predicted_rul``
arrives with a scaled mean of roughly 0.04 and ``risk_10`` with a scaled
mean of roughly 10 (see
``reports/Maintenance_Supervisor/feature_scaling_report.json``). Compared
against raw-unit thresholds, those scaled values made every reward branch
degenerate to (effectively) the same outcome regardless of true engine
health, which is exactly what produced a trained policy with zero
variance across evaluation episodes and a single collapsed action for
100% of rows in production runs. ``__init__`` now detects, once per
environment (mirroring the batch-level detection heuristic already used
in ``decision_fusion/safety_policy.py``), whether
``predicted_rul``/``risk_10`` arrived standardized, and
``_unscale_rul``/``_unscale_risk`` undo that transform before any reward
threshold is evaluated in ``reset()`` or ``step()``. The unscale
*constants* used here were independently re-derived by regressing
``processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv``
(pre-scaling) against ``supervisor_scaled_dataset.csv`` (post-scaling) for
this pipeline's actual fit -- the RUL constants agree with
``safety_policy.py``'s to 5+ decimal places, but its risk constants
(0.003, 1.14) do not match this data at all (see ``_unscale_risk``) and
should be corrected there too. The state feature vector fed to the
policy/value networks in ``_get_state()`` is left scaled, since
standardized inputs are correct practice for the neural network -- only
the reward-threshold comparisons needed physical units.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    DECISION_CLASSES,
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()
_CFG = SupervisorConfig()

# A trajectory whose final observed row has predicted_rul at or below this
# many cycles is treated as having actually run to failure. Above it, the
# trajectory is treated as censored (data collection stopped, not the
# engine) and running out the clock is not charged as a catastrophic event.
CENSORED_RUL_THRESHOLD: Final[float] = 2.0

# The state features actually available on the live serving path
# (POST /integration/run-full -> supervisor_bridge.run_supervisor_bridge),
# i.e. exactly the fields RULPredictionResponse / FailureRiskPredictionResponse
# return, plus the 3 of 12 ANOMALY_FEATURES the Anomaly & Health component's
# exchange contract (dashboard_api.py's REQUIRED_EXCHANGE_COLUMNS) actually
# carries today: anomaly_score, anomaly_flag (-> is_anomalous) and severity
# (-> anomaly_severity). Everything else in cfg.all_model_features either
# comes from a per-sensor/diagnostic anomaly signal that component doesn't
# export live (anomaly_threshold, residual_anomaly_score,
# isolation_forest_score, mahalanobis_score, top_sensor_*_score,
# context_confidence, context_drift_flag) or from a cross-component
# engineered feature (risk_anomaly_interaction, rul_risk_interaction) that
# only feature_engineering.py computes -- a step the live path never runs.
#
# Training PPO on the full cfg.all_model_features set (40 columns) means 11
# of them get silently zero-filled on every live decision, which is
# materially out-of-distribution for a policy that never saw those columns
# as zero during training. This is the default specifically so that gap
# can't reopen by accident; pass feature_columns explicitly to override it
# for an offline experiment.
LIVE_EXCHANGE_FEATURE_COLUMNS: Final[list[str]] = [
    # RUL (RULPredictionResponse -- all 10 fields)
    "predicted_rul", "lower_bound", "upper_bound", "uncertainty_width",
    "degradation_speed", "degradation_acceleration", "lifecycle_state",
    "confidence_collapse_flag", "sudden_health_drop_flag", "instability_flag",
    # Failure-Risk (FailureRiskPredictionResponse -- all 16 usable fields;
    # action_hint_for_supervisor is categorical and gets one-hot encoded
    # upstream of training, so it never survives as a single raw column)
    "risk_10", "risk_30", "risk_50", "uncertainty_10", "uncertainty_30",
    "uncertainty_50", "threshold_10", "threshold_30", "threshold_50",
    "control_risk", "risk_velocity", "risk_acceleration", "risk_state",
    "predicted_fail_in_10", "predicted_fail_in_30", "predicted_fail_in_50",
    # Anomaly & Health (only the 3 of 12 the live exchange contract carries)
    "anomaly_score", "is_anomalous", "anomaly_severity",
]


class TurbofanMaintenanceEnv:
    """
    Sequential MDP Environment simulating industrial engine degradation cycles
    for reinforcement learning policy optimization.
    """

    def __init__(
        self,
        df_episodes: pd.DataFrame,
        feature_columns: list[str] | None = None,
        config: SupervisorConfig | None = None,
    ):
        self.cfg = config or _CFG
        self.df_raw = df_episodes.copy()

        # Group dataset by engine trajectory
        self.engine_col = self.cfg.engine_id_column if self.cfg.engine_id_column in df_episodes.columns else "engine_id"
        self.cycle_col = self.cfg.cycle_column if self.cfg.cycle_column in df_episodes.columns else "cycle"

        if self.engine_col not in self.df_raw.columns:
            self.df_raw[self.engine_col] = 1

        self.engine_ids = sorted(self.df_raw[self.engine_col].unique())
        self.episodes: dict[Any, pd.DataFrame] = {
            eng_id: group.sort_values(self.cycle_col).reset_index(drop=True)
            for eng_id, group in self.df_raw.groupby(self.engine_col)
        }

        # Determine state feature columns. Default to LIVE_EXCHANGE_FEATURE_COLUMNS,
        # not the full cfg.all_model_features set, so a policy trained without an
        # explicit override matches what POST /integration/run-full actually
        # supplies at inference -- see the constant's docstring above.
        if feature_columns:
            self.feature_cols = [c for c in feature_columns if c in self.df_raw.columns]
        else:
            self.feature_cols = [
                c for c in LIVE_EXCHANGE_FEATURE_COLUMNS if c in self.df_raw.columns
            ]
            if not self.feature_cols:
                self.feature_cols = list(self.df_raw.select_dtypes(include=[np.number]).columns)

        self.state_dim = len(self.feature_cols)
        self.action_dim = len(DECISION_CLASSES)  # 5 actions

        self.current_engine_idx = 0
        self.current_step = 0
        self.current_trajectory: pd.DataFrame | None = None
        self.max_steps = 0
        self.trajectory_ends_in_failure = True

        # Detect once, at the whole-dataset level (same heuristic as
        # safety_policy.py's batch check), whether predicted_rul / risk_10
        # arrived RobustScaler-standardized rather than in raw physical
        # units. See the scale-mismatch note in the module docstring.
        self._rul_is_scaled = self._detect_scaled_rul(self.df_raw)
        self._risk_is_scaled = self._detect_scaled_risk(self.df_raw)
        if self._rul_is_scaled or self._risk_is_scaled:
            logger.info(
                "TurbofanMaintenanceEnv: detected standardized input "
                "(rul_scaled=%s, risk_scaled=%s) -- reward thresholds will "
                "be evaluated against unscaled physical values.",
                self._rul_is_scaled, self._risk_is_scaled,
            )

    @staticmethod
    def _detect_scaled_rul(df: pd.DataFrame) -> bool:
        """Mirrors safety_policy.py's RUL-standardization heuristic:
        raw RUL (a cycle count) is never negative, so requiring at least one
        negative value alongside a small max keeps this inert for genuinely
        low (but raw) RUL batches while still catching standardized input."""
        if "predicted_rul" not in df.columns or df.empty:
            return False
        vals = pd.to_numeric(df["predicted_rul"], errors="coerce").values
        if np.all(np.isnan(vals)):
            return False
        return bool(np.nanmax(vals) < 15.0 and np.nanmin(vals) < 0)

    @staticmethod
    def _detect_scaled_risk(df: pd.DataFrame) -> bool:
        """Mirrors safety_policy.py's risk-standardization heuristic: a raw
        risk_10 probability is bounded in [0, 1], so any value above 2.0
        can only be a standardized (unbounded) feature."""
        if "risk_10" not in df.columns or df.empty:
            return False
        vals = pd.to_numeric(df["risk_10"], errors="coerce").values
        if np.all(np.isnan(vals)):
            return False
        return bool(np.nanmax(vals) > 2.0)

    def _unscale_rul(self, value: float) -> float:
        """Undo the RobustScaler transform on predicted_rul so reward
        thresholds compare against a physical cycle count. Uses the same
        center/scale constants already relied on in safety_policy.py, so
        the RL environment and the safety guardrails agree on what "raw"
        means for this pipeline's fitted scaler."""
        if self._rul_is_scaled:
            return value * 101.69 + 99.32
        return value

    def _unscale_risk(self, value: float) -> float:
        """Undo the RobustScaler transform on risk_10 and clip back into the
        valid [0, 1] probability range for reward-threshold comparisons.

        NOTE: this does NOT reuse safety_policy.py's risk unscale constants
        (0.003, 1.14) -- those were verified against this pipeline's actual
        fitted scaler (via linear regression of
        processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv
        [pre-scaling] against supervisor_scaled_dataset.csv [post-scaling])
        and found to be wrong for risk_10 (they over-trigger the >= 0.82
        critical-risk safety override on ~6,600 rows where only ~1,350
        actually cross the true raw threshold -- a ~5x over-fire rate).
        safety_policy.py should be corrected separately with the same
        constants used here; see the audit report for the full
        before/after comparison.
        """
        if self._risk_is_scaled:
            return float(np.clip(value * 0.006 + 0.0061, 0.0, 1.0))
        return value

    def reset(self, engine_id: Any | None = None) -> np.ndarray:
        """Reset environment to the beginning of a trajectory."""
        if engine_id is not None and engine_id in self.episodes:
            self.current_trajectory = self.episodes[engine_id]
        else:
            eng_id = np.random.choice(self.engine_ids)
            self.current_trajectory = self.episodes[eng_id]

        self.current_step = 0
        self.max_steps = len(self.current_trajectory)

        last_row = self.current_trajectory.iloc[-1]
        final_rul_raw = last_row.get("predicted_rul", last_row.get("RUL", None))
        final_rul = (
            self._unscale_rul(float(final_rul_raw)) if final_rul_raw is not None else None
        )
        self.trajectory_ends_in_failure = (
            final_rul is not None and final_rul <= CENSORED_RUL_THRESHOLD
        )

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Extract state feature vector for the current step."""
        if self.current_trajectory is None or self.current_step >= self.max_steps:
            return np.zeros(self.state_dim, dtype=np.float32)

        row = self.current_trajectory.iloc[self.current_step]
        state = row[self.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        return state.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute maintenance action step.

        Actions
        -------
        0: continue_operation
        1: monitor_closely
        2: schedule_inspection
        3: schedule_maintenance
        4: immediate_maintenance
        """
        if self.current_trajectory is None or self.current_step >= self.max_steps:
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {}

        row = self.current_trajectory.iloc[self.current_step]
        predicted_rul_raw = float(row.get("predicted_rul", row.get("RUL", self.max_steps - self.current_step)))
        risk_10_raw = float(row.get("risk_10", row.get("failure_risk", 0.0)))
        # Unscale to physical units (cycle count, 0-1 probability) before any
        # reward threshold below is evaluated -- see the scale-mismatch note
        # in the module docstring and _unscale_rul/_unscale_risk above.
        predicted_rul = self._unscale_rul(predicted_rul_raw)
        risk_10 = self._unscale_risk(risk_10_raw)
        anomaly_score = float(row.get("anomaly_score", 0.0))

        reward = 0.0
        done = False
        info = {
            "action_name": DECISION_CLASSES[action],
            "step": self.current_step,
            "predicted_rul": predicted_rul,
            "risk_10": risk_10,
        }

        # A trajectory only reaches genuine catastrophic failure if it is
        # both at (or past) its last observed row AND that row's own RUL
        # reading marks a real run-to-failure ending, not a censored cutoff
        # (see CENSORED_RUL_THRESHOLD / trajectory_ends_in_failure).
        at_trajectory_end = self.current_step >= self.max_steps - 1
        reaches_failure = (
            predicted_rul <= 1.0
            or (at_trajectory_end and self.trajectory_ends_in_failure)
        )

        # ---------------------------------------------------------------------
        # Action Dynamics & Multi-Objective Reward Calculation
        # ---------------------------------------------------------------------

        # Fixed modelling bug (delayed-only failure signal): "wait" actions
        # (0/1/2) used to be free or flat-cost every step regardless of how
        # close the engine actually was to failure, with the entire
        # cost_unplanned_failure penalty landing only once, on the terminal
        # step. A retrained policy under that shape learned "continuing is
        # free" and ran 84% of held-out episodes to genuine failure (see
        # rl_wait_risk_penalty_weight's docstring in supervisor_config.py).
        # This dense, risk-proportional penalty applies to every wait step so
        # the agent gets gradient signal long before the delayed terminal
        # event, not just at it.
        wait_risk_penalty = self.cfg.rl_wait_risk_penalty_weight * self.cfg.cost_unplanned_failure * risk_10

        if action == 0:  # continue_operation
            reward -= self.cfg.cost_continue_operation
            reward -= wait_risk_penalty
            if reaches_failure:
                reward -= self.cfg.cost_unplanned_failure
                done = True
                info["event"] = "unplanned_catastrophic_failure"
            elif at_trajectory_end:
                done = True
                info["event"] = "censored_trajectory_end"
            elif risk_10 > 0.85:
                reward -= 500.0

        elif action == 1:  # monitor_closely
            reward -= self.cfg.cost_monitoring
            reward -= wait_risk_penalty
            if reaches_failure:
                reward -= self.cfg.cost_unplanned_failure
                done = True
                info["event"] = "unplanned_failure_during_monitoring"
            elif at_trajectory_end:
                done = True
                info["event"] = "censored_trajectory_end"

        elif action == 2:  # schedule_inspection
            reward -= self.cfg.cost_inspection
            reward -= wait_risk_penalty
            if reaches_failure:
                reward -= self.cfg.cost_unplanned_failure
                done = True
                info["event"] = "unplanned_failure_during_inspection"
            elif at_trajectory_end:
                done = True
                info["event"] = "censored_trajectory_end"

        elif action == 3:  # schedule_maintenance
            reward -= self.cfg.cost_scheduled_maintenance
            done = True
            info["event"] = "successful_scheduled_maintenance"
            if predicted_rul > 60.0:
                reward -= (predicted_rul - 60.0) * 10.0
            else:
                reward += 200.0

        elif action == 4:  # immediate_maintenance
            reward -= self.cfg.cost_immediate_maintenance
            done = True
            info["event"] = "emergency_shutdown_executed"
            if predicted_rul < 15.0 or risk_10 > 0.70:
                reward += 300.0
            else:
                reward -= 400.0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        next_state = self._get_state()
        return next_state, float(reward), done, info
