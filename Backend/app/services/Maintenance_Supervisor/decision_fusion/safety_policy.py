"""
safety_policy.py

Safety Policy & Hard Risk Guardrails for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Enforces non-negotiable safety guardrails and policy overrides on candidate ML model
decisions before final decision emission.

Safety Rules (Fail-Safe Overrides):
-----------------------------------
1. Critical RUL Override: If predicted_rul <= 10.0 -> OVERRIDE to "immediate_maintenance".
2. Critical Risk Override: If risk_10 >= 0.82 -> OVERRIDE to "immediate_maintenance".
3. Critical Anomaly Override: If anomaly_severity == "critical" -> OVERRIDE to "immediate_maintenance".
4. High Risk Override: If risk_30 >= 0.72 -> ENSURE AT LEAST "schedule_maintenance".
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    ARTIFACT_ROOT,
    DECISION_TO_SEVERITY,
    SEVERITY_TO_DECISION,
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()
_CFG = SupervisorConfig()
_ADAPTIVE_POLICY_PATH: Final[Path] = ARTIFACT_ROOT / "adaptive_policy_manifest.json"


def _load_adaptive_thresholds() -> dict[str, float] | None:
    """Load feedback::adaptive_policy.py's threshold nudges, if any exist.

    feedback/adaptive_policy.py computes these from logged engineer
    overrides (see its module docstring) but previously wrote them to a
    manifest nothing read -- the guardrails always used the static config
    defaults regardless of what the manifest said. Missing file / bad JSON
    is not an error: it just means no adaptation has run yet, so the
    guardrails fall back to the config defaults.
    """
    if not _ADAPTIVE_POLICY_PATH.exists():
        return None
    try:
        with open(_ADAPTIVE_POLICY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "immediate_rul_threshold": float(data["immediate_rul_threshold_adjusted"]),
            "critical_risk_10_threshold": float(data["critical_risk_10_threshold_adjusted"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Ignoring unreadable adaptive policy manifest %s: %s", _ADAPTIVE_POLICY_PATH, exc)
        return None


@dataclass
class PolicyOverrideResult:
    final_decisions: np.ndarray
    override_flags: np.ndarray
    override_reasons: list[str]
    total_overrides: int


class SupervisorSafetyPolicy:
    """Enforces safety guardrails over candidate model decisions."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

        adapted = _load_adaptive_thresholds()
        self.immediate_rul_threshold = (
            adapted["immediate_rul_threshold"] if adapted else self.cfg.immediate_rul_threshold
        )
        self.critical_risk_10_threshold = (
            adapted["critical_risk_10_threshold"] if adapted else self.cfg.critical_risk_10_threshold
        )
        if adapted:
            logger.info(
                "Safety policy using feedback-adapted thresholds: "
                "immediate_rul_threshold=%.2f, critical_risk_10_threshold=%.4f",
                self.immediate_rul_threshold, self.critical_risk_10_threshold,
            )

    def apply_policy_overrides(
        self,
        candidate_decisions: np.ndarray | list[str],
        df_features: pd.DataFrame,
    ) -> PolicyOverrideResult:
        # dtype=object avoids numpy inferring a fixed-width unicode dtype from
        # the initial values, which would silently truncate longer decision
        # strings (e.g. "immediate_maintenance") assigned in-place below when
        # a safety override fires for a sample whose original prediction was
        # a shorter class name.
        decisions_arr = np.array([str(d).strip().lower() for d in candidate_decisions], dtype=object)
        n_samples = len(decisions_arr)

        rul_raw = df_features.get("predicted_rul", pd.Series(999.0, index=df_features.index)).values
        r10_raw = df_features.get("risk_10", pd.Series(0.0, index=df_features.index)).values
        r30_raw = df_features.get("risk_30", pd.Series(0.0, index=df_features.index)).values
        anom_sev = df_features.get("anomaly_severity", pd.Series("none", index=df_features.index)).astype(str).str.lower().values

        # Detect if features are standardized and unscale for physical safety checks.
        # A magnitude-only check (max < 15.0) is unsound: a genuinely critical RAW
        # RUL (e.g. 5 cycles left -- exactly the case the immediate-maintenance rule
        # exists to catch) also has max < 15, so that check alone "unscales" a real
        # critical value away from danger and the safety override never fires. Raw
        # RUL (a cycle count) is never negative; standardized RUL, centered near 0,
        # takes negative values whenever the batch spans below its mean. Requiring
        # at least one negative value as corroboration keeps the check inert for
        # real critical batches while still catching genuinely standardized input.
        if len(rul_raw) > 0 and np.nanmax(rul_raw) < 15.0 and np.nanmin(rul_raw) < 0:
            rul = rul_raw * 101.69 + 99.32
        else:
            rul = rul_raw

        # Fixed bug: risk_10 and risk_30 each have their own fitted RobustScaler
        # center/scale (they are not interchangeable), and the single shared
        # constants previously used here (0.003, 1.14) did not match either
        # feature's actual fit. Re-derived by regressing
        # processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv
        # (pre-scaling) against supervisor_scaled_dataset.csv (post-scaling):
        # the old formula over-fired the >= 0.82 critical-risk override on
        # ~6,600 rows where only ~1,350 actually crossed the true raw
        # threshold (a ~5x over-fire rate), and similarly over-fired the
        # risk_30 >= 0.72 override roughly 2.4x. These constants should be
        # re-verified if the upstream scaler is ever refit on new data.
        if len(r10_raw) > 0 and np.nanmax(r10_raw) > 2.0:
            r10 = np.clip(r10_raw * 0.006 + 0.0061, 0.0, 1.0)
        else:
            r10 = r10_raw

        if len(r30_raw) > 0 and np.nanmax(r30_raw) > 2.0:
            r30 = np.clip(r30_raw * 0.062 + 0.0153, 0.0, 1.0)
        else:
            r30 = r30_raw

        final_decisions = decisions_arr.copy()
        override_flags = np.zeros(n_samples, dtype=bool)
        override_reasons = ["None"] * n_samples

        for i in range(n_samples):
            curr_sev = DECISION_TO_SEVERITY.get(final_decisions[i], 0)
            target_sev = curr_sev
            reasons = []

            # Rule 1: Critical RUL
            if rul[i] <= self.immediate_rul_threshold:
                if target_sev < 4:
                    target_sev = 4
                    reasons.append(f"RUL <= {self.immediate_rul_threshold}")

            # Rule 2: Critical Risk 10
            if r10[i] >= self.critical_risk_10_threshold:
                if target_sev < 4:
                    target_sev = 4
                    reasons.append(f"Risk_10 >= {self.critical_risk_10_threshold}")

            # Rule 3: Critical Anomaly
            if anom_sev[i] == "critical":
                if target_sev < 4:
                    target_sev = 4
                    reasons.append("Critical Anomaly Severity")

            # Rule 4: High Risk 30 -> Schedule Maintenance
            if r30[i] >= self.cfg.maintenance_risk_30_threshold:
                if target_sev < 3:
                    target_sev = 3
                    reasons.append(f"Risk_30 >= {self.cfg.maintenance_risk_30_threshold}")

            if target_sev != curr_sev:
                final_decisions[i] = SEVERITY_TO_DECISION.get(target_sev, "immediate_maintenance")
                override_flags[i] = True
                override_reasons[i] = "; ".join(reasons)

        return PolicyOverrideResult(
            final_decisions=final_decisions,
            override_flags=override_flags,
            override_reasons=override_reasons,
            total_overrides=int(np.sum(override_flags)),
        )
