"""
policy_evaluator.py

Maintenance-Decision-Optimization Evaluator for the Autonomous Maintenance
Supervisor Agent -- computes the three metrics the Proposal Report (S3.6)
specifies for this deliverable, as three distinct numbers rather than one
folded-together cost figure:

1. Maintenance cost reduction
2. System downtime reduction
3. Failure prevention rate

Why this exists separately from compare_baselines.py / cost_evaluator.py
--------------------------------------------------------------------------
Those two evaluate the champion by feeding it rows independently through
DecisionFusionEngine, which is the right thing to do for a classification-
style report but does not exercise the champion the way it actually
operates: PPO is a *sequential* policy, trained and defined by
TurbofanMaintenanceEnv episodes, not independent per-row classification.
This module rolls every candidate policy out through that same environment,
engine by engine, so "how it's evaluated" matches "how it actually makes
decisions" -- and it reports downtime and failure-prevention as their own
numbers because folding everything into one dollar figure is exactly what
made it impossible to see, in earlier reports, whether an improvement came
from fewer failures, less downtime, or lower cost.

Downtime-hours assumption (disclosed, not measured)
------------------------------------------------------
Neither C-MAPSS nor this codebase records how long an engine is actually
out of service per event type, so DOWNTIME_HOURS_BY_EVENT below is an
explicit modelling assumption, not a fitted or measured value -- order-of-
magnitude only, reflecting standard aviation MRO practice: an unplanned
in-flight-driven removal (unscheduled teardown, logistics not pre-staged)
takes materially longer than an expedited-but-planned removal, which in
turn takes longer than a slotted scheduled shop visit. Every report this
module produces carries these constants alongside the numbers they produced,
the same disclosure pattern already used in counterfactual_optimizer.py's
OPERATIONAL_SENSITIVITY.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    ARTIFACT_ROOT,
    PROCESSED_ROOT,
    REPORTS_ROOT,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    DECISION_CLASSES,
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.models.model_registry import load_model
from app.services.Maintenance_Supervisor.rl_agent.rl_environment import TurbofanMaintenanceEnv
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_TEST_SPLIT_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_RAW_ENCODED_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_encoded_dataset.csv"
_CHAMPION_MANIFEST: Final[Path] = ARTIFACT_ROOT / "champion_model_manifest.json"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "policy_evaluation_report.json"

#: See module docstring -- a disclosed modelling assumption, not a fitted value.
DOWNTIME_HOURS_BY_EVENT: Final[dict[str, float]] = {
    "unplanned_catastrophic_failure": 72.0,
    "unplanned_failure_during_monitoring": 72.0,
    "unplanned_failure_during_inspection": 72.0,
    "emergency_shutdown_executed": 24.0,
    "successful_scheduled_maintenance": 12.0,
    "censored_trajectory_end": 0.0,
    "max_steps_reached": 0.0,
}

_CATASTROPHIC_EVENTS: Final[frozenset[str]] = frozenset({
    "unplanned_catastrophic_failure",
    "unplanned_failure_during_monitoring",
    "unplanned_failure_during_inspection",
})

PolicyFn = Callable[[pd.Series, int, int], int]


def _load_raw_test_engines() -> pd.DataFrame:
    """Partition the raw (pre-scaling) encoded dataset onto test.csv's engine membership.

    Test, not validation: validation engines were used for checkpoint
    selection during training (see ppo_supervisor.py), so reporting final
    numbers on them would double-dip the same data the model was already
    selected to look good on. Test engines were never touched by anything.
    """
    test_cols = pd.read_csv(_TEST_SPLIT_PATH, nrows=0).columns
    usecols = [c for c in (ENGINE_ID_COLUMN, SUBSET_COLUMN) if c in test_cols]
    test_keys_df = pd.read_csv(_TEST_SPLIT_PATH, usecols=usecols, low_memory=False)
    if SUBSET_COLUMN in test_keys_df.columns:
        test_keys = set(zip(test_keys_df[ENGINE_ID_COLUMN], test_keys_df[SUBSET_COLUMN]))
    else:
        test_keys = {(eid, None) for eid in test_keys_df[ENGINE_ID_COLUMN]}

    raw = pd.read_csv(_RAW_ENCODED_PATH, low_memory=False)
    if SUBSET_COLUMN in raw.columns:
        raw_keys = pd.Series(list(zip(raw[ENGINE_ID_COLUMN], raw[SUBSET_COLUMN])), index=raw.index)
    else:
        raw_keys = pd.Series([(eid, None) for eid in raw[ENGINE_ID_COLUMN]], index=raw.index)

    return raw.loc[raw_keys.isin(test_keys)].reset_index(drop=True)


def _cost_for_action(cfg: SupervisorConfig, action: int) -> float:
    return [
        cfg.cost_continue_operation,
        cfg.cost_monitoring,
        cfg.cost_inspection,
        cfg.cost_scheduled_maintenance,
        cfg.cost_immediate_maintenance,
    ][action]


def make_champion_policy(model) -> PolicyFn:
    """Wrap a fitted champion model's predict_proba as a per-row greedy policy."""

    def policy(row: pd.Series, step: int, max_steps: int) -> int:
        x_df = row.reindex(model.feature_names).to_frame().T
        x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        probs = model.predict_proba(x_df)
        return int(np.argmax(probs[0]))

    return policy


def make_reactive_tbo_policy(cycle_limit: float = 180.0) -> PolicyFn:
    """Time-Between-Overhaul baseline: run until a fixed cycle count, then shut down.

    The reactive/scheduled-maintenance status quo this system is meant to
    improve on -- no health signal used at all.
    """

    def policy(row: pd.Series, step: int, max_steps: int) -> int:
        cycle = float(row.get("cycle", step))
        return 4 if cycle > cycle_limit else 0

    return policy


def make_rule_cascade_policy(cfg: SupervisorConfig | None = None) -> PolicyFn:
    """The same threshold cascade safety_policy.py enforces as a hard override,
    run here as a standalone policy -- i.e. "no learning at all, just the
    guardrail thresholds". This is the fairest non-RL comparison point: it
    uses the same health signals PPO does, just without any learned judgment.
    """
    cfg = cfg or _CFG

    def policy(row: pd.Series, step: int, max_steps: int) -> int:
        rul = float(row.get("predicted_rul", 999.0))
        r10 = float(row.get("risk_10", 0.0))
        r30 = float(row.get("risk_30", 0.0))
        r50 = float(row.get("risk_50", 0.0))

        if rul <= cfg.immediate_rul_threshold or r10 >= cfg.critical_risk_10_threshold:
            return 4
        if rul <= cfg.maintenance_rul_threshold or r30 >= cfg.maintenance_risk_30_threshold:
            return 3
        if rul <= cfg.inspection_rul_threshold or r30 >= cfg.inspection_risk_30_threshold:
            return 2
        if rul <= cfg.monitoring_rul_threshold or r50 >= cfg.monitoring_risk_50_threshold:
            return 1
        return 0

    return policy


def rollout_policy(
    env: TurbofanMaintenanceEnv, engine_ids: list, policy_fn: PolicyFn,
) -> list[dict]:
    """Run `policy_fn` through `env` for every engine in `engine_ids`, one episode each."""
    results: list[dict] = []
    for engine_id in engine_ids:
        env.reset(engine_id=engine_id)
        done = False
        realized_cost = 0.0
        last_event = "max_steps_reached"

        while not done:
            row = env.current_trajectory.iloc[env.current_step]
            action = policy_fn(row, env.current_step, env.max_steps)
            realized_cost += _cost_for_action(env.cfg, action)
            _, _, done, info = env.step(action)
            last_event = info.get("event", last_event)

        is_catastrophic = last_event in _CATASTROPHIC_EVENTS
        if is_catastrophic:
            realized_cost += env.cfg.cost_unplanned_failure

        results.append({
            "engine_id": engine_id,
            "terminal_event": last_event,
            "catastrophic": is_catastrophic,
            "realized_cost_usd": realized_cost,
            "downtime_hours": DOWNTIME_HOURS_BY_EVENT.get(last_event, 0.0),
        })
    return results


def summarize_rollout(results: list[dict]) -> dict:
    n = max(len(results), 1)
    catastrophic_count = sum(1 for r in results if r["catastrophic"])
    return {
        "engines_evaluated": len(results),
        "failure_prevention_rate": round(1.0 - catastrophic_count / n, 4),
        "catastrophic_failures": catastrophic_count,
        "mean_maintenance_cost_usd": round(sum(r["realized_cost_usd"] for r in results) / n, 2),
        "total_maintenance_cost_usd": round(sum(r["realized_cost_usd"] for r in results), 2),
        "mean_downtime_hours": round(sum(r["downtime_hours"] for r in results) / n, 3),
        "total_downtime_hours": round(sum(r["downtime_hours"] for r in results), 2),
        "terminal_event_distribution": pd.Series(
            [r["terminal_event"] for r in results]
        ).value_counts().to_dict(),
    }


def _pct_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return round((baseline - candidate) / baseline * 100.0, 2)


def run_policy_evaluation(config: SupervisorConfig | None = None) -> dict:
    """Evaluate the champion against the reactive and rule-cascade baselines
    on held-out test engines, and report FR7's three metrics explicitly."""
    cfg = config or _CFG

    section("MAINTENANCE DECISION-OPTIMIZATION EVALUATOR STARTED")
    start_time = time.perf_counter()

    if not _TEST_SPLIT_PATH.exists() or not _RAW_ENCODED_PATH.exists():
        return {"status": "error", "message": "Required processed dataset(s) not found."}

    test_df = _load_raw_test_engines()
    env = TurbofanMaintenanceEnv(test_df, config=cfg)

    if not _CHAMPION_MANIFEST.exists():
        return {"status": "error", "message": "No champion_model_manifest.json found."}
    with open(_CHAMPION_MANIFEST, "r", encoding="utf-8") as f:
        champion_manifest = json.load(f)
    champion_name = champion_manifest.get("champion_model_name", "unknown")
    # The manifest's stored path is absolute and baked in at training time --
    # invalid on any other machine or after this project directory is
    # moved/renamed (see decision_fusion_engine.py's
    # _resolve_champion_model_path). Fall back to re-deriving it from this
    # machine's own ARTIFACT_ROOT before failing.
    champion_model_path = champion_manifest.get("champion_model_path")
    if not (champion_model_path and Path(champion_model_path).exists()):
        fallback_path = ARTIFACT_ROOT / "models" / f"{champion_name}.joblib"
        if fallback_path.exists():
            champion_model_path = fallback_path
    champion_model = load_model(champion_model_path)

    policies: dict[str, PolicyFn] = {
        "champion": make_champion_policy(champion_model),
        "reactive_tbo_baseline": make_reactive_tbo_policy(),
        "rule_cascade_baseline": make_rule_cascade_policy(cfg),
    }

    summaries: dict[str, dict] = {}
    for name, policy_fn in policies.items():
        results = rollout_policy(env, env.engine_ids, policy_fn)
        summaries[name] = summarize_rollout(results)
        logger.info(
            "%s: failure_prevention_rate=%.3f, mean_cost=$%.0f, mean_downtime=%.1fh",
            name, summaries[name]["failure_prevention_rate"],
            summaries[name]["mean_maintenance_cost_usd"], summaries[name]["mean_downtime_hours"],
        )

    champion = summaries["champion"]
    reactive = summaries["reactive_tbo_baseline"]
    cascade = summaries["rule_cascade_baseline"]

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "champion_model_name": champion_name,
        "test_engines_evaluated": len(env.engine_ids),
        # The three Proposal Report S3.6 metrics, explicitly, champion vs.
        # the reactive/TBO status quo this system is meant to improve on.
        "maintenance_cost_reduction_pct_vs_reactive": _pct_reduction(
            reactive["mean_maintenance_cost_usd"], champion["mean_maintenance_cost_usd"],
        ),
        "system_downtime_reduction_pct_vs_reactive": _pct_reduction(
            reactive["mean_downtime_hours"], champion["mean_downtime_hours"],
        ),
        "failure_prevention_rate": champion["failure_prevention_rate"],
        # Same three, also vs. the rule-cascade (no-learning) baseline, since
        # that is the more informative comparison for "did learning help".
        "maintenance_cost_reduction_pct_vs_rule_cascade": _pct_reduction(
            cascade["mean_maintenance_cost_usd"], champion["mean_maintenance_cost_usd"],
        ),
        "system_downtime_reduction_pct_vs_rule_cascade": _pct_reduction(
            cascade["mean_downtime_hours"], champion["mean_downtime_hours"],
        ),
        "policies": summaries,
        "downtime_hours_assumption": DOWNTIME_HOURS_BY_EVENT,
        "action_cost_usd": {
            DECISION_CLASSES[i]: _cost_for_action(cfg, i) for i in range(len(DECISION_CLASSES))
        },
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info(
        "Champion vs reactive baseline: cost %.1f%% lower, downtime %.1f%% lower, "
        "failure_prevention_rate=%.3f",
        report["maintenance_cost_reduction_pct_vs_reactive"],
        report["system_downtime_reduction_pct_vs_reactive"],
        report["failure_prevention_rate"],
    )
    logger.info("Report written to: %s", _REPORT_PATH)
    section("MAINTENANCE DECISION-OPTIMIZATION EVALUATOR COMPLETED")

    return report


def main() -> int:
    try:
        res = run_policy_evaluation()
        print(json.dumps(res, indent=2))
        return 0 if res.get("status") == "success" else 1
    except Exception as exc:
        logger.error("Error in policy_evaluator: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
