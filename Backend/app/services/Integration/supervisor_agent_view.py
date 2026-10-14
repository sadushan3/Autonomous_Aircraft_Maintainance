"""
Read layer that assembles the Autonomous Maintenance Supervisor Agent's
operator view from output the three upstream agents have already produced.

Where the numbers come from
---------------------------
Nothing here predicts, scores or decides. Every figure is either read straight
out of an artefact the pipeline wrote, or is a labelled arithmetic reduction of
one:

* ``data/exchange/supervisor_decisions.csv`` -- the fused table produced by
  ``supervisor_bridge`` (RUL + Failure Risk + Anomaly & Health joined, then run
  through the Supervisor's own ``DecisionFusionEngine``). This supplies every
  agent output, the final decision, the safety-policy override record and the
  confidence/uncertainty/conflict scores.
* ``processed/raw_data.csv`` -- the C-MAPSS readings themselves, so a component
  card can show a real sensor value with its real label and unit rather than a
  decorative one.
* ``processed/residuals.csv`` -- the digital twin's per-sensor robust z-scores,
  rolled up to engine subsystems using the Anomaly component's own fixed
  sensor-to-subsystem map. This is what makes the component heat map real.

Two honesty constraints are enforced rather than papered over:

1. Sensors 1, 5, 18 and 19 are constant across the whole C-MAPSS corpus, so
   their residual z is identically zero and carries no health signal. Sensors
   10 and 16 are quantised near-constants whose z inflates to absurd magnitudes
   (|z| up to 2714 for sensor 16) as a property of the data, not of the engine.
   All six are excluded from component health, and any subsystem left with no
   informative sensor reports ``health_pct: null`` with a stated reason instead
   of a made-up percentage.
2. The upstream risk model's horizons are *cycles* (10/30/50), not hours. They
   are labelled as cycles here.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import get_config as get_ahm_config
from app.config.Maintenance_Supervisor.supervisor_config import (
    ACTION_MAPPING,
    ARTIFACT_ROOT,
    DECISION_CLASSES,
    DECISION_TO_SEVERITY,
    PRIORITY_MAPPING,
    URGENCY_MAPPING,
    SupervisorConfig,
)
from app.services.Anomaly_Health_Monitoring.reasoning.rule_engine import (
    _SUBSYSTEM_CAUSE_TEMPLATES,
)
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import (
    SENSOR_SUBSYSTEMS,
    SUBSYSTEM_DESCRIPTIONS,
)
from app.services.Maintenance_Supervisor.decision_fusion.counterfactual_optimizer import (
    CounterfactualOptimizer,
)
from app.services.Maintenance_Supervisor.reporting.cost_avoided import estimate_cost_avoided
from app.utils.Anomaly_Health_Monitoring.utils import get_logger

logger = get_logger(__name__)

_CFG = SupervisorConfig()
_ahm_config = get_ahm_config()

DECISIONS_CSV = _ahm_config.paths.data_exchange / "supervisor_decisions.csv"
RAW_CSV = Path(_ahm_config.paths.processed_path(_ahm_config.paths.raw_data_csv))
RESIDUALS_CSV = Path(_ahm_config.paths.processed_path(_ahm_config.paths.residuals_csv))
CHAMPION_MANIFEST = ARTIFACT_ROOT / "champion_model_manifest.json"
CONFORMAL_MANIFEST = ARTIFACT_ROOT / "conformal_rul_bounds.json"
FEEDBACK_STORE = ARTIFACT_ROOT / "feedback_store.jsonl"


# ---------------------------------------------------------------------------
# Sensor metadata
# ---------------------------------------------------------------------------

#: The C-MAPSS sensor dictionary, as published with the dataset: symbol,
#: human description and physical unit. Used so a component card can label the
#: reading it shows instead of printing a bare number.
SENSOR_META: Dict[str, Tuple[str, str, str]] = {
    "sensor_1": ("T2", "Total temperature at fan inlet", "°R"),
    "sensor_2": ("T24", "Total temperature at LPC outlet", "°R"),
    "sensor_3": ("T30", "Total temperature at HPC outlet", "°R"),
    "sensor_4": ("T50", "Total temperature at LPT outlet", "°R"),
    "sensor_5": ("P2", "Pressure at fan inlet", "psia"),
    "sensor_6": ("P15", "Total pressure in bypass duct", "psia"),
    "sensor_7": ("P30", "Total pressure at HPC outlet", "psia"),
    "sensor_8": ("Nf", "Physical fan speed", "rpm"),
    "sensor_9": ("Nc", "Physical core speed", "rpm"),
    "sensor_10": ("epr", "Engine pressure ratio (P50/P2)", "—"),
    "sensor_11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "sensor_12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "sensor_13": ("NRf", "Corrected fan speed", "rpm"),
    "sensor_14": ("NRc", "Corrected core speed", "rpm"),
    "sensor_15": ("BPR", "Bypass ratio", "—"),
    "sensor_16": ("farB", "Burner fuel-air ratio", "—"),
    "sensor_17": ("htBleed", "Bleed enthalpy", "—"),
    "sensor_18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "sensor_19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "sensor_20": ("W31", "HPT coolant bleed", "lbm/s"),
    "sensor_21": ("W32", "LPT coolant bleed", "lbm/s"),
}

#: Sensors whose residual z carries no usable health signal.
#:
#: 1/5/18/19 are exactly constant in C-MAPSS, so the twin predicts them
#: perfectly and their z is identically 0. 10 and 16 are quantised
#: near-constants: their z distribution has a long artificial tail (|z| reaches
#: 2714 on sensor 16) that reflects the quantisation, not engine condition.
#: Including any of them would either flatter or wreck a subsystem's health for
#: reasons that have nothing to do with the engine.
NON_INFORMATIVE_SENSORS = frozenset(
    {"sensor_1", "sensor_5", "sensor_18", "sensor_19", "sensor_10", "sensor_16"}
)

#: Gas-path order, front to back, then the two ducting systems. Short labels
#: match the abbreviations used on the engine cutaway.
COMPONENT_ORDER: List[Tuple[str, str, str]] = [
    ("fan", "FAN", "Fan Inlet"),
    ("lpc", "LPC", "Low Press. Compressor"),
    ("hpc", "HPC", "High Press. Compressor"),
    ("burner", "CC", "Combustor Chamber"),
    ("hpt", "HPT", "High Press. Turbine"),
    ("lpt", "LPT", "Low Press. Turbine"),
    ("bypass", "BYP", "Bypass Duct"),
    ("bleed", "BLD", "Bleed Air System"),
]

#: |z| at which a subsystem is treated as fully degraded for the 0-100 health
#: scale. Anchored on the Anomaly component's own status thresholds
#: (``subsystem_explainer``: elevated at 1.5, degraded at 3.0), extended to 6.0
#: so "degraded" lands at 50% rather than at 0% and there is headroom above it.
HEALTH_Z_FULL_SCALE = 6.0


def _component_sensors(subsystem: str) -> List[str]:
    """Sensors the dependency graph assigns to one subsystem, in index order."""
    members = [s for s, sub in SENSOR_SUBSYSTEMS.items() if sub == subsystem]
    return sorted(members, key=lambda s: int(s.split("_")[1]))


#: Root-cause phrase -> subsystem id, inverted from the reasoning layer's own
#: templates so the two never drift apart. The Anomaly agent writes the phrase
#: into ``root_cause``; this turns it back into the section to highlight.
CAUSE_TO_SUBSYSTEM: Dict[str, str] = {
    template[0].strip().lower(): subsystem
    for subsystem, template in _SUBSYSTEM_CAUSE_TEMPLATES.items()
}

#: Subsystem -> (root-cause phrase, what a technician inspects, suggested action).
COMPONENT_GUIDANCE: Dict[str, Tuple[str, str, str]] = dict(_SUBSYSTEM_CAUSE_TEMPLATES)


def _subsystem_from_cause(root_cause: Optional[str]) -> Optional[str]:
    if not root_cause:
        return None
    return CAUSE_TO_SUBSYSTEM.get(str(root_cause).strip().lower())


# ---------------------------------------------------------------------------
# Cached table loads
# ---------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}


def _cached_read(key: str, path: Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read a CSV once per on-disk version.

    The residual table is 99 MB and the raw table 45 MB; a dashboard that
    re-read either on every poll would spend seconds per request re-parsing
    files that only change when the pipeline re-runs. Keying the cache on the
    file's mtime means a fresh pipeline run is picked up automatically without
    a restart.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    stamp = path.stat().st_mtime
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    started = time.perf_counter()
    frame = pd.read_csv(path, low_memory=False, **read_kwargs)
    logger.info(
        "supervisor agent view: loaded %s (%d rows) in %.2fs",
        path.name,
        len(frame),
        time.perf_counter() - started,
    )
    _CACHE[key] = (stamp, frame)
    return frame


def load_decisions() -> pd.DataFrame:
    return _cached_read("decisions", DECISIONS_CSV)


def load_raw() -> pd.DataFrame:
    header = pd.read_csv(RAW_CSV, nrows=0)
    wanted = ["unit", "cycle", "subset_id", "op_setting_1", "op_setting_2", "op_setting_3"]
    cols = [c for c in wanted if c in header.columns]
    cols += [s for s in SENSOR_META if s in header.columns]
    return _cached_read("raw", RAW_CSV, usecols=cols)


def load_residual_z() -> pd.DataFrame:
    header = pd.read_csv(RESIDUALS_CSV, nrows=0)
    cols = ["unit", "cycle"] + [
        f"residual_z_{s}" for s in SENSOR_META if f"residual_z_{s}" in header.columns
    ]
    return _cached_read("residual_z", RESIDUALS_CSV, usecols=cols)


def decisions_written_at() -> Optional[str]:
    if not DECISIONS_CSV.is_file():
        return None
    return datetime.fromtimestamp(
        DECISIONS_CSV.stat().st_mtime, tz=timezone.utc
    ).isoformat()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """Coerce to a JSON-safe float, mapping NaN/inf/blank to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def _flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def _pretty(decision: Optional[str]) -> Optional[str]:
    if not decision:
        return None
    return decision.replace("_", " ").title()


def engine_rows(engine_id: int | str) -> pd.DataFrame:
    frame = load_decisions()
    rows = frame[frame["engine_id"].astype(str) == str(engine_id)]
    if rows.empty:
        raise KeyError(f"Engine {engine_id} has no fused decisions.")
    return rows.sort_values("cycle").reset_index(drop=True)


def pick_cycle(rows: pd.DataFrame, cycle: Optional[int]) -> pd.Series:
    """The requested cycle, or the engine's latest if unspecified/out of range."""
    if cycle is None:
        return rows.iloc[-1]
    exact = rows[rows["cycle"].astype(int) == int(cycle)]
    if not exact.empty:
        return exact.iloc[0]
    # Nearest cycle at or below the request, so a stale slider position still
    # resolves to a real observation instead of erroring.
    below = rows[rows["cycle"].astype(int) <= int(cycle)]
    return below.iloc[-1] if not below.empty else rows.iloc[0]


# ---------------------------------------------------------------------------
# Policy / provenance
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _feedback_count() -> int:
    if not FEEDBACK_STORE.is_file():
        return 0
    with FEEDBACK_STORE.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def policy_state() -> Dict[str, Any]:
    """What the decision head actually is, described as it actually trains.

    The champion is whatever ``champion_model_manifest.json`` names. PPO,
    when it is champion, is trained by genuine on-policy rollout collection
    against ``TurbofanMaintenanceEnv`` (GAE + a clipped surrogate objective +
    entropy bonus -- see ``models/ppo_supervisor.py``), never against the
    rule-derived ``final_decision`` label; every other candidate (including
    DQN) fits a static (X, y) matrix by cross-entropy against that label.
    Distinguishing the two here, rather than describing every candidate the
    same way, is the point of this card: a viewer needs to know which one
    they are looking at.
    """
    manifest = _read_json(CHAMPION_MANIFEST)
    champion = str(manifest.get("champion_model_name") or "unknown")
    is_rl = champion == "ppo" and manifest.get("episode_metrics") is not None
    regimes = {
        "ppo": (
            "actor-critic network, trained by on-policy PPO (rollout collection, GAE, "
            "clipped surrogate objective, entropy bonus) against TurbofanMaintenanceEnv"
            if is_rl
            else "actor-critic network, fitted by behaviour cloning (no rollouts, no advantage)"
        ),
        "dqn": "Q-network, fitted by supervised regression on labelled actions",
        "random_forest": "supervised ensemble classifier",
        "xgboost": "supervised gradient-boosted classifier",
        "lightgbm": "supervised gradient-boosted classifier",
        "logistic_regression": "supervised linear classifier",
    }
    target_note = (
        (
            "This is the RL policy: it was never trained to reproduce final_decision, "
            "so its label-agreement val_macro_f1 is intentionally low. The metric that "
            "matters for it is episode_metrics (mean return, catastrophic-failure rate, "
            "successful-maintenance rate) below, from rolling the policy out over "
            "held-out engine trajectories."
        )
        if is_rl
        else (
            "The training label is a deterministic rule over RUL, risk and anomaly "
            "severity, so near-perfect validation accuracy reflects reproducing a "
            "known policy, not independent predictive skill."
        )
    )
    feedback_entries = _feedback_count()
    return {
        "champion_model": champion,
        "champion_label": {"ppo": "PPO", "dqn": "DQN"}.get(champion, champion.replace("_", " ").title()),
        "training_regime": regimes.get(champion, "supervised classifier"),
        "is_reinforcement_learning": is_rl,
        "target_note": target_note,
        "val_macro_f1": _num(manifest.get("val_macro_f1")),
        "episode_metrics": manifest.get("episode_metrics"),
        "candidates": sorted(
            str(c.get("name")) for c in (manifest.get("all_candidates") or []) if c.get("name")
        ),
        "decision_classes": list(DECISION_CLASSES),
        "feedback_entries": feedback_entries,
        "continuous_learning_active": feedback_entries > 0,
        "safety_thresholds": {
            "immediate_rul": _CFG.immediate_rul_threshold,
            "maintenance_rul": _CFG.maintenance_rul_threshold,
            "inspection_rul": _CFG.inspection_rul_threshold,
            "monitoring_rul": _CFG.monitoring_rul_threshold,
            "critical_risk_10": _CFG.critical_risk_10_threshold,
            "maintenance_risk_30": _CFG.maintenance_risk_30_threshold,
            "inspection_risk_30": _CFG.inspection_risk_30_threshold,
            "monitoring_risk_50": _CFG.monitoring_risk_50_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Per-agent blocks
# ---------------------------------------------------------------------------

def _agent_status(row: pd.Series) -> Dict[str, Any]:
    """Whether each upstream agent actually contributed to this row.

    "Active" here means the agent's own columns carry values on this fused
    record -- not that a process answered a ping. An agent whose columns are
    all null did not reach the fusion step for this engine.
    """

    def alive(columns: List[str]) -> Tuple[bool, List[str]]:
        present = [c for c in columns if c in row.index and _num(row.get(c)) is not None]
        return bool(present), present

    rul_cols = ["predicted_rul", "lower_bound", "upper_bound", "degradation_speed"]
    risk_cols = ["risk_10", "risk_30", "risk_50", "control_risk"]
    anomaly_cols = ["anomaly_score", "health_index", "early_warning_score"]

    rul_ok, rul_present = alive(rul_cols)
    risk_ok, risk_present = alive(risk_cols)
    anom_ok, anom_present = alive(anomaly_cols)

    return {
        "rul": {
            "key": "rul",
            "name": "RUL Prognostics Agent",
            "status": "active" if rul_ok else "no_signal",
            "fields_present": rul_present,
            "model_used": _text(row.get("model_used")),
        },
        "failure_risk": {
            "key": "failure_risk",
            "name": "Failure Risk Agent",
            "status": "active" if risk_ok else "no_signal",
            "fields_present": risk_present,
            "model_used": None,
        },
        "anomaly": {
            "key": "anomaly",
            "name": "Explainable Anomaly Agent",
            "status": "active" if anom_ok else "no_signal",
            "fields_present": anom_present,
            "model_used": None,
        },
    }


def _risk_block(row: pd.Series) -> Dict[str, Any]:
    """Multi-horizon failure risk, with the horizons named in the unit they use.

    The upstream model's horizons are cycles (10 / 30 / 50), not hours. The
    30-cycle horizon is used as the headline because it is the one the safety
    policy's ``maintenance_risk_30_threshold`` rule reads.
    """
    horizons = []
    for key, label in (("10", "10 cycles"), ("30", "30 cycles"), ("50", "50 cycles")):
        horizons.append(
            {
                "key": f"risk_{key}",
                "label": label,
                "cycles": int(key),
                "value": _num(row.get(f"risk_{key}")),
                "uncertainty": _num(row.get(f"uncertainty_{key}")),
                "threshold": _num(row.get(f"threshold_{key}")),
                "predicted_failure": _flag(row.get(f"predicted_fail_in_{key}")),
            }
        )
    headline = next(h for h in horizons if h["key"] == "risk_30")
    return {
        "headline": headline,
        "horizons": horizons,
        "risk_state": _text(row.get("risk_state")),
        "control_risk": _num(row.get("control_risk")),
        "risk_velocity": _num(row.get("risk_velocity")),
        "risk_acceleration": _num(row.get("risk_acceleration")),
        "horizon_unit": "cycles",
    }


def _rul_block(row: pd.Series) -> Dict[str, Any]:
    """Predicted RUL with the widest defensible interval available.

    Preference order is deliberate: the inductive conformal margin, when the
    Supervisor has actually calibrated one, carries a distribution-free
    coverage guarantee that the RUL agent's own interval does not. Which one
    was used is reported so the figure can never be read as a stronger claim
    than it is.
    """
    predicted = _num(row.get("predicted_rul"))
    lower = _num(row.get("lower_bound"))
    upper = _num(row.get("upper_bound"))

    conformal = _read_json(CONFORMAL_MANIFEST)
    q_alpha = _num(conformal.get("margin_q_alpha") or conformal.get("q_alpha"))
    coverage = _num(conformal.get("target_coverage"))

    if predicted is not None and q_alpha is not None:
        interval = {
            "lower": round(max(0.0, predicted - q_alpha), 2),
            "upper": round(predicted + q_alpha, 2),
            "margin": round(q_alpha, 2),
            "source": "inductive_conformal",
            "target_coverage": coverage or 0.95,
            "label": "Inductive Conformal Predictor",
        }
    else:
        interval = {
            "lower": lower,
            "upper": upper,
            "margin": _num(row.get("uncertainty_width")),
            "source": "rul_agent_interval",
            "target_coverage": None,
            "label": "RUL agent prediction interval (no conformal calibration on disk)",
        }

    return {
        "predicted_rul": predicted,
        "interval": interval,
        "lifecycle_state": _text(row.get("lifecycle_state")),
        "degradation_speed": _num(row.get("degradation_speed")),
        "degradation_acceleration": _num(row.get("degradation_acceleration")),
        "model_used": _text(row.get("model_used")),
        "fallback_triggered": _flag(row.get("fallback_triggered")),
    }


def _focus_component(components: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The section to open the heat map on.

    Preference order: the worst measured section, then whichever section the
    reasoning layer named as root cause. The second case matters whenever the
    processed residual table cannot be aligned to these decisions -- the
    reasoning layer's verdict travels inside the fused row itself, so it is
    available even when the twin residuals are not.
    """
    measured = [c for c in components if c.get("health_pct") is not None]
    if measured:
        return min(measured, key=lambda c: c["health_pct"])
    return next((c for c in components if c.get("is_root_cause")), None)


def _anomaly_block(row: pd.Series, components: List[Dict[str, Any]]) -> Dict[str, Any]:
    focus = _focus_component(components)
    if focus is not None:
        focus = {
            "id": focus["id"],
            "short_label": focus["short_label"],
            "name": focus["name"],
            "mean_abs_z": focus.get("mean_abs_z"),
            "health_pct": focus.get("health_pct"),
            "top_sensor": focus.get("top_sensor"),
            "top_sensor_label": focus.get("top_sensor_label"),
            "is_root_cause": focus.get("is_root_cause"),
            "inspection_focus": focus.get("inspection_focus"),
            "basis": "twin residual" if focus.get("health_pct") is not None else "reasoning root cause",
        }
    return {
        "anomaly_score": _num(row.get("anomaly_score")),
        "is_anomalous": _flag(row.get("is_anomalous")),
        "severity": _text(row.get("anomaly_severity")) or "none",
        "anomaly_type": _text(row.get("anomaly_type")),
        "root_cause": _text(row.get("root_cause")),
        "health_index": _num(row.get("health_index")),
        "health_state": _text(row.get("health_state")),
        "health_trend": _num(row.get("health_trend")),
        "remaining_health_pct": _num(row.get("remaining_health_pct")),
        "early_warning_score": _num(row.get("early_warning_score")),
        "uncertainty_score": _num(row.get("uncertainty_score")),
        "focus_component": focus,
    }


def _recommendation_block(row: pd.Series) -> Dict[str, Any]:
    decision = _text(row.get("final_decision")) or "continue_operation"
    severity = DECISION_TO_SEVERITY.get(decision, 0)
    try:
        cost = estimate_cost_avoided(
            decision,
            risk_10=_num(row.get("risk_10")),
            risk_30=_num(row.get("risk_30")),
            predicted_rul=_num(row.get("predicted_rul")),
        )
    except ValueError:
        cost = None
    return {
        "final_decision": decision,
        "final_decision_label": _pretty(decision),
        "action_code": severity,
        "severity": severity,
        "recommended_action": _text(row.get("recommended_action")) or ACTION_MAPPING.get(decision),
        "priority": _text(row.get("priority")) or PRIORITY_MAPPING.get(decision),
        "urgency": _text(row.get("maintenance_urgency")) or URGENCY_MAPPING.get(decision),
        "urgency_label": _pretty(
            _text(row.get("maintenance_urgency")) or URGENCY_MAPPING.get(decision)
        ),
        "confidence": _num(row.get("confidence_score")),
        "confidence_level": _text(row.get("confidence_level")),
        "entropy_score": _num(row.get("entropy_score")),
        "conflict_score": _num(row.get("conflict_score")),
        "raw_model_decision": _text(row.get("raw_model_decision")),
        "raw_model_decision_label": _pretty(_text(row.get("raw_model_decision"))),
        "override_applied": _flag(row.get("override_applied")),
        "override_reason": _text(row.get("override_reason")),
        "requires_human_review": _flag(row.get("requires_human_review")),
        "review_reason": _text(row.get("review_reason")),
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# Component heat map
# ---------------------------------------------------------------------------

#: Per-subset engine-id offset applied when C-MAPSS subsets are pooled into one
#: processed table (``unit_id_offset_step`` in the Anomaly component's config:
#: FD001 stays 1.., FD002 becomes 10001.., and so on). A run over a single
#: uploaded subset keeps the file's own 1..N numbering instead, so both
#: candidates have to be tried.
_SUBSET_OFFSETS: Dict[str, int] = {"FD001": 0, "FD002": 10_000, "FD003": 20_000, "FD004": 30_000}


def resolve_unit(engine_id: int | str, fd_subset: Optional[str], cycle_span: Tuple[int, int]) -> Dict[str, Any]:
    """Find the ``unit`` id in the processed tables for a decision's ``engine_id``.

    The two do not always coincide. A pooled preprocessing run offsets engine
    ids per subset, while a run over one uploaded subset keeps that file's own
    numbering -- and the fused decisions carry whichever the run that produced
    them used.

    Matching on id alone is not enough. ``processed/raw_data.csv`` and
    ``processed/residuals.csv`` are overwritten by *whatever* pipeline run
    happened last, which may have been over a different split entirely: on this
    machine the decisions came from an uploaded ``test_FD002.txt`` while the
    processed tables were later rebuilt from the pooled *train* corpus, so unit
    10227 and engine 227 are simply different engines. Silently joining them
    would put another engine's sensor readings on this engine's card.

    So a candidate is accepted only if its recorded cycle span matches the
    decision's *exactly*. Containment is not enough: unit 10227 spans cycles
    1-184 and engine 227 spans 1-180, so a "covers it" test would happily pair
    two different engines. Exact equality is a cheap identity check on the
    trajectory, and when nothing matches the caller is told the heat map is
    unavailable and why, rather than being handed plausible wrong numbers.
    """
    try:
        raw = load_raw()
    except FileNotFoundError:
        return {"unit": None, "aligned": False, "reason": "processed/raw_data.csv is not present."}

    spans = raw.groupby("unit")["cycle"].agg(["min", "max"])
    subset = str(fd_subset or "").upper()
    offset = _SUBSET_OFFSETS.get(subset, 0)
    candidates = [int(engine_id)] + ([int(engine_id) + offset] if offset else [])

    by_unit_subset = (
        raw.groupby("unit")["subset_id"].first() if "subset_id" in raw.columns else None
    )

    for candidate in candidates:
        if candidate not in spans.index:
            continue
        # Only reject on a subset that is present and genuinely different. A
        # run over one uploaded file need not stamp subset_id at all, and a
        # blank there is an absence of evidence, not a mismatch.
        if by_unit_subset is not None and subset:
            stored = str(by_unit_subset.get(candidate, "") or "").strip().upper()
            if stored and stored not in {"NAN", "NONE"} and stored != subset:
                continue
        low, high = int(spans.loc[candidate, "min"]), int(spans.loc[candidate, "max"])
        if (low, high) == (int(cycle_span[0]), int(cycle_span[1])):
            return {"unit": candidate, "aligned": True, "reason": None}

    return {
        "unit": None,
        "aligned": False,
        "reason": (
            "The processed sensor and residual tables on disk were written by a "
            "different pipeline run than these decisions -- no unit id covers this "
            f"engine's cycle range ({cycle_span[0]}-{cycle_span[1]}). Re-run "
            "/integration/run-full over the same file to realign them."
        ),
    }


def component_heatmap(
    engine_id: int | str,
    cycle: int,
    fd_subset: Optional[str] = None,
    cycle_span: Optional[Tuple[int, int]] = None,
    root_cause: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-subsystem condition at one cycle, from twin residuals and raw readings.

    ``health_pct`` is a linear map of the subsystem's mean |residual z| onto
    0-100, saturating at :data:`HEALTH_Z_FULL_SCALE`. It is a twin-deviation
    score, not a manufacturer health index: 100 means the twin predicted every
    informative sensor in that section correctly at this cycle.

    Subsystems whose only sensors are in :data:`NON_INFORMATIVE_SENSORS` report
    ``health_pct: null`` and say why, rather than inventing a figure. If the
    processed tables cannot be aligned to these decisions at all (see
    :func:`resolve_unit`), every card still names its section, its sensors, its
    inspection focus and whether the reasoning layer blamed it -- only the
    measured values are withheld.
    """
    span = cycle_span or (int(cycle), int(cycle))
    resolution = resolve_unit(engine_id, fd_subset, span)
    unit = resolution["unit"]

    res_row = None
    raw_row = None
    if unit is not None:
        try:
            residuals = load_residual_z()
            res_rows = residuals[
                (residuals["unit"].astype(int) == unit)
                & (residuals["cycle"].astype(int) == int(cycle))
            ]
            res_row = res_rows.iloc[0] if not res_rows.empty else None
        except FileNotFoundError:
            res_row = None

        raw = load_raw()
        raw_rows = raw[
            (raw["unit"].astype(int) == unit) & (raw["cycle"].astype(int) == int(cycle))
        ]
        raw_row = raw_rows.iloc[0] if not raw_rows.empty else None

    blamed = _subsystem_from_cause(root_cause)

    # Total informative deviation, so each subsystem's contribution share sums
    # to 1 across the sections that actually carry signal.
    per_subsystem: Dict[str, Dict[str, Any]] = {}
    grand_total = 0.0
    for subsystem, _short, _name in COMPONENT_ORDER:
        members = _component_sensors(subsystem)
        informative = [s for s in members if s not in NON_INFORMATIVE_SENSORS]
        z_by_sensor: Dict[str, float] = {}
        if res_row is not None:
            for sensor in informative:
                value = _num(res_row.get(f"residual_z_{sensor}"))
                if value is not None:
                    z_by_sensor[sensor] = abs(value)
        total = sum(z_by_sensor.values())
        grand_total += total
        per_subsystem[subsystem] = {
            "members": members,
            "informative": informative,
            "z_by_sensor": z_by_sensor,
            "total_abs_z": total,
        }

    out: List[Dict[str, Any]] = []
    for subsystem, short_label, name in COMPONENT_ORDER:
        state = per_subsystem[subsystem]
        z_by_sensor = state["z_by_sensor"]
        readings = []
        if raw_row is not None:
            for sensor in state["members"]:
                value = _num(raw_row.get(sensor))
                if value is None:
                    continue
                symbol, description, unit_label = SENSOR_META[sensor]
                readings.append(
                    {
                        "sensor": sensor,
                        "symbol": symbol,
                        "description": description,
                        "unit": unit_label,
                        "value": round(value, 4),
                        "residual_z": (
                            round(_num(res_row.get(f"residual_z_{sensor}")) or 0.0, 3)
                            if res_row is not None
                            and _num(res_row.get(f"residual_z_{sensor}")) is not None
                            else None
                        ),
                        "informative": sensor not in NON_INFORMATIVE_SENSORS,
                    }
                )

        if z_by_sensor:
            mean_abs_z = float(np.mean(list(z_by_sensor.values())))
            health = 100.0 * max(0.0, 1.0 - mean_abs_z / HEALTH_Z_FULL_SCALE)
            status = (
                "degraded" if mean_abs_z >= 3.0 else "elevated" if mean_abs_z >= 1.5 else "nominal"
            )
            top_sensor = max(z_by_sensor, key=z_by_sensor.get)
            note = None
        else:
            mean_abs_z = None
            health = None
            status = "no_signal"
            top_sensor = None
            if not state["informative"]:
                note = (
                    "No informative twin residual: every sensor mapped to this section is "
                    "constant or quantised in C-MAPSS, so a health figure here would be "
                    "an artefact of the data rather than of the engine."
                )
            else:
                note = resolution["reason"] or "No residual row for this engine and cycle."

        cause_text, inspection_focus, suggested_action = COMPONENT_GUIDANCE.get(
            subsystem, (None, None, None)
        )

        out.append(
            {
                "id": subsystem,
                "short_label": short_label,
                "name": name,
                "subsystem_label": SUBSYSTEM_DESCRIPTIONS.get(subsystem, subsystem),
                "health_pct": round(health, 1) if health is not None else None,
                "status": status,
                "mean_abs_z": round(mean_abs_z, 3) if mean_abs_z is not None else None,
                "contribution_pct": (
                    round(100.0 * state["total_abs_z"] / grand_total, 2) if grand_total > 0 else None
                ),
                "sensors": state["members"],
                "informative_sensors": state["informative"],
                "top_sensor": top_sensor,
                "top_sensor_label": (
                    f"{SENSOR_META[top_sensor][0]} — {SENSOR_META[top_sensor][1]}"
                    if top_sensor
                    else None
                ),
                "readings": readings,
                "is_root_cause": subsystem == blamed,
                "root_cause_text": cause_text,
                "inspection_focus": inspection_focus,
                "suggested_action": suggested_action,
                "note": note,
            }
        )

    # Worst (lowest health) first among the sections that carry signal, then the
    # section the reasoning layer blamed, then the rest -- so whatever the run
    # can actually say most about is the card the operator lands on.
    out.sort(
        key=lambda c: (
            c["health_pct"] is None,
            c["health_pct"] if c["health_pct"] is not None else 0,
            not c["is_root_cause"],
        )
    )
    return {
        "available": bool(res_row is not None),
        "aligned": resolution["aligned"],
        "unavailable_reason": None if res_row is not None else resolution["reason"],
        "resolved_unit": unit,
        "root_cause_subsystem": blamed,
        "health_scale": {
            "metric": "mean |robust residual z| across the section's informative sensors",
            "full_scale_z": HEALTH_Z_FULL_SCALE,
            "elevated_z": 1.5,
            "degraded_z": 3.0,
            "excluded_sensors": sorted(NON_INFORMATIVE_SENSORS),
            "excluded_reason": (
                "sensors 1/5/18/19 are exactly constant in C-MAPSS and 10/16 are quantised "
                "near-constants whose z inflates as a property of the data"
            ),
        },
        "components": out,
    }


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _upstream_contributions(row: pd.Series) -> Dict[str, float]:
    """Share of the decision pressure coming from each upstream agent.

    Same normalisation the Supervisor's own ``SupervisorExplainer`` uses: RUL
    pressure is ``1 - rul/150`` (nearer failure means more pressure), risk
    pressure is ``risk_30`` directly, anomaly pressure is ``anomaly_score``
    directly; the three are then normalised to sum to 100%.
    """
    rul = _num(row.get("predicted_rul"))
    risk = _num(row.get("risk_30")) or 0.0
    anomaly = _num(row.get("anomaly_score")) or 0.0
    s_rul = max(0.0, 1.0 - ((rul if rul is not None else 150.0) / 150.0))
    total = s_rul + risk + anomaly + 1e-6
    return {
        "rul": round(100.0 * s_rul / total, 2),
        "failure_risk": round(100.0 * risk / total, 2),
        "anomaly": round(100.0 * anomaly / total, 2),
    }


def _policy_drivers(row: pd.Series) -> List[Dict[str, Any]]:
    """Every safety-policy rule, its live value, and whether it fired.

    These are the exact comparisons ``SupervisorSafetyPolicy`` performs, so an
    engineer can see not only which rule triggered the override but which ones
    came close and which stayed clear.
    """
    rul = _num(row.get("predicted_rul"))
    r10 = _num(row.get("risk_10"))
    r30 = _num(row.get("risk_30"))
    severity = (_text(row.get("anomaly_severity")) or "none").lower()

    drivers = [
        {
            "rule": "Critical RUL",
            "expression": f"predicted_rul ≤ {_CFG.immediate_rul_threshold:g}",
            "value": rul,
            "threshold": _CFG.immediate_rul_threshold,
            "fired": rul is not None and rul <= _CFG.immediate_rul_threshold,
            "forces": "immediate_maintenance",
        },
        {
            "rule": "Critical 10-cycle risk",
            "expression": f"risk_10 ≥ {_CFG.critical_risk_10_threshold:g}",
            "value": r10,
            "threshold": _CFG.critical_risk_10_threshold,
            "fired": r10 is not None and r10 >= _CFG.critical_risk_10_threshold,
            "forces": "immediate_maintenance",
        },
        {
            "rule": "Critical anomaly severity",
            "expression": "anomaly_severity == 'critical'",
            "value": severity,
            "threshold": "critical",
            "fired": severity == "critical",
            "forces": "immediate_maintenance",
        },
        {
            "rule": "High 30-cycle risk",
            "expression": f"risk_30 ≥ {_CFG.maintenance_risk_30_threshold:g}",
            "value": r30,
            "threshold": _CFG.maintenance_risk_30_threshold,
            "fired": r30 is not None and r30 >= _CFG.maintenance_risk_30_threshold,
            "forces": "at least schedule_maintenance",
        },
    ]
    return drivers


def _narrative(row: pd.Series, components: List[Dict[str, Any]]) -> str:
    """One paragraph naming the cycle, the three agent readings and the outcome."""
    cycle = int(row.get("cycle", 0))
    rul = _num(row.get("predicted_rul"))
    r30 = _num(row.get("risk_30"))
    anomaly = _num(row.get("anomaly_score"))
    health = _num(row.get("health_index"))
    decision = _text(row.get("final_decision")) or "continue_operation"
    override = _flag(row.get("override_applied"))
    reason = _text(row.get("override_reason"))
    focus = _focus_component(components)

    parts = [f"At cycle {cycle},"]
    if rul is not None:
        parts.append(f"the RUL agent predicts {rul:.1f} cycles of remaining life;")
    if r30 is not None:
        parts.append(f"the failure-risk agent puts 30-cycle failure probability at {r30 * 100:.1f}%;")
    if anomaly is not None:
        health_text = f" against a health index of {health:.1f}" if health is not None else ""
        parts.append(f"the anomaly agent scores {anomaly:.3f}{health_text}.")
    if focus is not None and focus.get("health_pct") is not None:
        parts.append(
            f"Twin residuals localise the largest deviation to the {focus['name']} "
            f"({focus['short_label']}, {focus['health_pct']:.0f}% twin agreement, "
            f"mean |z| {focus['mean_abs_z']:.2f})."
        )
    elif focus is not None:
        parts.append(
            f"The reasoning layer attributes the deviation to the {focus['name']} "
            f"({focus['short_label']}); inspection focus is {focus['inspection_focus']}."
        )
    if override and reason:
        parts.append(
            f"The safety policy overrode the model's candidate to {decision.replace('_', ' ')} "
            f"because {reason}."
        )
    else:
        parts.append(f"Fusion resolves to {decision.replace('_', ' ')}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public views
# ---------------------------------------------------------------------------

def engine_catalog(limit: Optional[int] = None) -> Dict[str, Any]:
    """One entry per engine at its latest cycle, for the unit picker."""
    frame = load_decisions()
    ordered = frame.sort_values(["engine_id", "cycle"])
    latest = ordered.drop_duplicates(subset="engine_id", keep="last")
    counts = frame.groupby("engine_id")["cycle"].agg(["min", "max", "count"])

    engines: List[Dict[str, Any]] = []
    for _, row in latest.iterrows():
        engine_id = row["engine_id"]
        span = counts.loc[engine_id]
        decision = _text(row.get("final_decision")) or "continue_operation"
        engines.append(
            {
                "engine_id": int(engine_id) if str(engine_id).isdigit() else str(engine_id),
                "label": f"ENG-{_text(row.get('fd_subset')) or 'FD'}-{int(engine_id):03d}"
                if str(engine_id).isdigit()
                else str(engine_id),
                "fd_subset": _text(row.get("fd_subset")),
                "latest_cycle": int(row["cycle"]),
                "first_cycle": int(span["min"]),
                "cycles_observed": int(span["count"]),
                "final_decision": decision,
                "severity": DECISION_TO_SEVERITY.get(decision, 0),
                "priority": _text(row.get("priority")),
                "predicted_rul": _num(row.get("predicted_rul")),
                "risk_30": _num(row.get("risk_30")),
                "health_index": _num(row.get("health_index")),
                "anomaly_score": _num(row.get("anomaly_score")),
                "requires_human_review": _flag(row.get("requires_human_review")),
            }
        )

    # Most urgent first: an operator opening this picker wants the engine that
    # needs a decision, not engine 1.
    engines.sort(key=lambda e: (-e["severity"], e["predicted_rul"] if e["predicted_rul"] is not None else 1e9))
    total = len(engines)
    if limit:
        engines = engines[:limit]
    return {
        "total": total,
        "returned": len(engines),
        "engines": engines,
        "decisions_written_at": decisions_written_at(),
    }


def _trajectory(rows: pd.DataFrame, max_points: int = 240) -> List[Dict[str, Any]]:
    """Strided history of the fused signals, for the multi-horizon chart."""
    step = max(1, len(rows) // max_points)
    sampled = rows.iloc[::step]
    if len(rows) and sampled.iloc[-1]["cycle"] != rows.iloc[-1]["cycle"]:
        sampled = pd.concat([sampled, rows.iloc[[-1]]])
    points = []
    for _, row in sampled.iterrows():
        points.append(
            {
                "cycle": int(row["cycle"]),
                "risk_10": _num(row.get("risk_10")),
                "risk_30": _num(row.get("risk_30")),
                "risk_50": _num(row.get("risk_50")),
                "threshold_30": _num(row.get("threshold_30")),
                "predicted_rul": _num(row.get("predicted_rul")),
                "rul_lower": _num(row.get("lower_bound")),
                "rul_upper": _num(row.get("upper_bound")),
                "health_index": _num(row.get("health_index")),
                "anomaly_score": _num(row.get("anomaly_score")),
                "confidence_score": _num(row.get("confidence_score")),
                "final_decision": _text(row.get("final_decision")),
                "severity": DECISION_TO_SEVERITY.get(_text(row.get("final_decision")) or "", 0),
            }
        )
    return points


def _critical_offset(rows: pd.DataFrame, cycle: int) -> Optional[int]:
    """Cycles from ``cycle`` to the first recorded critical decision, if any."""
    ahead = rows[rows["cycle"].astype(int) > int(cycle)]
    if ahead.empty:
        return None
    severities = ahead["final_decision"].astype(str).map(DECISION_TO_SEVERITY).fillna(0)
    critical = ahead[severities >= 4]
    if critical.empty:
        return None
    return int(critical.iloc[0]["cycle"]) - int(cycle)


def agent_view(engine_id: int | str, cycle: Optional[int] = None) -> Dict[str, Any]:
    """The full operator view for one engine at one cycle."""
    rows = engine_rows(engine_id)
    row = pick_cycle(rows, cycle)
    resolved_cycle = int(row["cycle"])

    heatmap = component_heatmap(
        engine_id,
        resolved_cycle,
        fd_subset=_text(row.get("fd_subset")),
        cycle_span=(int(rows["cycle"].min()), int(rows["cycle"].max())),
        root_cause=_text(row.get("root_cause")),
    )
    components = heatmap["components"]
    recommendation = _recommendation_block(row)
    counterfactual = CounterfactualOptimizer.generate_counterfactual(
        predicted_rul=_num(row.get("predicted_rul")),
        risk_10=_num(row.get("risk_10")),
        risk_30=_num(row.get("risk_30")),
        anomaly_severity=_text(row.get("anomaly_severity")),
        current_decision=recommendation["final_decision"],
        focus_component=_focus_component(components),
    ).to_dict()

    return {
        "engine_id": int(engine_id) if str(engine_id).isdigit() else str(engine_id),
        "label": f"ENG-{_text(row.get('fd_subset')) or 'FD'}-{int(engine_id):03d}"
        if str(engine_id).isdigit()
        else str(engine_id),
        "cycle": resolved_cycle,
        "requested_cycle": cycle,
        "fd_subset": _text(row.get("fd_subset")),
        "cycle_range": {
            "min": int(rows["cycle"].min()),
            "max": int(rows["cycle"].max()),
            "observed": int(len(rows)),
        },
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "decisions_csv": str(DECISIONS_CSV),
            "decisions_written_at": decisions_written_at(),
            "basis": "fused Supervisor decision row, as written by the last pipeline run",
        },
        "agents": _agent_status(row),
        "policy": policy_state(),
        "failure_risk": _risk_block(row),
        "rul": _rul_block(row),
        "anomaly": _anomaly_block(row, components),
        "recommendation": recommendation,
        "components": components,
        "component_source": {
            key: heatmap[key]
            for key in ("available", "aligned", "unavailable_reason", "resolved_unit", "root_cause_subsystem", "health_scale")
        },
        "xai": {
            "narrative": _narrative(row, components),
            "upstream_contributions": _upstream_contributions(row),
            "policy_drivers": _policy_drivers(row),
        },
        "counterfactual": counterfactual,
        "trajectory": _trajectory(rows),
        "critical_offset": _critical_offset(rows, resolved_cycle),
    }


#: Cycles of recent history used to measure a signal's per-cycle slope.
EXTRAPOLATION_WINDOW = 40


def _extrapolate(
    rows: pd.DataFrame,
    column: str,
    base_cycle: int,
    steps: int,
    floor: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> Optional[float]:
    """Project ``column`` ``steps`` cycles past ``base_cycle`` along its own recent slope.

    Least-squares slope over the last :data:`EXTRAPOLATION_WINDOW` observed
    cycles, which measures the per-cycle rate directly rather than trusting the
    agents' stride-dependent velocity columns. A single usable point holds the
    value flat instead of guessing a direction.
    """
    if column not in rows.columns:
        return None
    window = rows[rows["cycle"].astype(int) <= int(base_cycle)].tail(EXTRAPOLATION_WINDOW)
    series = pd.to_numeric(window[column], errors="coerce")
    valid = window.loc[series.notna()]
    values = series.dropna()
    if values.empty:
        return None
    last = float(values.iloc[-1])
    if len(values) < 2:
        projected = last
    else:
        cycles = valid["cycle"].astype(float).to_numpy()
        slope = float(np.polyfit(cycles, values.to_numpy(dtype=float), 1)[0])
        projected = last + slope * steps
    if floor is not None:
        projected = max(floor, projected)
    if ceiling is not None:
        projected = min(ceiling, projected)
    return projected


def what_if(engine_id: int | str, cycle: int, cycles_ahead: int) -> Dict[str, Any]:
    """Project the engine's state ``cycles_ahead`` cycles forward.

    Uses the engine's own recorded future rows when the run observed them --
    C-MAPSS training units run to failure, so for most engines this is a real
    observation rather than a forecast. Past the last observed cycle it falls
    back to first-order extrapolation from the RUL agent's ``degradation_speed``
    and the risk agent's ``risk_velocity``, and says so in ``basis``.
    """
    rows = engine_rows(engine_id)
    base = pick_cycle(rows, cycle)
    base_cycle = int(base["cycle"])
    target_cycle = base_cycle + int(cycles_ahead)

    observed = rows[rows["cycle"].astype(int) == target_cycle]
    if not observed.empty:
        row = observed.iloc[0]
        heatmap = component_heatmap(
            engine_id,
            target_cycle,
            fd_subset=_text(row.get("fd_subset")),
            cycle_span=(int(rows["cycle"].min()), int(rows["cycle"].max())),
            root_cause=_text(row.get("root_cause")),
        )
        components = heatmap["components"]
        return {
            "engine_id": int(engine_id) if str(engine_id).isdigit() else str(engine_id),
            "base_cycle": base_cycle,
            "target_cycle": target_cycle,
            "cycles_ahead": int(cycles_ahead),
            "basis": "recorded",
            "basis_note": "This cycle was actually observed and scored in the last run.",
            "failure_risk": _risk_block(row),
            "rul": _rul_block(row),
            "anomaly": _anomaly_block(row, components),
            "recommendation": _recommendation_block(row),
            "components": components,
        }

    # Beyond the observed history: extrapolate each signal along the slope of
    # its own recent trajectory.
    #
    # The row's `degradation_speed` and `risk_velocity` columns are deliberately
    # NOT used as per-cycle rates. Both are deltas against the previous *scored*
    # point, and scoring runs at a stride (20 by default), so treating them as
    # per-cycle overstates the rate by that stride -- engine 227's risk_velocity
    # of 0.931 projected 15 cycles ahead saturated every horizon at 1.0 within
    # two steps. Fitting the slope over the last observed cycles measures the
    # per-cycle rate directly, whatever stride produced the points.
    steps = int(cycles_ahead)
    projected_rul = _extrapolate(rows, "predicted_rul", base_cycle, steps, floor=0.0)
    horizons = []
    for key in ("10", "30", "50"):
        projected = _extrapolate(rows, f"risk_{key}", base_cycle, steps, floor=0.0, ceiling=1.0)
        horizons.append(
            {
                "key": f"risk_{key}",
                "label": f"{key} cycles",
                "cycles": int(key),
                "value": None if projected is None else round(projected, 6),
                "threshold": _num(base.get(f"threshold_{key}")),
                "uncertainty": None,
                "predicted_failure": None,
            }
        )

    projected_decision = _project_decision(projected_rul, horizons, _text(base.get("anomaly_severity")))
    return {
        "engine_id": int(engine_id) if str(engine_id).isdigit() else str(engine_id),
        "base_cycle": base_cycle,
        "target_cycle": target_cycle,
        "cycles_ahead": steps,
        "basis": "extrapolated",
        "basis_note": (
            "Past the last observed cycle. Each signal is projected along the linear "
            "slope of its own last observed cycles; no model was re-run and no "
            "interval is claimed."
        ),
        "failure_risk": {
            "headline": next(h for h in horizons if h["key"] == "risk_30"),
            "horizons": horizons,
            "risk_state": None,
            "control_risk": None,
            "risk_velocity": _num(base.get("risk_velocity")),
            "risk_acceleration": _num(base.get("risk_acceleration")),
            "horizon_unit": "cycles",
        },
        "rul": {
            "predicted_rul": None if projected_rul is None else round(projected_rul, 2),
            "interval": {"lower": None, "upper": None, "margin": None, "source": "extrapolated", "label": "Extrapolated — no interval"},
            "lifecycle_state": None,
            "degradation_speed": _num(base.get("degradation_speed")),
            "degradation_acceleration": _num(base.get("degradation_acceleration")),
            "model_used": None,
            "fallback_triggered": None,
        },
        "anomaly": None,
        "recommendation": projected_decision,
        "components": [],
    }


def _project_decision(
    rul: Optional[float], horizons: List[Dict[str, Any]], severity: Optional[str]
) -> Dict[str, Any]:
    """Re-apply the safety policy's own thresholds to a projected state.

    Only the guardrail rules are replayed -- the champion model is not called on
    invented feature values -- so this reports the floor the policy would
    enforce, not a model prediction.
    """
    risk_10 = next((h["value"] for h in horizons if h["key"] == "risk_10"), None)
    risk_30 = next((h["value"] for h in horizons if h["key"] == "risk_30"), None)
    decision = "continue_operation"
    reasons: List[str] = []

    if rul is not None:
        if rul <= _CFG.immediate_rul_threshold:
            decision, _ = "immediate_maintenance", reasons.append(
                f"RUL ≤ {_CFG.immediate_rul_threshold:g}"
            )
        elif rul <= _CFG.maintenance_rul_threshold:
            decision = "schedule_maintenance"
        elif rul <= _CFG.inspection_rul_threshold:
            decision = "schedule_inspection"
        elif rul <= _CFG.monitoring_rul_threshold:
            decision = "monitor_closely"

    if risk_10 is not None and risk_10 >= _CFG.critical_risk_10_threshold:
        decision = "immediate_maintenance"
        reasons.append(f"risk_10 ≥ {_CFG.critical_risk_10_threshold:g}")
    if (severity or "").lower() == "critical":
        decision = "immediate_maintenance"
        reasons.append("anomaly severity critical")
    if (
        risk_30 is not None
        and risk_30 >= _CFG.maintenance_risk_30_threshold
        and DECISION_TO_SEVERITY.get(decision, 0) < 3
    ):
        decision = "schedule_maintenance"
        reasons.append(f"risk_30 ≥ {_CFG.maintenance_risk_30_threshold:g}")

    return {
        "final_decision": decision,
        "final_decision_label": _pretty(decision),
        "action_code": DECISION_TO_SEVERITY.get(decision, 0),
        "severity": DECISION_TO_SEVERITY.get(decision, 0),
        "recommended_action": ACTION_MAPPING.get(decision),
        "priority": PRIORITY_MAPPING.get(decision),
        "urgency": URGENCY_MAPPING.get(decision),
        "urgency_label": _pretty(URGENCY_MAPPING.get(decision)),
        "confidence": None,
        "confidence_level": None,
        "basis": "safety-policy thresholds replayed on the projected state",
        "reasons": reasons,
        "requires_human_review": None,
    }


def decision_certificate(engine_id: int | str, cycle: Optional[int] = None) -> Dict[str, Any]:
    """A signed-off, self-contained record of one decision, for the audit file.

    Everything an auditor needs to reconstruct the decision without the
    dashboard: the three agent readings that went in, every safety rule and
    whether it fired, the model that produced the candidate, the override
    record, and the artefact the numbers were read from.
    """
    view = agent_view(engine_id, cycle)
    issued = datetime.now(tz=timezone.utc)
    return {
        "certificate_id": f"AMS-{view['engine_id']}-C{view['cycle']}-{issued.strftime('%Y%m%dT%H%M%SZ')}",
        "issued_at": issued.isoformat(),
        "issuer": "Autonomous Maintenance Supervisor Agent",
        "unit": {
            "engine_id": view["engine_id"],
            "label": view["label"],
            "fd_subset": view["fd_subset"],
            "cycle": view["cycle"],
            "cycles_observed": view["cycle_range"]["observed"],
        },
        "agent_inputs": {
            "rul": view["rul"],
            "failure_risk": view["failure_risk"],
            "anomaly": view["anomaly"],
        },
        "decision": view["recommendation"],
        "policy": view["policy"],
        "safety_rules": view["xai"]["policy_drivers"],
        "upstream_contributions": view["xai"]["upstream_contributions"],
        "justification": view["xai"]["narrative"],
        "counterfactual": view["counterfactual"],
        "provenance": view["source"],
        "disclaimer": (
            "Generated from the fused decision table written by the last pipeline "
            "run. Figures are read, not recomputed at issue time."
        ),
    }


def fusion_summary() -> Dict[str, Any]:
    """Fleet-level view of what the fusion step did on the last run."""
    frame = load_decisions()
    ordered = frame.sort_values(["engine_id", "cycle"])
    latest = ordered.drop_duplicates(subset="engine_id", keep="last")

    def tally(column: str) -> Dict[str, int]:
        if column not in latest.columns:
            return {}
        return {str(k): int(v) for k, v in latest[column].dropna().astype(str).value_counts().items()}

    agreed = 0
    if {"raw_model_decision", "final_decision"} <= set(latest.columns):
        agreed = int((latest["raw_model_decision"] == latest["final_decision"]).sum())

    overrides = latest["override_applied"].astype(str).str.lower().isin({"true", "1", "1.0"})
    return {
        "engines": int(len(latest)),
        "rows_total": int(len(frame)),
        "decision_counts": tally("final_decision"),
        "priority_counts": tally("priority"),
        "risk_state_counts": tally("risk_state"),
        "health_state_counts": tally("health_state"),
        "model_final_agreement": round(100.0 * agreed / max(len(latest), 1), 2),
        "override_count": int(overrides.sum()),
        "override_reasons": {
            str(k): int(v)
            for k, v in latest.loc[overrides, "override_reason"].dropna().astype(str).value_counts().items()
        }
        if "override_reason" in latest.columns
        else {},
        "mean_confidence": _num(latest["confidence_score"].mean())
        if "confidence_score" in latest.columns
        else None,
        "decisions_written_at": decisions_written_at(),
    }


# ---------------------------------------------------------------------------
# Human feedback and policy adaptation
# ---------------------------------------------------------------------------

def submit_feedback(
    engine_id: int | str,
    cycle: int,
    engineer_decision: str,
    engineer_id: str = "CHIEF_ENGINEER",
    comments: str = "",
    adapt_policy: bool = True,
) -> Dict[str, Any]:
    """Record an engineer's verdict on a fused decision, then re-derive the policy.

    The Supervisor's ``SupervisorFeedbackStore`` and ``AdaptivePolicyEngine``
    already existed but nothing in the product ever called them, so the
    continuous-learning loop had never run on real input. This is the entry
    point: the verdict is appended to the audit-grade JSONL store keyed to the
    exact engine and cycle, and the adaptive-policy pass is re-run over the full
    feedback history.

    It does not retrain any model. Retraining stays a deliberate offline action
    -- no request through this API can change a model artefact.

    Args:
        engine_id: Engine the decision was about.
        cycle: Cycle the decision was made at.
        engineer_decision: The engineer's chosen action; must be a member of
            ``DECISION_CLASSES``.
        engineer_id: Who signed off.
        comments: Free-text rationale.
        adapt_policy: Run the adaptive-policy pass after logging.

    Returns:
        The stored entry, whether it agreed with the agent, and the policy
        report if one was produced.

    Raises:
        ValueError: If ``engineer_decision`` is not a known decision class.
    """
    decision = str(engineer_decision).strip().lower()
    if decision not in DECISION_CLASSES:
        raise ValueError(
            f"Unknown decision '{engineer_decision}'. Expected one of: {DECISION_CLASSES}."
        )

    rows = engine_rows(engine_id)
    row = pick_cycle(rows, cycle)
    original = _text(row.get("final_decision")) or "continue_operation"

    from app.services.Maintenance_Supervisor.feedback.feedback_store import (
        SupervisorFeedbackStore,
    )

    store = SupervisorFeedbackStore()
    entry = store.log_feedback(
        # Same `ENG_<id>_CYCLE_<n>` shape the store's own history filter
        # substring-matches on, so a logged verdict is retrievable by engine.
        sample_id=f"ENG_{engine_id}_CYCLE_{int(row['cycle'])}",
        original_decision=original,
        engineer_decision=decision,
        engineer_id=str(engineer_id or "CHIEF_ENGINEER"),
        comments=str(comments or ""),
    )

    policy_report = None
    if adapt_policy:
        from app.services.Maintenance_Supervisor.feedback.adaptive_policy import (
            run_adaptive_policy,
        )

        policy_report = run_adaptive_policy()

    return {
        "status": "success",
        "entry": entry.__dict__,
        "agreed_with_agent": decision == original,
        "agent_decision": original,
        "agent_decision_label": _pretty(original),
        "engineer_decision": decision,
        "engineer_decision_label": _pretty(decision),
        "severity_delta": DECISION_TO_SEVERITY.get(decision, 0)
        - DECISION_TO_SEVERITY.get(original, 0),
        "policy": policy_report,
        "total_feedback_entries": _feedback_count(),
        "note": "Feedback is logged and the policy pass re-run. No model was retrained.",
    }


def feedback_history(engine_id: Optional[int | str] = None, limit: int = 50) -> Dict[str, Any]:
    """Engineer verdicts logged so far, newest first."""
    from app.services.Maintenance_Supervisor.feedback.feedback_store import (
        SupervisorFeedbackStore,
    )

    records = SupervisorFeedbackStore().load_all_feedback()
    if engine_id is not None:
        # Anchored on the delimiter so engine 1 does not match engine 101.
        needle = f"ENG_{engine_id}_CYCLE_"
        records = [r for r in records if str(r.get("sample_id", "")).startswith(needle)]

    records.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    total = len(records)
    page = records[:limit]
    for record in page:
        stamp = record.get("timestamp")
        record["timestamp_iso"] = (
            datetime.fromtimestamp(float(stamp), tz=timezone.utc).isoformat() if stamp else None
        )
        record["agreed"] = record.get("original_decision") == record.get("engineer_decision")

    overrides = sum(1 for r in records if r.get("original_decision") != r.get("engineer_decision"))
    return {
        "total": total,
        "returned": len(page),
        "override_count": overrides,
        "agreement_pct": round(100.0 * (total - overrides) / total, 2) if total else None,
        "records": page,
    }


def run_live_fusion(engine_id: int | str, cycle: Optional[int] = None) -> Dict[str, Any]:
    """Re-run the Supervisor's decision-fusion engine on one engine, right now.

    Calls ``DecisionFusionEngine.fuse_decisions`` over that engine's own stored
    upstream features -- the same code path the batch pipeline uses -- so the
    button labelled "Run Decision Fusion" genuinely executes the fusion step
    rather than re-reading its cached result. Scoped to one engine because the
    full table is ~34k rows.

    The stored decision is left untouched: this returns what fusion produces
    now, next to what it produced on the last batch run, so a drift between the
    two is visible instead of silently overwritten.
    """
    from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import (
        DecisionFusionEngine,
    )

    rows = engine_rows(engine_id)
    row = pick_cycle(rows, cycle)
    resolved_cycle = int(row["cycle"])

    # The fused table carries the Supervisor's own output columns alongside the
    # upstream features. Feeding its own decisions back in would let the engine
    # read them as inputs, so they are dropped first.
    produced = [
        "raw_model_decision", "final_decision", "override_applied", "override_reason",
        "priority", "maintenance_urgency", "recommended_action", "confidence_score",
        "confidence_level", "entropy_score", "conflict_score", "requires_human_review",
        "review_reason",
    ]
    features = rows.drop(columns=[c for c in produced if c in rows.columns])

    started = time.perf_counter()
    fused = DecisionFusionEngine().fuse_decisions(features)
    elapsed_ms = round(1000.0 * (time.perf_counter() - started), 1)

    match = fused[fused["cycle"].astype(int) == resolved_cycle]
    live = match.iloc[0] if not match.empty else fused.iloc[-1]

    stored_decision = _text(row.get("final_decision"))
    live_decision = _text(live.get("final_decision"))

    return {
        "status": "success",
        "engine_id": int(engine_id) if str(engine_id).isdigit() else str(engine_id),
        "cycle": resolved_cycle,
        "rows_fused": int(len(fused)),
        "latency_ms": elapsed_ms,
        "live": _recommendation_block(live),
        "stored": _recommendation_block(row),
        "matches_stored": stored_decision == live_decision,
        "decision_distribution": {
            str(k): int(v) for k, v in fused["final_decision"].value_counts().items()
        },
        "note": (
            "Fusion was executed live over this engine's stored upstream features. "
            "The persisted decision table was not modified."
        ),
    }
