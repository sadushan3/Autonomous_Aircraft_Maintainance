"""
supervisor_explainer.py

Unified Explainability & Reasoning Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Orchestrates multi-level decision explainability combining:
1. Feature attribution (SHAP values or feature importance).
2. Rule-based natural language template narratives.
3. Upstream contributor breakdown (RUL vs Risk vs Anomaly contribution percentages).

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.explanation.supervisor_explainer

Expected outputs
----------------
- reports/Maintenance_Supervisor/supervisor_explainer_report.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, ARTIFACT_ROOT
from app.services.Maintenance_Supervisor.explanation.template_explainer import generate_decision_narrative
from app.services.Maintenance_Supervisor.explanation.shap_explainer import compute_feature_attributions_labelled
from app.services.Maintenance_Supervisor.models.model_registry import load_model
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_CHAMPION_MANIFEST: Final[Path] = ARTIFACT_ROOT / "champion_model_manifest.json"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "supervisor_explainer_report.json"


def _load_champion_model():
    """Load the champion model for attribution, mirroring decision_fusion_engine.py."""
    if not _CHAMPION_MANIFEST.exists():
        return None
    try:
        with open(_CHAMPION_MANIFEST, "r", encoding="utf-8") as f:
            champ_data = json.load(f)
        model_path = champ_data.get("champion_model_path")
        # The manifest's path is absolute and baked in at training time --
        # invalid on any other machine or after this project directory is
        # moved/renamed (see decision_fusion_engine.py's
        # _resolve_champion_model_path). Fall back to re-deriving it from
        # this machine's own ARTIFACT_ROOT before giving up.
        if not (model_path and Path(model_path).exists()):
            champion_name = champ_data.get("champion_model_name")
            fallback_path = ARTIFACT_ROOT / "models" / f"{champion_name}.joblib" if champion_name else None
            model_path = fallback_path if fallback_path and fallback_path.exists() else None
        if model_path and Path(model_path).exists():
            return load_model(model_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not load champion model for SHAP attribution: %s", exc)
    return None


def load_champion_model_and_background(background_size: int = 30) -> tuple[Any | None, pd.DataFrame | None]:
    """Load the champion model plus a matching background sample for SHAP.

    Public so callers outside this module (supervisor_service.py's live
    singleton, api_routes.py's /explain endpoint) can wire real SHAP the
    same way run_supervisor_explainer() below does, instead of each
    constructing a bare SupervisorExplainer() that silently falls back to
    the magnitude-proxy heuristic (see shap_explainer.py).

    The background sample must be in the same representation the champion
    was trained on: PPO/DQN train on raw physical-unit features
    (processed/.../supervisor_encoded_dataset.csv), while the classifier
    baselines train on the RobustScaler-scaled splits. Feeding a
    scaled-trained classifier raw values (or vice versa) reproduces the
    exact representation-mismatch bug documented in ppo_supervisor.py's
    module docstring, so the source is picked by champion type.
    """
    model = _load_champion_model()
    if model is None:
        return None, None

    is_rl_style = model.model_name in ("ppo", "dqn")
    source_path = (
        PROCESSED_ROOT / "supervisor_encoded_dataset.csv" if is_rl_style else _TEST_PATH
    )
    if not source_path.exists():
        source_path = _TEST_PATH if source_path != _TEST_PATH and _TEST_PATH.exists() else source_path
    if not source_path.exists():
        logger.warning("No background dataset found at %s for SHAP attribution.", source_path)
        return model, None

    try:
        background = pd.read_csv(source_path, low_memory=False)
        if len(background) > background_size:
            background = background.sample(n=background_size, random_state=42)
        return model, background
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not load SHAP background dataset from %s: %s", source_path, exc)
        return model, None


@dataclass
class ExplanationResult:
    sample_id: int
    decision: str
    explanation_narrative: str
    top_feature_attributions: dict[str, float]
    attribution_method: str
    upstream_contributions: dict[str, float]


class SupervisorExplainer:
    """Master Decision Explainer."""

    def __init__(self, model: Any | None = None, background: pd.DataFrame | None = None):
        self.model = model
        self.background = background

    def explain_sample(self, row: pd.Series, decision: str) -> ExplanationResult:
        narrative = generate_decision_narrative(row, decision)
        attribution_result = compute_feature_attributions_labelled(
            row, model=self.model, background=self.background,
        )
        attributions = attribution_result["attributions"]

        # Estimate upstream contributor share
        rul_val = float(row.get("predicted_rul", 100.0))
        risk_val = float(row.get("risk_30", 0.0))
        anom_val = float(row.get("anomaly_score", 0.0))

        s_rul = max(0.0, 1.0 - (rul_val / 150.0))
        s_risk = risk_val
        s_anom = anom_val
        total_s = s_rul + s_risk + s_anom + 1e-6

        upstream = {
            "rul_contribution_pct": round(100.0 * s_rul / total_s, 2),
            "risk_contribution_pct": round(100.0 * s_risk / total_s, 2),
            "anomaly_contribution_pct": round(100.0 * s_anom / total_s, 2),
        }

        return ExplanationResult(
            sample_id=int(row.get("cycle", 0)),
            decision=decision,
            explanation_narrative=narrative,
            top_feature_attributions=attributions,
            attribution_method=attribution_result["method"],
            upstream_contributions=upstream,
        )


def run_supervisor_explainer(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("SUPERVISOR EXPLAINABILITY ENGINE STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)
    # Background must match the champion's own training representation
    # (raw for PPO/DQN, scaled for the classifier baselines) -- see
    # load_champion_model_and_background's docstring -- so it is not
    # simply sampled from whatever input_path happens to be.
    model, background = load_champion_model_and_background()
    explainer = SupervisorExplainer(model=model, background=background)

    sample_explanations = []
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        res = explainer.explain_sample(row, decision=str(row.get("final_decision", "immediate_maintenance")))
        sample_explanations.append({
            "sample_index": idx,
            "decision": res.decision,
            "narrative": res.explanation_narrative,
            "upstream_contributions": res.upstream_contributions,
            "top_attributions": res.top_feature_attributions,
            "attribution_method": res.attribution_method,
        })

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "sample_explanations": sample_explanations,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Generated explanations for sample rows.")
    logger.info("Report written to: %s", _REPORT_PATH)
    section("SUPERVISOR EXPLAINABILITY ENGINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_supervisor_explainer()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in supervisor_explainer: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
