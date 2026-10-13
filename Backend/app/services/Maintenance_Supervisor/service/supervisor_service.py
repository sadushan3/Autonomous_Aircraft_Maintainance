"""
supervisor_service.py

High-Performance Core In-Memory Service Instance for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Maintains an in-memory singleton wrapper around the Champion ML model, safety policies,
and confidence engines to handle ultra-fast streaming predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.services.Maintenance_Supervisor.explanation.supervisor_explainer import (
    SupervisorExplainer,
    load_champion_model_and_background,
)
from app.services.Maintenance_Supervisor.service.schemas import TelemetryInputSchema, SupervisorDecisionResponseSchema
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()


class MaintenanceSupervisorService:
    """Core In-Memory Production Supervisor Service."""

    _instance: MaintenanceSupervisorService | None = None

    def __init__(self):
        logger.info("Initializing Maintenance Supervisor Core Service...")
        self.fusion_engine = DecisionFusionEngine()
        # Wired with the champion model + a matching background sample so
        # /predict and /explain compute real SHAP attributions instead of
        # silently falling back to the magnitude-proxy heuristic (the
        # bare SupervisorExplainer() this used to construct never had a
        # model to explain against).
        explainer_model, explainer_background = load_champion_model_and_background()
        self.explainer = SupervisorExplainer(model=explainer_model, background=explainer_background)
        logger.info("Maintenance Supervisor Core Service Ready.")

    @classmethod
    def get_instance(cls) -> MaintenanceSupervisorService:
        if cls._instance is None:
            cls._instance = MaintenanceSupervisorService()
        return cls._instance

    def predict_single(self, telemetry: TelemetryInputSchema) -> SupervisorDecisionResponseSchema:
        data_dict = telemetry.model_dump()
        extra = data_dict.pop("extra_features", {}) or {}
        data_dict.update(extra)

        df = pd.DataFrame([data_dict])
        fused_df = self.fusion_engine.fuse_decisions(df)
        res_row = fused_df.iloc[0]

        return SupervisorDecisionResponseSchema(
            engine_id=str(res_row.get("engine_id", telemetry.engine_id)),
            cycle=int(res_row.get("cycle", telemetry.cycle)),
            final_decision=str(res_row["final_decision"]),
            priority=str(res_row["priority"]),
            maintenance_urgency=str(res_row["maintenance_urgency"]),
            recommended_action=str(res_row["recommended_action"]),
            confidence_score=float(res_row["confidence_score"]),
            confidence_level=str(res_row["confidence_level"]),
            entropy_score=float(res_row["entropy_score"]),
            conflict_score=float(res_row["conflict_score"]),
            requires_human_review=bool(res_row["requires_human_review"]),
            review_reasons=str(res_row.get("review_reason", "")),
            override_applied=bool(res_row["override_applied"]),
            override_reason=str(res_row.get("override_reason", "")),
        )

    def explain_single(self, telemetry: TelemetryInputSchema) -> dict:
        """Fuse a decision for `telemetry` and explain it with the wired champion model."""
        data_dict = telemetry.model_dump()
        extra = data_dict.pop("extra_features", {}) or {}
        data_dict.update(extra)

        df = pd.DataFrame([data_dict])
        fused_df = self.fusion_engine.fuse_decisions(df)
        res_row = fused_df.iloc[0]
        decision = str(res_row["final_decision"])

        result = self.explainer.explain_sample(res_row, decision=decision)
        return {
            "decision": result.decision,
            "explanation_narrative": result.explanation_narrative,
            "upstream_contributions": result.upstream_contributions,
            "top_attributions": result.top_feature_attributions,
            "attribution_method": result.attribution_method,
        }
