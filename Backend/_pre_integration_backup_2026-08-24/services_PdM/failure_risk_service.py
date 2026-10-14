from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config.Predictive_Maintenance.config import Settings
from app.schemas.Predictive_Maintenance.failure_risk_schema import (
    FailureRiskPredictionRequest,
    FailureRiskPredictionResponse,
)
from app.services.Predictive_Maintenance.adaptive_threshold_service import AdaptiveThresholdService
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader
from app.services.Predictive_Maintenance.preprocessing import PreprocessingService
from app.utils.Predictive_Maintenance.errors import PredictionError
from app.utils.Predictive_Maintenance.math_utils import clamp_probability, recent_acceleration, recent_velocity


class FailureRiskService:
    def __init__(
        self,
        settings: Settings,
        artifact_loader: ArtifactLoader,
        preprocessing: PreprocessingService,
        adaptive_threshold_service: Optional[AdaptiveThresholdService] = None,
    ):
        self.settings = settings
        self.artifact_loader = artifact_loader
        self.preprocessing = preprocessing
        self.adaptive_threshold_service = adaptive_threshold_service

    def predict(self, request: FailureRiskPredictionRequest) -> FailureRiskPredictionResponse:
        fd_subset = self.artifact_loader.validate_fd_subset(request.fd_subset)
        artifacts = self.artifact_loader.load("failure_risk", fd_subset)
        window_size = self._window_size(artifacts.metadata)

        tensor, prepared_window = self.preprocessing.sequence_to_tensor(
            sequence=request.sequence,
            feature_columns=artifacts.feature_columns,
            scaler=artifacts.scaler,
            window_size=window_size,
        )

        try:
            deterministic = artifacts.model.predict(tensor, verbose=0)
        except Exception as exc:
            raise PredictionError(
                f"Failure risk model prediction failed for {fd_subset}: {exc}"
            ) from exc

        risks = self._extract_risks(deterministic)
        uncertainties = (0.0, 0.0, 0.0)
        if request.include_uncertainty:
            uncertainties = self._mc_dropout_uncertainty(artifacts.model, tensor)

        thresholds = self._thresholds(fd_subset, artifacts.metadata)
        risk_velocity = recent_velocity(
            request.previous_risk_scores, risks[1], increasing=True
        )
        risk_acceleration = recent_acceleration(
            request.previous_risk_scores, risks[1], increasing=True
        )
        risk_state = self._risk_state(
            risk_30=risks[1],
            threshold_30=thresholds["30"],
            risk_velocity=risk_velocity,
            risk_acceleration=risk_acceleration,
        )
        action_hint = self._action_hint(risk_state)

        debug: Optional[Dict[str, Any]] = None
        if request.include_debug:
            debug = {
                "window_size": window_size,
                "feature_count": len(artifacts.feature_columns),
                "artifact_folder": str(artifacts.paths.folder),
                "model_path": str(artifacts.paths.model),
                "scaler_path": str(artifacts.paths.scaler),
                "feature_columns_path": str(artifacts.paths.feature_columns),
                "metadata_path": str(artifacts.paths.metadata) if artifacts.paths.metadata else None,
                "last_preprocessed_cycle_count": len(prepared_window),
                "zero_filled_features": prepared_window.attrs.get(
                    "zero_filled_features", []
                ),
                "mc_passes": self.settings.FAILURE_RISK_MC_PASSES
                if request.include_uncertainty
                else 0,
            }

        return FailureRiskPredictionResponse(
            fd_subset=fd_subset,
            risk_10=round(risks[0], 4),
            risk_30=round(risks[1], 4),
            risk_50=round(risks[2], 4),
            uncertainty_10=round(uncertainties[0], 4),
            uncertainty_30=round(uncertainties[1], 4),
            uncertainty_50=round(uncertainties[2], 4),
            threshold_10=thresholds["10"],
            threshold_30=thresholds["30"],
            threshold_50=thresholds["50"],
            predicted_fail_in_10=risks[0] >= thresholds["10"],
            predicted_fail_in_30=risks[1] >= thresholds["30"],
            predicted_fail_in_50=risks[2] >= thresholds["50"],
            control_risk=round(risks[1], 4),
            risk_velocity=round(risk_velocity, 4),
            risk_acceleration=round(risk_acceleration, 4),
            risk_state=risk_state,
            action_hint_for_supervisor=action_hint,
            model_used="BiLSTM_MC_Dropout",
            debug=debug,
        )

    def _window_size(self, metadata: Dict[str, Any]) -> int:
        for key in ("window_size", "sequence_length", "timesteps"):
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return self.settings.FAILURE_RISK_WINDOW_SIZE

    def _thresholds(self, fd_subset: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        metadata_thresholds = metadata.get("thresholds")
        source = (
            metadata_thresholds
            if isinstance(metadata_thresholds, dict)
            else self.settings.FAILURE_RISK_DEFAULT_THRESHOLDS.get(fd_subset, {})
        )
        static = {
            "10": clamp_probability(source.get("10", source.get(10, 0.5))),
            "30": clamp_probability(source.get("30", source.get(30, 0.5))),
            "50": clamp_probability(source.get("50", source.get(50, 0.5))),
        }
        if self.adaptive_threshold_service is None:
            return static
        return {
            horizon: clamp_probability(
                self.adaptive_threshold_service.get_threshold(fd_subset, horizon, value)
            )
            for horizon, value in static.items()
        }

    @staticmethod
    def _extract_risks(prediction: object) -> Tuple[float, float, float]:
        values = np.asarray(prediction).reshape(-1)
        if values.size < 3:
            raise PredictionError(
                "Failure risk model must return at least three outputs for 10, 30, and 50 cycles."
            )
        return (
            clamp_probability(values[0]),
            clamp_probability(values[1]),
            clamp_probability(values[2]),
        )

    def _mc_dropout_uncertainty(self, model: object, tensor: np.ndarray) -> Tuple[float, float, float]:
        samples = []
        for _ in range(max(1, self.settings.FAILURE_RISK_MC_PASSES)):
            try:
                stochastic = model(tensor, training=True)
            except Exception as exc:
                raise PredictionError(f"MC dropout inference failed: {exc}") from exc
            samples.append(np.asarray(stochastic).reshape(-1)[:3])
        std = np.std(np.asarray(samples, dtype=float), axis=0)
        return (
            clamp_probability(std[0]),
            clamp_probability(std[1]),
            clamp_probability(std[2]),
        )

    @staticmethod
    def _risk_state(
        risk_30: float,
        threshold_30: float,
        risk_velocity: float,
        risk_acceleration: float,
    ) -> str:
        # Compare against the calibrated threshold alone. The previous
        # ``max(0.80, threshold_30)`` floor silently overrode any subset whose
        # calibrated cut sits below 0.80: FD002's is 0.031, so a window could
        # be returned with ``predicted_fail_in_30=True`` while this reported
        # "Stable" and ``action_hint_for_supervisor`` said "continue_monitoring".
        # The two fields are consumed together by the Maintenance Supervisor,
        # so they must not be allowed to contradict each other.
        if risk_30 >= threshold_30:
            return "Critical"
        if risk_acceleration > 0.10 and risk_velocity > 0.05:
            return "Accelerating"
        if risk_velocity > 0.05:
            return "Rising"
        return "Stable"

    @staticmethod
    def _action_hint(risk_state: str) -> str:
        hints = {
            "Critical": "maintenance_supervisor_priority_review",
            "Accelerating": "increase_monitoring_and_prepare_inspection",
            "Rising": "monitor_closely",
            "Stable": "continue_monitoring",
        }
        return hints.get(risk_state, "continue_monitoring")
