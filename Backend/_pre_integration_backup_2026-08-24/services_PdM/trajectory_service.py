from __future__ import annotations

from app.config.Predictive_Maintenance.config import Settings
from app.schemas.Predictive_Maintenance.failure_risk_schema import FailureRiskPredictionRequest
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionRequest
from app.schemas.Predictive_Maintenance.trajectory_schema import (
    TrajectoryPoint,
    TrajectoryRequest,
    TrajectoryResponse,
)
from app.services.Predictive_Maintenance.failure_risk_service import FailureRiskService
from app.services.Predictive_Maintenance.rul_lifecycle_service import RULLifecycleService
from app.utils.Predictive_Maintenance.errors import InvalidSequenceError


class TrajectoryService:
    """Replays a full sequence through both agents at growing window sizes.

    The single-shot endpoints only ever score the latest window. This walks
    the sequence forward so the frontend can chart how RUL, uncertainty and
    risk evolved cycle-over-cycle, instead of a single point-in-time read.
    """

    def __init__(
        self,
        settings: Settings,
        rul_service: RULLifecycleService,
        failure_risk_service: FailureRiskService,
    ):
        self.settings = settings
        self.rul_service = rul_service
        self.failure_risk_service = failure_risk_service

    def compute(self, request: TrajectoryRequest) -> TrajectoryResponse:
        fd_subset = self.rul_service.artifact_loader.validate_fd_subset(request.fd_subset)
        sequence = sorted(request.sequence, key=lambda row: row.cycle)
        total = len(sequence)

        min_window = max(
            self.settings.RUL_WINDOW_SIZES.get(fd_subset, 30),
            self.settings.FAILURE_RISK_WINDOW_SIZE,
            30,
        )
        if total < min_window:
            raise InvalidSequenceError(
                f"Trajectory for {fd_subset} needs at least {min_window} cycles, got {total}."
            )

        stride = request.stride or max(1, (total - min_window) // request.max_points or 1)
        end_indices = list(range(min_window, total + 1, stride))
        if end_indices[-1] != total:
            end_indices.append(total)

        points: list[TrajectoryPoint] = []
        previous_rul: list[float] = []
        previous_risk: list[float] = []

        for end in end_indices:
            window = sequence[:end]
            rul_response = self.rul_service.predict(
                RULPredictionRequest(
                    fd_subset=fd_subset,
                    engine_id=request.engine_id,
                    sequence=window,
                    previous_rul_predictions=previous_rul,
                    include_debug=False,
                )
            )
            risk_response = self.failure_risk_service.predict(
                FailureRiskPredictionRequest(
                    fd_subset=fd_subset,
                    engine_id=request.engine_id,
                    sequence=window,
                    previous_risk_scores=previous_risk,
                    include_uncertainty=False,
                    include_debug=False,
                )
            )

            points.append(
                TrajectoryPoint(
                    cycle=window[-1].cycle,
                    predicted_rul=rul_response.predicted_rul,
                    lower_bound=rul_response.lower_bound,
                    upper_bound=rul_response.upper_bound,
                    uncertainty_width=rul_response.uncertainty_width,
                    degradation_speed=rul_response.degradation_speed,
                    degradation_acceleration=rul_response.degradation_acceleration,
                    lifecycle_state=rul_response.lifecycle_state,
                    confidence_collapse_flag=rul_response.confidence_collapse_flag,
                    correction_applied=rul_response.correction_applied,
                    risk_10=risk_response.risk_10,
                    risk_30=risk_response.risk_30,
                    risk_50=risk_response.risk_50,
                    risk_state=risk_response.risk_state,
                )
            )

            previous_rul = (previous_rul + [rul_response.predicted_rul])[-5:]
            previous_risk = (previous_risk + [risk_response.risk_30])[-5:]

        return TrajectoryResponse(
            fd_subset=fd_subset,
            engine_id=request.engine_id,
            stride=stride,
            window_start_cycle=sequence[0].cycle,
            points=points,
        )
