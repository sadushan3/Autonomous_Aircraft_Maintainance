from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.config.Predictive_Maintenance.config import Settings
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionRequest, RULPredictionResponse
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader
from app.services.Predictive_Maintenance.preprocessing import PreprocessingService
from app.utils.Predictive_Maintenance.errors import PredictionError
from app.utils.Predictive_Maintenance.math_utils import recent_acceleration, recent_velocity, to_float


class RULLifecycleService:
    def __init__(
        self,
        settings: Settings,
        artifact_loader: ArtifactLoader,
        preprocessing: PreprocessingService,
    ):
        self.settings = settings
        self.artifact_loader = artifact_loader
        self.preprocessing = preprocessing

    def predict(self, request: RULPredictionRequest) -> RULPredictionResponse:
        fd_subset = self.artifact_loader.validate_fd_subset(request.fd_subset)
        artifacts = self.artifact_loader.load("rul", fd_subset)
        window_size = self.settings.RUL_WINDOW_SIZES.get(fd_subset, 30)

        tensor, prepared_window = self.preprocessing.sequence_to_tensor(
            sequence=request.sequence,
            feature_columns=artifacts.feature_columns,
            scaler=artifacts.scaler,
            window_size=window_size,
        )

        try:
            raw_prediction = artifacts.model.predict(tensor, verbose=0)
        except Exception as exc:
            raise PredictionError(f"RUL model prediction failed for {fd_subset}: {exc}") from exc

        primary_predicted_rul = max(0.0, to_float(np.asarray(raw_prediction).reshape(-1)[0]))
        q_hat = self.settings.DEFAULT_RUL_Q_HAT.get(fd_subset, 20.0)

        degradation_speed = recent_velocity(
            request.previous_rul_predictions, primary_predicted_rul, increasing=False
        )
        degradation_acceleration = recent_acceleration(
            request.previous_rul_predictions, primary_predicted_rul, increasing=False
        )

        # Confidence collapse is a *relative* property of the interval, not an
        # absolute one. ``q_hat`` is a per-subset constant, so the old test --
        # ``2 * q_hat > RUL_UNCERTAINTY_WIDTH_THRESHOLD[fd_subset]`` -- compared
        # a constant against a constant, and the config happens to set that
        # threshold to exactly ``2 * q_hat`` for all four subsets. The flag was
        # therefore False for every input the system has ever scored (verified:
        # 33,991/33,991 rows in the supervisor's inference input), which in turn
        # made ``instability_flag`` always False and left the whole baseline
        # fallback below, the "Unstable" lifecycle state, and the
        # ``fallback_triggered``/``correction_applied`` response fields dead.
        #
        # A +/-25 band around a predicted RUL of 200 is informative; the same
        # band around a predicted RUL of 10 is not. So compare the band actually
        # reported (clamped at zero, matching ``lower_bound`` below) against the
        # estimate it brackets: the interval is collapsed when it is wider than
        # the number it is supposed to qualify.
        primary_uncertainty_width = self.interval_width(primary_predicted_rul, q_hat)
        confidence_collapse_flag = self.is_confidence_collapsed(
            primary_predicted_rul, q_hat
        )
        sudden_health_drop_flag = degradation_speed > 5.0 or degradation_acceleration > 2.0
        instability_flag = confidence_collapse_flag and sudden_health_drop_flag

        # Self-correction gate: an unreliable TCN read (instability_flag) is
        # conditionally swapped for a tabular baseline regressor, but only if
        # the baseline artifact is available (fallback_allowed) and its
        # prediction clears a sanity check against the primary read
        # (validation gate) before it is accepted as the final prediction.
        predicted_rul = primary_predicted_rul
        model_used = "TCN_Conformal"
        fallback_allowed = False
        correction_applied = False
        fallback_model_used: Optional[str] = None
        fallback_debug: Dict[str, Any] = {}

        if instability_flag:
            baseline = self.artifact_loader.load_baseline_model(fd_subset)
            fallback_allowed = baseline is not None
            if baseline is not None:
                try:
                    row = self.preprocessing.tabular_last_row(
                        sequence=request.sequence,
                        feature_columns=artifacts.feature_columns,
                        scaler=artifacts.scaler,
                    )
                    baseline_prediction = max(0.0, to_float(baseline["model"].predict(row)[0]))
                    # A limit of 5x the larger of the two operands admits
                    # essentially any finite baseline value, so this "validation
                    # gate" passed unconditionally. It never mattered while the
                    # path above was dead; now that the path can fire, the gate
                    # has to actually reject a baseline that disagrees wildly,
                    # or the TCN is silently replaced with an unchecked number.
                    # Allow a factor-of-two disagreement (plus a small absolute
                    # floor so near-zero RUL comparisons stay meaningful) and
                    # keep the TCN read otherwise.
                    divergence_limit = max(primary_predicted_rul, baseline_prediction) + 10.0
                    validation_passed = (
                        baseline_prediction >= 0.0
                        and abs(baseline_prediction - primary_predicted_rul) < divergence_limit
                    )
                    fallback_debug = {
                        "fallback_prediction": round(baseline_prediction, 4),
                        "validation_passed": validation_passed,
                    }
                    if validation_passed:
                        predicted_rul = baseline_prediction
                        fallback_model_used = baseline["label"]
                        model_used = f"TCN_Conformal+{baseline['label']}_Fallback"
                        correction_applied = True
                except Exception as exc:
                    fallback_debug = {"fallback_error": str(exc)}

        lower_bound = max(0.0, predicted_rul - q_hat)
        upper_bound = predicted_rul + q_hat
        uncertainty_width = upper_bound - lower_bound
        lifecycle_state = self._lifecycle_state(
            predicted_rul=predicted_rul,
            degradation_speed=degradation_speed,
            degradation_acceleration=degradation_acceleration,
            instability_flag=instability_flag,
        )

        debug: Optional[Dict[str, Any]] = None
        if request.include_debug:
            debug = {
                "window_size": window_size,
                "feature_count": len(artifacts.feature_columns),
                "artifact_folder": str(artifacts.paths.folder),
                "model_path": str(artifacts.paths.model),
                "scaler_path": str(artifacts.paths.scaler),
                "feature_columns_path": str(artifacts.paths.feature_columns),
                "last_preprocessed_cycle_count": len(prepared_window),
                "zero_filled_features": prepared_window.attrs.get(
                    "zero_filled_features", []
                ),
                "q_hat": q_hat,
                **fallback_debug,
            }

        return RULPredictionResponse(
            fd_subset=fd_subset,
            predicted_rul=round(predicted_rul, 4),
            lower_bound=round(lower_bound, 4),
            upper_bound=round(upper_bound, 4),
            uncertainty_width=round(uncertainty_width, 4),
            lifecycle_state=lifecycle_state,
            degradation_speed=round(degradation_speed, 4),
            degradation_acceleration=round(degradation_acceleration, 4),
            confidence_collapse_flag=confidence_collapse_flag,
            sudden_health_drop_flag=sudden_health_drop_flag,
            instability_flag=instability_flag,
            model_used=model_used,
            fallback_triggered=instability_flag,
            fallback_allowed=fallback_allowed,
            correction_applied=correction_applied,
            fallback_model_used=fallback_model_used,
            primary_predicted_rul=round(primary_predicted_rul, 4),
            debug=debug,
        )

    @staticmethod
    def interval_width(predicted_rul: float, q_hat: float) -> float:
        """Width of the reported prediction interval, clamped at zero.

        Matches how ``lower_bound`` is derived, so the width used for the
        collapse test is the width actually reported to the caller.
        """
        return (predicted_rul + q_hat) - max(0.0, predicted_rul - q_hat)

    @classmethod
    def is_confidence_collapsed(cls, predicted_rul: float, q_hat: float) -> bool:
        """True when the interval is too wide to qualify the estimate it brackets."""
        return cls.interval_width(predicted_rul, q_hat) > predicted_rul

    @staticmethod
    def _lifecycle_state(
        predicted_rul: float,
        degradation_speed: float,
        degradation_acceleration: float,
        instability_flag: bool,
    ) -> str:
        if instability_flag:
            return "Unstable"
        if predicted_rul <= 20.0:
            return "Critical"
        if predicted_rul <= 50.0 and degradation_acceleration > 0.5:
            return "Accelerating"
        if predicted_rul <= 80.0 or degradation_speed > 0.5:
            return "Degrading"
        return "Stable"
