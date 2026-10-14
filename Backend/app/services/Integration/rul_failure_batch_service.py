"""
Drives the existing (unmodified) RUL Lifecycle and Failure Risk services across
every engine/cycle in an uploaded C-MAPSS file, and writes the results as CSVs
keyed on (engine_id, cycle, fd_subset) -- the "Data exchange" layer the
Maintenance Supervisor consumes.

This module contains NO prediction logic of its own. It only:
  1. groups the already-validated raw frame by engine ("unit"),
  2. builds the trailing-window ``EngineSensorRow`` sequences the ported
     ``RULLifecycleService``/``FailureRiskService`` require,
  3. calls those services in-process, and
  4. serialises their response objects straight to CSV (field names already
     match what the Maintenance Supervisor expects -- see the integration
     plan; no renaming is done here).

Performance note: each cycle with a full window triggers one model inference
call per agent. For large multi-engine files, pass a larger ``stride`` to
sub-sample cycles instead of scoring every single one.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.config.Predictive_Maintenance.config import get_settings
from app.schemas.Predictive_Maintenance.common import EngineSensorRow
from app.schemas.Predictive_Maintenance.failure_risk_schema import FailureRiskPredictionRequest
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionRequest
from app.services.Predictive_Maintenance.adaptive_threshold_service import AdaptiveThresholdService
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader
from app.services.Predictive_Maintenance.failure_risk_service import FailureRiskService
from app.services.Predictive_Maintenance.preprocessing import PreprocessingService
from app.services.Predictive_Maintenance.rul_lifecycle_service import RULLifecycleService

_settings = get_settings()
_artifact_loader = ArtifactLoader(_settings)
_preprocessing = PreprocessingService()
_adaptive_threshold_service = AdaptiveThresholdService(
    Path(__file__).resolve().parents[3] / "data" / "exchange" / "Predictive_Maintenance" / "adaptive_thresholds.json",
    _settings.FAILURE_RISK_DEFAULT_THRESHOLDS,
)
_rul_service = RULLifecycleService(_settings, _artifact_loader, _preprocessing)
_failure_risk_service = FailureRiskService(
    _settings, _artifact_loader, _preprocessing, _adaptive_threshold_service
)

logger = logging.getLogger(__name__)

_SENSOR_ROW_FIELDS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

DEFAULT_FD_SUBSET = "FD001"

#: Emit a progress line at most this often. This loop runs one model
#: inference per agent per cycle and takes tens of minutes on a full subset
#: while writing nothing to disk until the very end, so without periodic
#: output it is indistinguishable from a hang.
_PROGRESS_INTERVAL_SECONDS = 15.0


def _format_duration(seconds: float) -> str:
    """Render a duration as ``HH:MM:SS`` for progress lines."""
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _count_eligible_points(
    frame: pd.DataFrame, stride: int, settings: Any
) -> int:
    """Count the (engine, cycle) points the batch will actually score.

    Mirrors the eligibility rule used by the scoring loop so the progress
    line can report a real percentage and ETA rather than counting engines,
    which vary hugely in trajectory length.

    Args:
        frame: The frame about to be scored.
        stride: Cycle sub-sampling stride.
        settings: Predictive Maintenance settings supplying window sizes.

    Returns:
        Total scoring points across all engines.
    """
    total = 0
    for _, engine_frame in frame.groupby("unit"):
        fd_subset = DEFAULT_FD_SUBSET
        if "subset_id" in engine_frame.columns and pd.notna(engine_frame["subset_id"].iloc[0]):
            candidate = str(engine_frame["subset_id"].iloc[0]).upper()
            if candidate in settings.SUPPORTED_FDS:
                fd_subset = candidate
        window_needed = max(
            settings.RUL_WINDOW_SIZES.get(fd_subset, 50),
            settings.FAILURE_RISK_WINDOW_SIZE,
        )
        if len(engine_frame) < window_needed:
            continue
        total += len(
            set(range(window_needed - 1, len(engine_frame), max(stride, 1)))
            | {len(engine_frame) - 1}
        )
    return total


def _row_to_engine_sensor_row(row: pd.Series, unit: int, fd_subset: str) -> EngineSensorRow:
    payload: Dict[str, Any] = {
        "cycle": int(row["cycle"]),
        "unit": int(unit),
        "fd_subset": fd_subset,
    }
    for field in _SENSOR_ROW_FIELDS:
        payload[field] = float(row[field])
    return EngineSensorRow(**payload)


def run_rul_failure_batch(
    frame: pd.DataFrame,
    output_dir: Path,
    stride: int = 1,
    include_uncertainty: bool = True,
) -> Dict[str, Any]:
    """Score every (engine, cycle) window in ``frame`` and write two CSVs.

    Args:
        frame: Narrowed/validated raw frame with ``unit``, ``cycle``,
            ``op_setting_1..3``, ``sensor_1..21`` and optionally ``subset_id``
            (falls back to ``FD001`` per engine when absent).
        output_dir: Directory to write ``rul_predictions.csv`` and
            ``failure_risk_predictions.csv`` into.
        stride: Only score every Nth eligible cycle per engine (1 = every
            cycle once a full window is available).
        include_uncertainty: Run the Failure-Risk model's Monte-Carlo dropout
            passes to produce real ``uncertainty_10/30/50`` values.

            ``FailureRiskPredictionRequest.include_uncertainty`` defaults to
            ``False``, and this batch previously never set it, so the service
            returned its hardcoded ``(0.0, 0.0, 0.0)`` placeholder and every
            uncertainty column in the exchange layer was exactly zero -- a
            dead signal the Maintenance Supervisor's confidence engine was
            nonetheless consuming. The model produces a genuine spread when
            actually asked (measured std ~0.07/0.14/0.07 on real windows), so
            this now defaults to on.

            It costs ``FAILURE_RISK_MC_PASSES`` extra forward passes per
            scored point (20 by default), which dominates Failure-Risk
            inference time. Pass ``False`` -- or raise ``stride`` -- when a
            fast, uncertainty-free pass is wanted instead.

    Returns:
        A report dict with row counts and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rul_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total_engines = int(frame["unit"].nunique())
    total_points = _count_eligible_points(frame, stride, _settings)
    subsets_present = (
        sorted(frame["subset_id"].dropna().unique().tolist())
        if "subset_id" in frame.columns
        else []
    )
    mc_passes = _settings.FAILURE_RISK_MC_PASSES if include_uncertainty else 0
    logger.info(
        "RUL/Failure-Risk batch starting: %d engine(s), %d scoring point(s) at "
        "stride=%d, %d model call(s) total (uncertainty=%s%s). "
        "Subset labels present: %s",
        total_engines,
        total_points,
        stride,
        total_points * (2 + mc_passes),
        "ON" if include_uncertainty else "OFF",
        f", +{mc_passes} MC passes/point" if include_uncertainty else "",
        subsets_present or f"NONE -- every engine falls back to {DEFAULT_FD_SUBSET}",
    )
    if not subsets_present:
        logger.warning(
            "No '%s' column on the input frame, so every engine is scored with "
            "%s models regardless of which subset the data actually came from. "
            "Check that the upload filename names its subset.",
            "subset_id",
            DEFAULT_FD_SUBSET,
        )

    started = time.perf_counter()
    last_report = started
    points_done = 0
    engines_done = 0
    engines_skipped = 0

    for unit, engine_frame in frame.groupby("unit"):
        engine_frame = engine_frame.sort_values("cycle").reset_index(drop=True)
        fd_subset = DEFAULT_FD_SUBSET
        if "subset_id" in engine_frame.columns and pd.notna(engine_frame["subset_id"].iloc[0]):
            candidate = str(engine_frame["subset_id"].iloc[0]).upper()
            if candidate in _settings.SUPPORTED_FDS:
                fd_subset = candidate

        window_needed = max(
            _settings.RUL_WINDOW_SIZES.get(fd_subset, 50),
            _settings.FAILURE_RISK_WINDOW_SIZE,
        )
        if len(engine_frame) < window_needed:
            # Too short to form even one window. Counted rather than dropped
            # silently: an engine that never appears in the output at all is
            # easy to mistake for a scoring failure.
            engines_skipped += 1
            continue

        engine_id = str(int(unit))
        previous_rul: List[float] = []
        previous_risk: List[float] = []
        # Cycle of the last point scored for this engine, so the RUL service can
        # be told how far apart its readings actually are. The stride is not a
        # safe proxy: each engine's final cycle is always scored regardless of
        # where the stride grid falls, so the last gap is usually shorter.
        previous_cycle: int | None = None

        # Always score the engine's most recent cycle even if it doesn't land
        # on the stride grid -- that cycle is the one real-time monitoring
        # actually cares about, and silently skipping it left the fused
        # dashboard falling back to an imputed default for exactly the row
        # that matters most.
        eligible_indices = sorted(
            set(range(window_needed - 1, len(engine_frame), max(stride, 1)))
            | {len(engine_frame) - 1}
        )
        for end_idx in eligible_indices:
            window = engine_frame.iloc[end_idx - window_needed + 1 : end_idx + 1]
            sequence = [
                _row_to_engine_sensor_row(row, unit, fd_subset)
                for _, row in window.iterrows()
            ]
            cycle = int(window.iloc[-1]["cycle"])

            try:
                rul_response = _rul_service.predict(
                    RULPredictionRequest(
                        fd_subset=fd_subset,
                        engine_id=engine_id,
                        sequence=sequence,
                        previous_rul_predictions=previous_rul[-5:],
                        prediction_cycle_gap=(
                            float(cycle - previous_cycle)
                            if previous_cycle is not None and cycle > previous_cycle
                            else 1.0
                        ),
                    )
                )
                rul_rows.append(
                    {"engine_id": engine_id, "cycle": cycle, "fd_subset": fd_subset,
                     **rul_response.model_dump(exclude={"debug"})}
                )
                previous_rul.append(rul_response.predicted_rul)
                previous_cycle = cycle
            except Exception as exc:  # noqa: BLE001
                errors.append({"engine_id": engine_id, "cycle": cycle, "agent": "rul", "error": str(exc)})

            try:
                risk_response = _failure_risk_service.predict(
                    FailureRiskPredictionRequest(
                        fd_subset=fd_subset,
                        engine_id=engine_id,
                        sequence=sequence,
                        previous_risk_scores=previous_risk[-5:],
                        include_uncertainty=include_uncertainty,
                    )
                )
                risk_rows.append(
                    {"engine_id": engine_id, "cycle": cycle, "fd_subset": fd_subset,
                     **risk_response.model_dump(exclude={"debug"})}
                )
                previous_risk.append(risk_response.risk_30)
            except Exception as exc:  # noqa: BLE001
                errors.append({"engine_id": engine_id, "cycle": cycle, "agent": "failure_risk", "error": str(exc)})

            points_done += 1
            now = time.perf_counter()
            if now - last_report >= _PROGRESS_INTERVAL_SECONDS:
                elapsed = now - started
                fraction = points_done / total_points if total_points else 0.0
                eta = (elapsed / fraction - elapsed) if fraction > 0 else 0.0
                logger.info(
                    "RUL/Failure-Risk batch: %5.1f%% | %d/%d points | "
                    "engine %d/%d | %.1f points/s | elapsed %s | ETA %s%s",
                    100.0 * fraction,
                    points_done,
                    total_points,
                    engines_done + 1,
                    total_engines,
                    points_done / elapsed if elapsed > 0 else 0.0,
                    _format_duration(elapsed),
                    _format_duration(eta),
                    f" | {len(errors)} error(s)" if errors else "",
                )
                last_report = now

        engines_done += 1

    total_elapsed = time.perf_counter() - started
    logger.info(
        "RUL/Failure-Risk batch complete: %d point(s) over %d engine(s) in %s "
        "(%.1f points/s), %d RUL row(s), %d risk row(s), %d error(s)%s.",
        points_done,
        engines_done,
        _format_duration(total_elapsed),
        points_done / total_elapsed if total_elapsed > 0 else 0.0,
        len(rul_rows),
        len(risk_rows),
        len(errors),
        (
            f"; {engines_skipped} engine(s) skipped as shorter than the "
            f"required window"
            if engines_skipped
            else ""
        ),
    )

    rul_path = output_dir / "rul_predictions.csv"
    risk_path = output_dir / "failure_risk_predictions.csv"
    pd.DataFrame(rul_rows).to_csv(rul_path, index=False)
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False)

    return {
        "status": "success" if not errors else "partial",
        "rul_rows": len(rul_rows),
        "failure_risk_rows": len(risk_rows),
        "rul_predictions_csv": str(rul_path),
        "failure_risk_predictions_csv": str(risk_path),
        "errors": errors[:20],
        "error_count": len(errors),
    }
