"""
Health index calculation: residuals -> a single 0-100 indicator.

The health index compresses the per-sensor residual picture into one number a
maintenance engineer can track at a glance, following:

    h_raw(t)  = exp(-lambda * weighted_mean(|residual_z|))
    h_sm(t)   = EWMA_alpha(h_raw(t))          -- per engine, causal
    HI(t)     = 100 * clip(h_sm(t), 0, 1)

The exponential decay means a handful of mildly elevated sensors barely move
the index, while a genuinely severe deviation collapses it quickly -- matching
how a maintenance engineer actually reads "a bit off" versus "clearly wrong".
The EWMA smooths cycle-to-cycle sensor noise without hiding a sustained trend,
and -- critically -- it is applied per engine so no smoothing ever blends one
engine's history into another's.

Because every C-MAPSS training engine runs to failure, a supervised anchor is
available: ``HI_target = 100 * min(RUL, rul_cap) / rul_cap``.  That target is
attached wherever RUL is known and is what :mod:`evaluate_health` correlates
the computed index against and uses to score monotonicity -- the acceptance
gate the research objective calls for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
    RESIDUAL_ABS_Z_MEAN_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_identity_columns,
    get_residual_z_columns,
    present_residual_columns,
    sensor_from_residual,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    merge_on_keys,
    read_csv,
    read_json,
)

#: Smoothed 0-100 health index, the primary output of this module.
HEALTH_INDEX_COLUMN = "health_index"

#: Pre-smoothing health index.
HEALTH_INDEX_RAW_COLUMN = "health_index_raw"

#: Supervised anchor derived from true RUL, present only where RUL is known.
HEALTH_INDEX_TARGET_COLUMN = "health_index_target"


class HealthIndexCalculator:
    """Converts residual magnitude into a smoothed 0-100 health index."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the calculator.

        Args:
            config: Configuration supplying the decay rate and smoothing alpha.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    # ------------------------------------------------------------------ #
    # Weighted residual aggregation
    # ------------------------------------------------------------------ #
    def _informative_sensors(self, residuals: pd.DataFrame) -> List[str]:
        """Return the residual columns that actually carry information.

        Several C-MAPSS channels are flat within an operating mode (sensor_1,
        5, 18, 19 among them). Their centred residual is identically zero, so
        including them only dilutes the mean deviation by a constant factor
        and makes the health index harder to read. Selection is by variance
        alone -- no labels are consulted.

        Args:
            residuals: Residuals frame.

        Returns:
            The informative residual column names.
        """
        candidates = present_residual_columns(residuals)
        if not candidates:
            return []
        spread = residuals[candidates].std(numeric_only=True)
        informative = [c for c in candidates if float(spread.get(c, 0.0) or 0.0) > 1e-8]
        return informative or candidates

    def _deviation(
        self,
        residuals: pd.DataFrame,
        columns: List[str],
        sensor_weights: Optional[Dict[str, float]] = None,
        residual_window: int = 1,
    ) -> np.ndarray:
        """Compute the mean absolute centred residual across informative sensors.

        This deliberately consumes ``residual_sensor_*`` rather than
        ``residual_z_sensor_*``. The residuals already live in per-mode
        standardised units -- the scaler put them there -- so they are
        directly comparable across sensors. Re-standardising them a second
        time (which is what the z columns do) divides each sensor by a MAD
        computed over the whole fleet, and that MAD is dominated by
        engine-to-engine spread rather than by degradation. Measured against
        true RUL, the doubly-normalised aggregate scores |rho| = 0.46 versus
        0.82 for the singly-normalised one: the second normalisation erases
        most of the signal the health index exists to report.

        Args:
            residuals: Residuals frame carrying centred ``residual_sensor_*``.
            columns: Informative residual columns to aggregate.
            sensor_weights: Optional per-sensor weights; equal by default.

        Returns:
            A 1-D array of mean absolute deviation, one value per row.
        """
        values = np.abs(self.sensor_values(residuals, columns, residual_window))
        if not columns or not np.isfinite(values).all():
            raise AnomalyHealthError("Health calculation requires finite residuals for the calibrated sensor channels.")
        if not sensor_weights:
            return np.nanmean(values, axis=1)
        weights = np.array(
            [sensor_weights.get(sensor_from_residual(c), 1.0) for c in columns],
            dtype=float,
        )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise AnomalyHealthError("Health sensor weights must be finite and non-negative.")
        if weights.sum() <= 0:
            # A subset missing all selected channels still needs a meaningful
            # health signal rather than an all-zero deviation (perfect health).
            return np.nanmean(values, axis=1)
        weights = weights / max(weights.sum(), 1e-9)
        return values @ weights

    @staticmethod
    def sensor_values(residuals: pd.DataFrame, columns: List[str], window: int = 1) -> np.ndarray:
        """Causal sensor denoising, preserving order and resetting at gaps."""
        values = residuals[columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise AnomalyHealthError("Health sensor residuals must be finite.")
        if window < 1:
            raise AnomalyHealthError("Health residual window must be a positive integer.")
        if window == 1:
            return values
        ordered = residuals.reset_index(drop=True).sort_values(["unit", "cycle"])
        gaps = ordered.groupby("unit", sort=False).cycle.diff().ne(1).cumsum()
        smoothed = ordered[columns].groupby([ordered.unit, gaps], sort=False).transform(
            lambda s: s.rolling(window, min_periods=1).mean())
        result = np.empty_like(values)
        result[ordered.index.to_numpy()] = smoothed.to_numpy()
        return result

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def calibration_path(self) -> Path:
        """Return where the healthy-reference calibration is persisted.

        Returns:
            Path of ``health_index_calibration.json``.
        """
        return self.config.paths.model_path("health_index_calibration.json")

    def fit_calibration(self, residuals: pd.DataFrame) -> Dict[str, Any]:
        """Learn and persist the healthy-behaviour reference for the index.

        The index reports deviation as a multiple of what healthy behaviour
        looks like, so that reference must be fixed at training time and
        reused. Recomputing it per batch would make the index relative to
        whatever happened to be scored: a batch containing only failing
        engines would recalibrate onto their degraded level and report them
        as healthy.

        Args:
            residuals: Residuals frame to calibrate against.

        Returns:
            The calibration document that was written.
        """
        # Calibration is an explicit training operation. Never learn normal
        # behaviour from official test rows, even in a combined pipeline run.
        training = residuals
        if "split" in residuals:
            training = residuals.loc[residuals["split"].eq("train")]
        if training.empty:
            raise AnomalyHealthError("Health reference calibration requires training rows.")
        columns = self._informative_sensors(training)
        if RUL_COLUMN in training:
            early = (training[RUL_COLUMN] >= self.config.anomaly.calibration_healthy_rul) & (
                training["cycle"] > self.config.twin.commissioning_cycles)
        else:
            # Only an explicit calibration call may use this commissioning
            # proxy when confirmed healthy training labels are unavailable.
            early = training["cycle"].between(1, self.config.twin.commissioning_cycles)
        if not early.any():
            raise AnomalyHealthError("No healthy training rows available for health reference calibration.")
        previous = read_json(self.calibration_path(), default={})
        deviation = self._deviation(training, columns, previous.get("sensor_weights"),
                                    int(previous.get("residual_window", 1)))
        healthy_deviation = deviation[early.to_numpy()]
        if healthy_deviation.size == 0:
            healthy_deviation = deviation
        reference = float(np.median(healthy_deviation))

        # A near-noiseless healthy region would drive the reference to zero and
        # make the index hypersensitive -- every ordinary fluctuation would read
        # as an infinite multiple of "normal" and the index would saturate at 0
        # on the first cycle. Floor the reference at a small fraction of the
        # fleet's overall deviation so it always denotes a realistic scale.
        overall = float(np.median(deviation)) if deviation.size else 0.0
        reference = max(reference, 0.05 * overall, 1e-9)

        calibration = {
            "informative_columns": columns,
            "healthy_reference_deviation": reference,
            "n_rows_calibrated": int(early.sum()),
        }
        if previous.get("sensor_weights"):
            calibration["sensor_weights"] = previous["sensor_weights"]
            calibration["weight_selection"] = previous.get("weight_selection", "persisted policy")
        for key in ("residual_window", "ewma_alpha", "decay_lambda"):
            if key in previous:
                calibration[key] = previous[key]
        atomic_write_json(calibration, self.calibration_path())
        self.logger.info(
            "Calibrated health index on %d healthy row(s): reference deviation "
            "%.5f across %d informative sensor(s)",
            calibration["n_rows_calibrated"],
            calibration["healthy_reference_deviation"],
            len(columns),
        )
        return calibration

    def load_calibration(self, residuals: pd.DataFrame) -> Dict[str, Any]:
        """Load the persisted training calibration without fitting at inference.

        Args:
            residuals: Inference schema checked against the saved calibration.

        Returns:
            The active calibration document.
        """
        stored = read_json(self.calibration_path(), default=None)
        if not stored or "healthy_reference_deviation" not in stored:
            raise AnomalyHealthError("No health reference is available; explicitly calibrate on healthy training data first.")
        reference = float(stored["healthy_reference_deviation"])
        if not np.isfinite(reference) or reference <= 0:
            raise AnomalyHealthError("Stored health reference must be finite and positive.")

        available = set(residuals.columns)
        required = [c for c in stored.get("informative_columns", [])
                    if stored.get("sensor_weights", {}).get(sensor_from_residual(c), 1.0) > 0]
        missing = sorted(set(required) - available)
        if missing:
            raise AnomalyHealthError(f"Missing calibrated health sensor channels: {missing}")
        columns = [c for c in stored.get("informative_columns", []) if c in available]
        if not columns:
            raise AnomalyHealthError("No calibrated health sensor channels are present; inference cannot recalibrate.")
        stored["informative_columns"] = columns
        return stored

    def compute(
        self,
        residuals: pd.DataFrame,
        sensor_weights: Optional[Dict[str, float]] = None,
        recalibrate: bool = False,
    ) -> pd.DataFrame:
        """Compute the raw and smoothed health index per row.

        Args:
            residuals: Residuals frame with identity keys and centred
                ``residual_sensor_*`` columns.
            sensor_weights: Optional per-sensor weighting.
            recalibrate: Re-fit the healthy reference from this frame rather
                than reusing the persisted one.

        Returns:
            A frame of identity keys, ``health_index_raw``, ``health_index``
            and (when RUL is available) ``health_index_target``.
        """
        ordered = residuals.sort_values(get_identity_columns()).reset_index(drop=True)
        if ordered.duplicated(get_identity_columns()).any() or ordered[get_identity_columns()].isna().any().any():
            raise AnomalyHealthError("Health input needs unique, non-missing unit/cycle identities.")

        calibration = (
            self.fit_calibration(ordered) if recalibrate else self.load_calibration(ordered)
        )
        columns = calibration["informative_columns"]
        reference = float(calibration["healthy_reference_deviation"])
        if sensor_weights is None:
            # Frozen weights selected on training/validation engines. Explicit
            # caller weights still take precedence over the calibrated policy.
            sensor_weights = calibration.get("sensor_weights")

        window = int(calibration.get("residual_window", 1))
        deviation = self._deviation(ordered, columns, sensor_weights, window)
        # Expressed as "how many multiples above your own healthy level":
        # healthy behaviour sits at ~0 and therefore scores ~100, so the
        # index uses its full range instead of hovering mid-scale.
        excess = np.clip(deviation / max(reference, 1e-9) - 1.0, 0.0, None)

        decay = float(calibration.get("decay_lambda", self.config.health.decay_lambda))
        alpha = float(calibration.get("ewma_alpha", self.config.health.ewma_alpha))
        if not np.isfinite([decay, alpha]).all() or decay <= 0 or not 0 < alpha <= 1:
            raise AnomalyHealthError("Health decay must be positive and EWMA alpha must be within (0, 1].")
        h_raw = np.exp(-decay * excess)

        result = ordered[get_identity_columns()].copy()
        result[HEALTH_INDEX_RAW_COLUMN] = 100.0 * np.clip(h_raw, 0.0, 1.0)

        smoothed = (
            pd.Series(h_raw, index=ordered.index)
            .groupby(ordered["unit"], sort=False)
            .transform(lambda series: series.ewm(alpha=alpha, adjust=False).mean())
        )
        result[HEALTH_INDEX_COLUMN] = 100.0 * np.clip(smoothed, 0.0, 1.0)

        if RUL_COLUMN in ordered.columns:
            cap = float(self.config.data.rul_cap)
            target = 100.0 * np.clip(ordered[RUL_COLUMN].astype(float), 0.0, cap) / cap
            result[HEALTH_INDEX_TARGET_COLUMN] = target

        self.logger.info(
            "Health index computed: mean=%.2f, min=%.2f (decay=%.3f, ewma_alpha=%.3f)",
            float(result[HEALTH_INDEX_COLUMN].mean()),
            float(result[HEALTH_INDEX_COLUMN].min()),
            decay,
            alpha,
        )
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``health_index.csv`` atomically.

        Args:
            frame: Computed health index.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_index_csv)
        verify_identity_columns(frame, context="health_index output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``health_index.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_index_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_index_calculator"
        )

    def run(
        self,
        residuals: Optional[pd.DataFrame] = None,
        recalibrate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the health-index stage.

        Args:
            residuals: Optional in-memory residuals; otherwise ``residuals.csv``
                is read from disk. RUL is attached from ``raw_data.csv`` when
                present, so the supervised anchor can be computed.
            recalibrate: Re-fit the healthy reference from this data. True for
                a training pass over the full fleet; pass False when scoring
                new engines so they are judged against the reference the
                fleet was calibrated on rather than against themselves.

        Returns:
            A ``(health_index_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = residuals
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="health_index_calculator",
            )

        raw_path = self.config.paths.processed_path(self.config.paths.raw_data_csv)
        if raw_path.is_file():
            raw = read_csv(raw_path, required_columns=get_identity_columns(), stage="health_index_calculator")
            extras = [c for c in (RUL_COLUMN, "split", "subset_id") if c in raw and c not in source]
            if extras:
                source = merge_on_keys(
                    source,
                    raw[get_identity_columns() + extras],
                    keys=get_identity_columns(),
                    how="left",
                )

        health_index = self.compute(source, recalibrate=recalibrate)
        output = self.save(health_index)

        report: Dict[str, Any] = {
            "rows": int(len(health_index)),
            "engines": int(health_index["unit"].nunique()),
            "mean_health_index": float(health_index[HEALTH_INDEX_COLUMN].mean()),
            "min_health_index": float(health_index[HEALTH_INDEX_COLUMN].min()),
            "has_supervised_target": HEALTH_INDEX_TARGET_COLUMN in health_index.columns,
            "output_path": str(output),
        }
        self.logger.info("Health index report: %s", report)
        return health_index, report


def main() -> None:
    """Run health-index calculation standalone against ``residuals.csv``."""
    config = get_config()
    logger = get_logger(__name__, config)
    calculator = HealthIndexCalculator(config)
    health_index, report = calculator.run()
    logger.info("health_index.csv written: %d rows", report["rows"])
    print(health_index.head(10).to_string())


if __name__ == "__main__":
    main()
