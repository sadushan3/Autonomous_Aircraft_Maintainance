"""
Data cleaning and quality scoring.

The cleaner does four things, in this order:

1. **Type and key hygiene** -- coerce numerics, sort by ``(unit, cycle)``, and
   remove duplicate key rows.
2. **Constant-sensor detection** -- several C-MAPSS sensors are perfectly flat
   in FD001 and carry no information.  They are recorded to
   ``models/dropped_sensors.json`` and dropped only when config says so.  The
   surviving list always comes from :func:`resolve_active_sensors`, so column
   alignment between training and inference is preserved by construction.
3. **Missing / non-finite repair** -- forward-fill *within* an engine (never
   across engines, which would smear one engine's history onto another), then
   fall back to a per-column median.
4. **Per-row data quality scoring** -- how much repair a row needed becomes a
   ``data_quality_score`` in [0, 1] that flows all the way through to the
   confidence estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    detect_constant_sensors,
    get_feature_columns,
    get_identity_columns,
    get_op_setting_columns,
    get_sensor_columns,
    resolve_active_sensors,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)

#: Name of the per-row quality column produced by this stage.
DATA_QUALITY_COLUMN = "data_quality_score"

#: Name of the per-row repair-count column.
IMPUTED_COUNT_COLUMN = "imputed_value_count"


class DataCleaner:
    """Cleans raw C-MAPSS data and annotates it with quality information."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the cleaner.

        Args:
            config: Configuration supplying the constant-sensor policy and
                output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.dropped_sensors: List[str] = []
        self.active_sensors: List[str] = get_sensor_columns()

    # ------------------------------------------------------------------ #
    # Individual cleaning steps
    # ------------------------------------------------------------------ #
    def deduplicate(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate ``(unit, cycle)`` rows, keeping the first.

        Duplicate keys would break every downstream merge, which validates
        one-to-one joins.

        Args:
            frame: Input data.

        Returns:
            A ``(deduplicated_frame, removed_count)`` tuple.
        """
        keys = get_identity_columns()
        before = len(frame)
        result = frame.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        removed = before - len(result)
        if removed:
            self.logger.warning("Removed %d duplicate (unit, cycle) row(s)", removed)
        return result, removed

    def find_constant_sensors(self, frame: pd.DataFrame) -> List[str]:
        """Detect flat sensors across the whole dataset.

        Args:
            frame: Input data.

        Returns:
            The names of constant sensors, in canonical order.
        """
        return detect_constant_sensors(
            frame,
            std_epsilon=self.config.data.constant_sensor_std_eps,
            sensors=get_sensor_columns(),
        )

    def repair_values(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Replace non-finite and missing feature values.

        Repair is engine-scoped: a gap is filled from the same engine's earlier
        cycles first, then its later cycles, and only then from the global
        column median.  Filling across engine boundaries would import one
        engine's operating point into another's trajectory.

        Args:
            frame: Input data.

        Returns:
            A ``(repaired_frame, counts)`` tuple where counts reports how many
            non-finite values and how many missing values were fixed.
        """
        result = frame.copy()
        feature_columns = [c for c in get_feature_columns() if c in result.columns]

        non_finite_mask = ~np.isfinite(result[feature_columns].to_numpy(dtype=float))
        n_non_finite = int(non_finite_mask.sum())
        if n_non_finite:
            self.logger.warning("Replacing %d non-finite feature value(s)", n_non_finite)
            result[feature_columns] = result[feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )

        missing_before = result[feature_columns].isna()
        n_missing = int(missing_before.to_numpy().sum())

        # Per-row repair count feeds the quality score.
        result[IMPUTED_COUNT_COLUMN] = missing_before.sum(axis=1).astype(int)

        if n_missing:
            self.logger.warning(
                "Imputing %d missing feature value(s) within engine trajectories",
                n_missing,
            )
            grouped = result.groupby("unit", sort=False)[feature_columns]
            result[feature_columns] = grouped.transform(
                lambda column: column.ffill().bfill()
            )
            # Anything still missing means an entire engine lacked the channel.
            still_missing = result[feature_columns].isna().to_numpy().sum()
            if still_missing:
                medians = result[feature_columns].median()
                result[feature_columns] = result[feature_columns].fillna(medians)
                result[feature_columns] = result[feature_columns].fillna(0.0)
                self.logger.warning(
                    "%d value(s) had no in-engine neighbour; filled from the "
                    "column median",
                    int(still_missing),
                )

        return result, {"non_finite": n_non_finite, "missing": n_missing}

    def apply_constant_sensor_policy(
        self, frame: pd.DataFrame, constant_sensors: List[str]
    ) -> pd.DataFrame:
        """Drop flat sensors when configured to, and record the active set.

        The active sensor list is always derived through
        :func:`resolve_active_sensors` so it stays in canonical order.  It is
        persisted to ``models/dropped_sensors.json`` because every later stage
        -- twins, detectors, explainers -- must agree on exactly which columns
        are in play.

        Args:
            frame: Input data.
            constant_sensors: Sensors detected as flat.

        Returns:
            The frame, with flat sensors removed when the policy says so.
        """
        if self.config.data.drop_constant_sensors and constant_sensors:
            self.dropped_sensors = list(constant_sensors)
            result = frame.drop(columns=constant_sensors, errors="ignore")
            self.logger.info(
                "Dropped %d constant sensor(s): %s",
                len(constant_sensors),
                constant_sensors,
            )
        else:
            self.dropped_sensors = []
            result = frame
            if constant_sensors:
                self.logger.info(
                    "Retaining %d constant sensor(s) because drop_constant_sensors "
                    "is disabled: %s",
                    len(constant_sensors),
                    constant_sensors,
                )

        self.active_sensors = resolve_active_sensors(self.dropped_sensors)
        self._persist_sensor_policy(constant_sensors)
        return result

    def apply_persisted_sensor_policy(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop exactly the sensors dropped at training time, without persisting.

        Unlike :meth:`apply_constant_sensor_policy`, this never writes
        ``dropped_sensors.json``. It reads the existing policy and applies it,
        so a frame scored through inference is guaranteed to expose the same
        active-sensor set every twin, detector and calibration was fit
        against, regardless of what looks constant in this particular batch.

        Args:
            frame: Input data.

        Returns:
            The frame, with the persisted set of dropped sensors removed.
        """
        active = self.load_sensor_policy()
        all_sensors = get_sensor_columns()
        self.dropped_sensors = [s for s in all_sensors if s not in active]
        self.active_sensors = active
        result = frame.drop(columns=self.dropped_sensors, errors="ignore")
        if self.dropped_sensors:
            self.logger.info(
                "Applied persisted sensor policy (inference): dropped %d "
                "sensor(s) fit at training time: %s",
                len(self.dropped_sensors),
                self.dropped_sensors,
            )
        return result

    def _persist_sensor_policy(self, constant_sensors: List[str]) -> Path:
        """Write the sensor policy so later stages align their columns.

        Args:
            constant_sensors: Everything detected as flat.

        Returns:
            The path written.
        """
        payload = {
            "detected_constant_sensors": constant_sensors,
            "dropped_sensors": self.dropped_sensors,
            "active_sensors": self.active_sensors,
            "drop_policy_enabled": bool(self.config.data.drop_constant_sensors),
            "std_epsilon": self.config.data.constant_sensor_std_eps,
        }
        return atomic_write_json(
            payload, self.config.paths.models / self.config.paths.dropped_sensors_json
        )

    def score_quality(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach a per-row data-quality score in [0, 1].

        The score starts at 1 and is reduced in proportion to how many of the
        row's feature values had to be repaired.  It travels with the row all
        the way to the confidence estimator, so a heavily patched reading
        yields a visibly less confident assessment.

        Args:
            frame: Repaired data carrying the imputed-count column.

        Returns:
            The frame with ``data_quality_score`` added.
        """
        result = frame.copy()
        n_features = max(
            len([c for c in get_feature_columns() if c in result.columns]), 1
        )
        imputed = result.get(
            IMPUTED_COUNT_COLUMN, pd.Series(np.zeros(len(result)), index=result.index)
        )
        penalty = (imputed.astype(float) / float(n_features)).clip(0.0, 1.0)
        result[DATA_QUALITY_COLUMN] = (1.0 - penalty).clip(0.0, 1.0)
        return result

    # ------------------------------------------------------------------ #
    # Stage orchestration
    # ------------------------------------------------------------------ #
    def clean(
        self, frame: pd.DataFrame, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the full cleaning sequence.

        Args:
            frame: Raw data as produced by the loader.
            fit_policy: Derive the constant-sensor policy from this frame and
                persist it (training). When False, the persisted policy from
                ``dropped_sensors.json`` is applied as-is and never rewritten
                -- the caller is scoring new data through an already-trained
                pipeline, and every downstream model (twins, detectors,
                calibrations) was fit against a specific active-sensor set.
                Re-deriving it from whatever happens to be constant in a
                fresh upload -- which can be a single engine, or a different
                subset mix -- would silently invalidate every one of those
                artifacts for every future run, not just this one.

        Returns:
            A ``(cleaned_frame, report)`` tuple.
        """
        rows_in = len(frame)
        result = frame.sort_values(get_identity_columns()).reset_index(drop=True)

        result, duplicates_removed = self.deduplicate(result)
        result, repair_counts = self.repair_values(result)
        if fit_policy:
            constant_sensors = self.find_constant_sensors(result)
            result = self.apply_constant_sensor_policy(result, constant_sensors)
        else:
            constant_sensors = self.find_constant_sensors(result)
            result = self.apply_persisted_sensor_policy(result)
        result = self.score_quality(result)

        result = self._clip_rul(result)

        report: Dict[str, Any] = {
            "rows_in": int(rows_in),
            "rows_out": int(len(result)),
            "duplicate_rows_removed": int(duplicates_removed),
            "missing_values_imputed": int(repair_counts["missing"]),
            "non_finite_values_fixed": int(repair_counts["non_finite"]),
            "constant_sensors_detected": constant_sensors,
            "constant_sensors_dropped": self.dropped_sensors,
            "active_sensors": self.active_sensors,
            "mean_data_quality_score": float(result[DATA_QUALITY_COLUMN].mean()),
            "engines": int(result["unit"].nunique()),
        }
        self.logger.info("Cleaning report: %s", report)
        return result, report

    def _clip_rul(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the standard C-MAPSS RUL ceiling.

        Degradation is not observable in the sensors during the long healthy
        plateau, so the community convention is to cap RUL at 125-130 cycles.
        The uncapped value is retained as ``RUL_true`` for evaluation.

        Args:
            frame: Data carrying a RUL column.

        Returns:
            The frame with RUL clipped and the original preserved.
        """
        if RUL_COLUMN not in frame.columns:
            return frame
        result = frame.copy()
        cap = int(self.config.data.rul_cap)
        result["RUL_true"] = result[RUL_COLUMN].astype(int)
        result[RUL_COLUMN] = result[RUL_COLUMN].clip(upper=cap).astype(int)
        self.logger.info("Clipped %s at %d cycles (RUL_true retains the raw value)", RUL_COLUMN, cap)
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``cleaned_data.csv`` atomically.

        Args:
            frame: Cleaned data.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.cleaned_data_csv
        )
        verify_identity_columns(frame, context="cleaned_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back a previously written ``cleaned_data.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.cleaned_data_csv
        )
        return read_csv(target, required_columns=get_identity_columns(), stage="cleaner")

    def load_sensor_policy(self) -> List[str]:
        """Read the persisted active-sensor list.

        Every stage after cleaning calls this so they all agree on which
        sensors exist, without re-deriving the policy or sniffing columns.

        Returns:
            The active sensor list, defaulting to all 21 when no policy file
            has been written yet.
        """
        payload = read_json(
            self.config.paths.models / self.config.paths.dropped_sensors_json,
            default=None,
        )
        if not payload:
            return get_sensor_columns()
        return resolve_active_sensors(payload.get("dropped_sensors", []))

    def run(
        self, frame: Optional[pd.DataFrame] = None, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the cleaning stage.

        Args:
            frame: Optional in-memory input.  When omitted ``raw_data.csv`` is
                read from disk.
            fit_policy: See :meth:`clean`. False for inference over new data.

        Returns:
            A ``(cleaned_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.raw_data_csv),
                required_columns=get_identity_columns(),
                stage="cleaner",
            )
        cleaned, report = self.clean(source, fit_policy=fit_policy)
        report["output_path"] = str(self.save(cleaned))
        return cleaned, report


def get_active_sensors(config: Optional[Config] = None) -> List[str]:
    """Return the active sensor list agreed by the cleaning stage.

    This is the accessor every downstream module uses; it never sniffs columns,
    it reads the recorded policy and resolves it through the canonical resolver.

    Args:
        config: Optional configuration override.

    Returns:
        The active sensor columns in canonical order.
    """
    return DataCleaner(config).load_sensor_policy()


def main() -> None:
    """Run the cleaner standalone against ``raw_data.csv``."""
    config = get_config()
    logger = get_logger(__name__, config)
    cleaner = DataCleaner(config)
    cleaned, report = cleaner.run()
    logger.info("cleaned_data.csv written: %d rows", report["rows_out"])
    logger.info("Active sensors (%d): %s", len(report["active_sensors"]), report["active_sensors"])
    print(cleaned.head(8).to_string())


if __name__ == "__main__":
    main()
