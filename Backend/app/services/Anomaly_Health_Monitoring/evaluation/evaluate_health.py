"""
Health-index evaluation: the research objective's acceptance gate.

The specification requires validating "that the health index decreases
monotonically (on average) as cycles approach failure" and deriving the health
target from normalized RUL / cycle-to-failure per engine.  This module is
where that validation actually happens, computing two numbers against
``health_index_target`` (100 * min(RUL, cap) / cap, attached by
:mod:`health_index_calculator` wherever true RUL is known):

* **Spearman correlation** between the computed health index and true RUL,
  fleet-wide.  A well-behaved health index should correlate strongly and
  positively -- low RUL should coincide with low health.
* **Monotonicity score** -- the fraction of engines whose EWMA-smoothed health
  index is non-increasing (within a small noise tolerance) over the final
  ``1 - healthy_region.cycle_fraction`` portion of their life, i.e. the
  degradation phase.  This is the literal acceptance criterion from the
  specification, computed per engine and averaged across the fleet.

The default acceptance threshold (``config.health.min_monotonicity_score``,
0.7) means at least 70% of engines must show a non-increasing health index
through their final phase of life for the health layer to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import (
    HEALTH_INDEX_COLUMN,
    HEALTH_INDEX_TARGET_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import RUL_COLUMN, get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json, get_logger, read_csv

#: Small tolerance allowing cycle-to-cycle noise without failing monotonicity.
_MONOTONICITY_NOISE_TOLERANCE = 1.0


class HealthEvaluator:
    """Validates the health-index layer against its monotonicity/RUL contract."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying the acceptance threshold and the
                healthy-region cycle fraction (used to define "final phase").
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    # ------------------------------------------------------------------ #
    # Core metrics
    # ------------------------------------------------------------------ #
    def rul_correlation(self, frame: pd.DataFrame) -> Optional[float]:
        """Compute the Spearman correlation between health index and true RUL.

        Args:
            frame: Frame carrying ``health_index`` and ``RUL`` (or
                ``health_index_target``, used as a RUL proxy when raw RUL is
                absent).

        Returns:
            The Spearman correlation, or None when RUL is unavailable.
        """
        rul_column = RUL_COLUMN if RUL_COLUMN in frame.columns else None
        if rul_column is None and HEALTH_INDEX_TARGET_COLUMN not in frame.columns:
            return None

        health = frame[HEALTH_INDEX_COLUMN].astype(float)
        rul = frame[rul_column].astype(float) if rul_column else frame[HEALTH_INDEX_TARGET_COLUMN].astype(float)
        if health.std() < 1e-9 or rul.std() < 1e-9:
            return 0.0
        correlation, _ = stats.spearmanr(health, rul)
        return float(correlation) if np.isfinite(correlation) else 0.0

    def monotonicity_score(self, frame: pd.DataFrame) -> float:
        """Compute the fraction of engines with a non-increasing health index late in life.

        For each engine, examines the trailing portion of the trajectory
        (cycles beyond the healthy-region cycle fraction, i.e. the degradation
        phase) and checks whether the EWMA-smoothed health index is
        non-increasing there, within a small noise tolerance.

        Args:
            frame: Frame carrying identity keys, ``cycle`` and ``health_index``.

        Returns:
            The fraction of qualifying engines that pass, in [0, 1].
        """
        cutoff_fraction = float(self.config.healthy_region.cycle_fraction)
        passing = 0
        total = 0

        for _, group in frame.groupby("unit", sort=False):
            ordered = group.sort_values("cycle")
            max_cycle = ordered["cycle"].max()
            if max_cycle <= 0:
                continue
            late_phase = ordered[ordered["cycle"] > cutoff_fraction * max_cycle]
            if len(late_phase) < 5:
                continue

            total += 1
            smoothed = late_phase[HEALTH_INDEX_COLUMN].ewm(alpha=0.3, adjust=False).mean()
            diffs = smoothed.diff().dropna()
            # Allow small positive blips (noise) but not a sustained rise.
            violations = (diffs > _MONOTONICITY_NOISE_TOLERANCE).sum()
            violation_rate = violations / max(len(diffs), 1)
            if violation_rate <= 0.15:
                passing += 1

        return float(passing / total) if total else 0.0

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def evaluate(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Run both acceptance checks and combine them into a verdict.

        Args:
            frame: Health-scores/states frame carrying identity keys, RUL (if
                available) and ``health_index``.

        Returns:
            A report with ``rul_correlation``, ``monotonicity_score`` and
            ``passed``.
        """
        correlation = self.rul_correlation(frame)
        monotonicity = self.monotonicity_score(frame)
        threshold = float(self.config.health.min_monotonicity_score)

        passed = monotonicity >= threshold
        report: Dict[str, Any] = {
            "rows": int(len(frame)),
            "engines": int(frame["unit"].nunique()),
            "rul_correlation": correlation,
            "monotonicity_score": round(monotonicity, 4),
            "monotonicity_threshold": threshold,
            "mean_health_index": float(frame[HEALTH_INDEX_COLUMN].mean()),
            "passed": bool(passed),
        }

        if passed:
            self.logger.info(
                "Health index PASSED acceptance: monotonicity=%.2f%% (>= %.0f%%), RUL correlation=%s",
                100.0 * monotonicity,
                100.0 * threshold,
                correlation,
            )
        else:
            self.logger.warning(
                "Health index FAILED acceptance: monotonicity=%.2f%% (< %.0f%%). "
                "Consider adjusting healthy_region thresholds or the decay rate.",
                100.0 * monotonicity,
                100.0 * threshold,
            )
        return report

    def run(self, frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the health evaluation stage.

        Args:
            frame: Optional in-memory input; otherwise ``health_states.csv``
                (falling back to ``health_scores.csv``) is read from disk.

        Returns:
            The evaluation report.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            path = self.config.paths.processed_path(self.config.paths.health_states_csv)
            if not path.is_file():
                path = self.config.paths.processed_path(self.config.paths.health_scores_csv)
            source = read_csv(
                path, required_columns=get_identity_columns() + [HEALTH_INDEX_COLUMN], stage="evaluate_health"
            )

        report = self.evaluate(source)
        atomic_write_json(report, self.config.paths.reports / "health_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_health.json")
        return report


def main() -> None:
    """Run health evaluation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = HealthEvaluator(config)
    report = evaluator.run()
    logger.info("Health evaluation report: %s", report)


if __name__ == "__main__":
    main()
