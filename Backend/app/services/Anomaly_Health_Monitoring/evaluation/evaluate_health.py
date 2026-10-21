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
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import engine_columns


_MONOTONICITY_NOISE_TOLERANCE = 1.0


class HealthEvaluator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def rul_correlation(self, frame: pd.DataFrame) -> Optional[float]:


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


        cutoff_fraction = float(self.config.healthy_region.cycle_fraction)
        passing = 0
        total = 0

        for _, group in frame.groupby(engine_columns(frame), sort=False):
            ordered = group.sort_values("cycle")
            max_cycle = ordered["cycle"].max()
            if max_cycle <= 0:
                continue
            late_phase = ordered[ordered["cycle"] > cutoff_fraction * max_cycle]
            if len(late_phase) < 5:
                continue

            total += 1


            diffs = late_phase[HEALTH_INDEX_COLUMN].diff().dropna()

            violations = (diffs > _MONOTONICITY_NOISE_TOLERANCE).sum()
            violation_rate = violations / max(len(diffs), 1)
            if violation_rate <= 0.15:
                passing += 1

        return float(passing / total) if total else 0.0

    def trajectory_metrics(self, frame: pd.DataFrame) -> Dict[str, Any]:

        drops, correlations, flat = [], [], 0
        eligible = 0
        for _, group in frame.groupby(engine_columns(frame), sort=True):
            ordered = group.sort_values("cycle")
            values = ordered[HEALTH_INDEX_COLUMN].to_numpy(float)
            if len(values) < 10:
                continue
            flat += bool(np.ptp(values) < 1.0)


            if RUL_COLUMN in ordered and ordered[RUL_COLUMN].min() > 30:
                continue
            eligible += 1
            window = min(20, max(len(values) // 5, 1))
            drops.append(float(np.median(values[:window]) - np.median(values[-window:])))
            if RUL_COLUMN in ordered and np.std(values) > 1e-9 and ordered[RUL_COLUMN].std() > 1e-9:
                correlations.append(float(stats.spearmanr(values, ordered[RUL_COLUMN]).statistic))
        engines = frame.groupby(engine_columns(frame)).ngroups
        return {"progression_engines": eligible,
                "engines_with_health_drop_at_least_5": sum(d >= 5 for d in drops),
                "degradation_fraction": float(np.mean(np.asarray(drops) >= 5)) if drops else None,
                "median_health_drop": float(np.median(drops)) if drops else None,
                "flat_engine_fraction": flat / engines if engines else None,
                "median_within_engine_rul_correlation": float(np.median(correlations)) if correlations else None,
                "additional_evaluation_smoothing": False}


    def evaluate(self, frame: pd.DataFrame) -> Dict[str, Any]:


        finite = np.isfinite(frame[HEALTH_INDEX_COLUMN].to_numpy(float))
        in_range = frame[HEALTH_INDEX_COLUMN].between(0, 100).to_numpy()
        valid = frame.loc[finite & in_range].copy()
        correlation = self.rul_correlation(valid) if len(valid) else None
        monotonicity = self.monotonicity_score(valid)
        trajectories = self.trajectory_metrics(valid)
        threshold = float(self.config.health.min_monotonicity_score)

        acceptance = {
            "finite_bounded_health": bool(len(frame) and (finite & in_range).all()),
            "monotonicity": monotonicity >= threshold,
            "rul_proxy_correlation": correlation is not None and correlation >= 0.3,
            "detects_degradation": (trajectories["degradation_fraction"] or 0) >= 0.5,
        }
        passed = all(acceptance.values())
        report: Dict[str, Any] = {
            "rows": int(len(frame)),
            "engines": int(frame.groupby(engine_columns(frame)).ngroups),
            "evaluation_scope": "Current batch; RUL correlation is a condition proxy, not independently labelled health accuracy.",
            "rul_correlation": correlation,
            "monotonicity_score": round(monotonicity, 4),
            "monotonicity_threshold": threshold,
            "mean_health_index": float(valid[HEALTH_INDEX_COLUMN].mean()) if len(valid) else None,
            "trajectory_metrics": trajectories,
            "monotonicity_definition": "Fraction of engines with <=15% late-life steps rising by more than 1 health point; production index, no second smoothing.",
            "acceptance": acceptance,
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
                "Health index FAILED acceptance: monotonicity=%.2f%% (minimum %.0f%%); checks=%s",
                100.0 * monotonicity,
                100.0 * threshold,
                acceptance,
            )
        return report

    def run(self, frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            path = self.config.paths.processed_path(self.config.paths.health_states_csv)
            if not path.is_file():
                path = self.config.paths.processed_path(self.config.paths.health_scores_csv)
            source = read_csv(
                path, required_columns=get_identity_columns() + [HEALTH_INDEX_COLUMN], stage="evaluate_health"
            )

        raw_path = self.config.paths.processed_path(self.config.paths.raw_data_csv)
        if raw_path.is_file():
            raw = read_csv(raw_path, stage="evaluate_health")
            extras = [c for c in (RUL_COLUMN, "subset_id", "split") if c in raw and c not in source]
            if extras:
                source = source.merge(raw[get_identity_columns() + extras], on=get_identity_columns(),
                                      how="left", validate="one_to_one")

        report = self.evaluate(source)
        atomic_write_json(report, self.config.paths.reports / "health_evaluation.json")
        atomic_write_json(report, self.config.paths.metrics / "evaluate_health.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = HealthEvaluator(config)
    report = evaluator.run()
    logger.info("Health evaluation report: %s", report)


if __name__ == "__main__":
    main()
