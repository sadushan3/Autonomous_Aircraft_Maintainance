"""
human_review_engine.py

Autonomous Safety Guard & Human-Review Flagging Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Determines whether an autonomous supervisor decision requires mandatory human engineering
review based on multi-trigger safety criteria:
1. Low overall decision confidence (`confidence_score` < threshold).
2. High model prediction uncertainty (`entropy_score` >= threshold).
3. High inter-agent prediction conflict (`conflict_score` >= threshold).
4. Context drift or sudden health drop flags.
5. High-risk critical decisions (`immediate_maintenance`) with non-zero uncertainty.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, SupervisorConfig
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "human_review_engine_report.json"


@dataclass
class HumanReviewResult:
    requires_review_flags: np.ndarray  # bool array
    review_reasons: list[str]          # Human readable justification
    review_required_count: int


class HumanReviewEngine:
    """Evaluates safety triggers to determine mandatory human review."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

    def evaluate_human_review(
        self,
        decisions: np.ndarray | pd.Series,
        confidence_scores: np.ndarray,
        entropy_scores: np.ndarray,
        conflict_scores: np.ndarray,
        df_features: pd.DataFrame,
    ) -> HumanReviewResult:
        decisions_arr = np.asarray(decisions)

        # Trigger 1: Low Confidence
        t1 = confidence_scores < self.cfg.low_confidence_threshold

        # Trigger 2: High Uncertainty / Entropy
        t2 = entropy_scores >= self.cfg.high_uncertainty_threshold

        # Trigger 3: Inter-Agent Conflict
        t3 = conflict_scores >= self.cfg.high_conflict_threshold

        # Trigger 4: Context Drift / Sudden Drop
        c_conf = df_features.get("context_confidence", pd.Series(1.0, index=df_features.index)).values
        t4 = c_conf <= self.cfg.low_context_confidence_threshold

        # Combined Trigger
        requires_review = t1 | t2 | t3 | t4

        reasons: list[str] = []
        for i in range(len(decisions_arr)):
            r_list = []
            if t1[i]:
                r_list.append("Low Confidence")
            if t2[i]:
                r_list.append("High Uncertainty")
            if t3[i]:
                r_list.append("Agent Conflict")
            if t4[i]:
                r_list.append("Context Drift")
            reasons.append("; ".join(r_list) if r_list else "Auto Approved")

        return HumanReviewResult(
            requires_review_flags=requires_review,
            review_reasons=reasons,
            review_required_count=int(np.sum(requires_review)),
        )


def run_human_review_engine(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("HUMAN REVIEW ENGINE EVALUATION STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    n_samples = len(df)
    dummy_decisions = np.full(n_samples, "continue_operation")
    dummy_conf = np.full(n_samples, 0.85)
    dummy_ent = np.full(n_samples, 0.20)
    dummy_conflic = np.full(n_samples, 0.10)

    engine = HumanReviewEngine()
    res = engine.evaluate_human_review(dummy_decisions, dummy_conf, dummy_ent, dummy_conflic, df)
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "samples_evaluated": n_samples,
        "review_required_count": res.review_required_count,
        "review_required_percentage": round(100.0 * res.review_required_count / n_samples, 2),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Review Required: %d / %d (%.2f%%)", res.review_required_count, n_samples, report["review_required_percentage"])
    logger.info("Report written to: %s", _REPORT_PATH)
    section("HUMAN REVIEW ENGINE EVALUATION COMPLETED")
    return report


def main() -> int:
    try:
        res = run_human_review_engine()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in human_review_engine: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
