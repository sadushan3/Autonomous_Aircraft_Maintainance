"""
Health state classification: Healthy -> Degrading -> Warning -> Critical -> Failure Risk.

This is deliberately a **threshold classifier**, not a trained model.  The
research specification is explicit about this: state assignment must stay
threshold-driven so the boundary between "Warning" and "Critical" is a number
an engineer can read, audit and adjust -- not a decision boundary buried inside
a black-box classifier.  Introducing a trained model here would also create a
second place (besides the twin) where "normal" has to be learned, for no
benefit.

Thresholds are persisted to ``models/health_state_thresholds.json`` so they can
be inspected, and so the feedback layer has something concrete to adapt if
operators find the boundaries too aggressive or too lax over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import (
    HEALTH_SCORE_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)

#: The assigned state label.
HEALTH_STATE_COLUMN = "health_state"

#: True on the exact cycle the state changed from the previous cycle.
STATE_CHANGED_COLUMN = "state_changed"

#: Ordered worst-to-best-free state ladder, matching config.health.state_order.
_DEFAULT_ORDER = ["Healthy", "Degrading", "Warning", "Critical", "Failure Risk"]


class HealthStateClassifier:
    """Assigns a discrete health state from persisted score thresholds."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the classifier.

        Args:
            config: Configuration supplying default thresholds and the state
                order.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.thresholds_path = self.config.paths.model_path(
            self.config.paths.health_state_thresholds_json
        )
        self.thresholds: Dict[str, float] = dict(self.config.health.state_thresholds)
        self.state_order: List[str] = list(self.config.health.state_order)

    # ------------------------------------------------------------------ #
    # Threshold management
    # ------------------------------------------------------------------ #
    def load_thresholds(self) -> Dict[str, float]:
        """Load persisted thresholds, falling back to config defaults.

        Returns:
            The active threshold mapping (``healthy``, ``degrading``,
            ``warning``, ``critical`` -> minimum score for that state).
        """
        payload = read_json(self.thresholds_path, default=None)
        if payload and "thresholds" in payload:
            self.thresholds = {k: float(v) for k, v in payload["thresholds"].items()}
            self.logger.info("Loaded persisted health-state thresholds: %s", self.thresholds)
        else:
            self.save_thresholds()
        return self.thresholds

    def save_thresholds(self) -> Path:
        """Persist the active thresholds.

        Returns:
            The path written.
        """
        payload = {"thresholds": self.thresholds, "state_order": self.state_order}
        return atomic_write_json(payload, self.thresholds_path)

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Apply an update to specific threshold keys and persist it.

        Used by the feedback / threshold-adaptation layer when operators
        indicate the state boundaries need adjusting.

        Args:
            new_thresholds: Partial or full mapping of threshold updates.

        Returns:
            The resulting full threshold mapping.
        """
        self.thresholds.update({k: float(v) for k, v in new_thresholds.items()})
        self.save_thresholds()
        self.logger.info("Updated health-state thresholds: %s", self.thresholds)
        return self.thresholds

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _classify_scalar(self, score: float) -> str:
        """Classify a single health score against the loaded thresholds.

        Args:
            score: A 0-100 health score.

        Returns:
            The matching state label.
        """
        if score >= self.thresholds.get("healthy", 80.0):
            return "Healthy"
        if score >= self.thresholds.get("degrading", 60.0):
            return "Degrading"
        if score >= self.thresholds.get("warning", 40.0):
            return "Warning"
        if score >= self.thresholds.get("critical", 20.0):
            return "Critical"
        return "Failure Risk"

    def classify(self, health_scores: pd.DataFrame) -> pd.DataFrame:
        """Assign a health state to every row and mark state transitions.

        Args:
            health_scores: Frame carrying identity keys and ``health_score``.

        Returns:
            The frame with ``health_state`` and ``state_changed`` added.
        """
        self.load_thresholds()
        result = health_scores.sort_values(get_identity_columns()).reset_index(drop=True)

        result[HEALTH_STATE_COLUMN] = result[HEALTH_SCORE_COLUMN].apply(self._classify_scalar)

        previous_state = result.groupby("unit", sort=False)[HEALTH_STATE_COLUMN].shift()
        first_cycle = result.groupby("unit", sort=False).cumcount() == 0
        result[STATE_CHANGED_COLUMN] = (
            (result[HEALTH_STATE_COLUMN] != previous_state) & ~first_cycle
        )

        distribution = result[HEALTH_STATE_COLUMN].value_counts().to_dict()
        self.logger.info("Health state distribution: %s", distribution)
        return result

    def state_rank(self, state: str) -> int:
        """Return the ordinal rank of a state, 0 = best.

        Args:
            state: A health-state label.

        Returns:
            Its position in ``state_order``, or the worst rank when unknown.
        """
        try:
            return self.state_order.index(state)
        except ValueError:
            return len(self.state_order) - 1

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``health_states.csv`` atomically.

        Args:
            frame: Classified data.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_states_csv)
        verify_identity_columns(frame, context="health_states output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``health_states.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_states_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_state_classifier"
        )

    def run(self, health_scores: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the state-classification stage.

        Args:
            health_scores: Optional in-memory input; otherwise
                ``health_scores.csv`` is read from disk.

        Returns:
            A ``(classified_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = health_scores
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.health_scores_csv),
                required_columns=get_identity_columns() + [HEALTH_SCORE_COLUMN],
                stage="health_state_classifier",
            )

        classified = self.classify(source)
        output = self.save(classified)

        distribution = classified[HEALTH_STATE_COLUMN].value_counts().to_dict()
        report: Dict[str, Any] = {
            "rows": int(len(classified)),
            "engines": int(classified["unit"].nunique()),
            "state_distribution": {str(k): int(v) for k, v in distribution.items()},
            "state_changes": int(classified[STATE_CHANGED_COLUMN].sum()),
            "thresholds": self.thresholds,
            "output_path": str(output),
        }
        self.logger.info("Health state report: %s", report)
        return classified, report


def main() -> None:
    """Run health-state classification standalone against ``health_scores.csv``."""
    config = get_config()
    logger = get_logger(__name__, config)
    classifier = HealthStateClassifier(config)
    classified, report = classifier.run()
    logger.info("health_states.csv written: %d rows", report["rows"])
    print(classified.head(10).to_string())


if __name__ == "__main__":
    main()
