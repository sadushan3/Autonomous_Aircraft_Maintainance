"""
Operating-mode assignment and regime-transition marking.

This is the orchestrator of the context-modelling layer.  It selects the
configured clustering algorithm, assigns an ``operating_mode`` to every row,
records how confidently that assignment was made, and -- critically -- marks
the rows that sit near a regime *change*.

The transition marking is what lets the anomaly layer tell a fault from a
throttle change.  When an engine switches regime, several sensors jump within a
single cycle.  Those rows are flagged ``is_mode_transition=True``, and the
anomaly fusion stage uses that flag, together with whether the deviation
reverts and whether it persists, to classify a detection as ``operational``
rather than ``fault``.

Output: ``context_clusters.csv`` -- the cleaned/engineered data plus
``operating_mode``, ``mode_confidence``, ``mode_posterior``,
``is_mode_transition`` and ``cycles_since_mode_change``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.context_modeling.gmm_context import (
    GMMContextModel,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.kmeans_context import (
    KMeansContextModel,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    SUBSET_COLUMN,
    expected_operating_conditions,
    get_identity_columns,
    get_op_setting_columns,
    get_operating_mode_column,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
)

#: Confidence of the assigned regime, in [0, 1].
MODE_CONFIDENCE_COLUMN = "mode_confidence"

#: Posterior probability of the assigned regime (GMM) or margin proxy (KMeans).
MODE_POSTERIOR_COLUMN = "mode_posterior"

#: True when the row lies within the guard window of a regime change.
MODE_TRANSITION_COLUMN = "is_mode_transition"

#: Cycles elapsed since this engine last changed regime.
CYCLES_SINCE_CHANGE_COLUMN = "cycles_since_mode_change"


class OperatingModeDetector:
    """Assigns operating modes and marks regime transitions per engine."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the detector.

        Args:
            config: Configuration supplying the algorithm choice and the
                transition guard width.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.mode_column = get_operating_mode_column()
        self.algorithm = str(self.config.context.algorithm).lower().strip()
        self.model: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #
    def _build_model(self) -> Any:
        """Instantiate the configured clustering model.

        Returns:
            An unfitted context model.

        Raises:
            AnomalyHealthError: If the configured algorithm is unknown.
        """
        if self.algorithm == "kmeans":
            return KMeansContextModel(self.config)
        if self.algorithm == "gmm":
            return GMMContextModel(self.config)
        raise AnomalyHealthError(
            f"Unknown context algorithm '{self.algorithm}'; expected 'kmeans' or 'gmm'."
        )

    # ------------------------------------------------------------------ #
    # Training / inference
    # ------------------------------------------------------------------ #
    def fit(self, frame: pd.DataFrame) -> "OperatingModeDetector":
        """Fit the clustering model on the operating settings.

        Args:
            frame: Data containing the three settings columns.

        Returns:
            This fitted instance.
        """
        self.model = self._build_model()
        self.model.fit(frame)
        return self

    def load(self) -> "OperatingModeDetector":
        """Reload a previously fitted clustering model.

        Returns:
            This instance, ready to assign modes.
        """
        self.model = self._build_model()
        self.model.load()
        return self

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach mode labels, confidence and transition markers.

        Args:
            frame: Data containing the operating settings and identity keys.

        Returns:
            A copy with the context columns added.

        Raises:
            AnomalyHealthError: If no model has been fitted or loaded.
        """
        if self.model is None:
            raise AnomalyHealthError("OperatingModeDetector used before fit/load.")

        result = frame.sort_values(get_identity_columns()).reset_index(drop=True)
        result[self.mode_column] = self.model.predict(result).astype(int)

        confidence = self.model.confidence(result)
        result[MODE_CONFIDENCE_COLUMN] = np.clip(confidence, 0.0, 1.0)
        result[MODE_POSTERIOR_COLUMN] = result[MODE_CONFIDENCE_COLUMN]

        result = self.mark_transitions(result)
        return result

    def mark_transitions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Flag rows lying within the guard window of a regime change.

        The guard is forward-only: only the change cycle itself and up to
        ``guard`` cycles *after* it are marked.  A sensor's transient response
        to a throttle step is necessarily observed after the step occurs, so
        there is no physical basis for also guarding cycles *before* a change
        -- a centred window did that anyway, which silently doubled the
        guard's footprint for no reason. That matters most on subsets like
        FD002/FD004, where the operating regime can change every one to two
        cycles: a centred window there overlaps almost every row in the
        dataset, while a forward-only window only ever covers the cycles that
        could plausibly still be settling from the most recent switch.

        Args:
            frame: Data carrying the operating-mode column, sorted by keys.

        Returns:
            The frame with transition markers added.
        """
        result = frame.copy()
        guard = max(int(self.config.context.mode_transition_guard), 0)

        changed = (
            result.groupby("unit", sort=False)[self.mode_column]
            .transform(lambda series: series.ne(series.shift()).fillna(False))
            .astype(bool)
        )
        # The first cycle of an engine is not a transition, it is a start.
        first_cycle = (
            result.groupby("unit", sort=False).cumcount() == 0
        )
        changed = changed & ~first_cycle

        if guard > 0:
            window = guard + 1
            near_change = (
                changed.astype(float)
                .groupby(result["unit"], sort=False)
                .transform(
                    lambda series: series.rolling(window, center=False, min_periods=1).max()
                )
                > 0
            )
        else:
            near_change = changed

        result[MODE_TRANSITION_COLUMN] = near_change.astype(bool)

        # Cycles elapsed since the last regime change, per engine.
        change_group = changed.groupby(result["unit"], sort=False).cumsum()
        result[CYCLES_SINCE_CHANGE_COLUMN] = (
            result.groupby(["unit", change_group], sort=False).cumcount().astype(int)
        )

        n_changes = int(changed.sum())
        n_marked = int(result[MODE_TRANSITION_COLUMN].sum())
        self.logger.info(
            "Marked %d regime change(s); %d row(s) lie within the +/-%d cycle guard",
            n_changes,
            n_marked,
            guard,
        )
        return result

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def verify_mode_count(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Compare the discovered regime count against subset expectations.

        FD002 and FD004 should yield six regimes; FD001 and FD003 one.  A
        mismatch is reported rather than raised, because a user may be running
        a truncated smoke sample where the full regime set is not present.

        Args:
            frame: Labelled data.

        Returns:
            A report of found versus expected regimes.
        """
        found = int(frame[self.mode_column].nunique())
        subsets: List[str] = (
            sorted(frame[SUBSET_COLUMN].dropna().unique().tolist())
            if SUBSET_COLUMN in frame.columns
            else list(self.config.data.subsets)
        )
        expected = expected_operating_conditions(subsets)
        matches = found == expected
        if not matches:
            self.logger.warning(
                "Discovered %d operating regime(s) but subsets %s document %d. "
                "Per-mode normalisation still applies to whatever was found.",
                found,
                subsets,
                expected,
            )
        return {
            "modes_found": found,
            "modes_expected": expected,
            "matches_expectation": matches,
            "subsets": subsets,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``context_clusters.csv`` atomically.

        Args:
            frame: Labelled data.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.context_clusters_csv
        )
        verify_identity_columns(frame, context="context_clusters output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``context_clusters.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.context_clusters_csv
        )
        return read_csv(
            target,
            required_columns=get_identity_columns() + [self.mode_column],
            stage="operating_mode_detector",
        )

    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        refit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the operating-mode stage.

        Args:
            frame: Optional in-memory input; otherwise
                ``engineered_features.csv`` is read from disk.
            refit: Fit a fresh clustering model, or reload a saved one.

        Returns:
            A ``(labelled_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(
                    self.config.paths.engineered_features_csv
                ),
                required_columns=get_identity_columns() + get_op_setting_columns(),
                stage="operating_mode_detector",
            )

        if refit:
            self.fit(source)
            self.model.save()
        else:
            self.load()

        labelled = self.assign(source)
        verification = self.verify_mode_count(labelled)

        distribution = (
            labelled[self.mode_column].value_counts().sort_index().to_dict()
        )
        report: Dict[str, Any] = {
            "rows": int(len(labelled)),
            "engines": int(labelled["unit"].nunique()),
            "algorithm": self.algorithm,
            "n_modes": verification["modes_found"],
            "expected_modes": verification["modes_expected"],
            "matches_expectation": verification["matches_expectation"],
            "mode_distribution": {str(k): int(v) for k, v in distribution.items()},
            "mean_mode_confidence": float(labelled[MODE_CONFIDENCE_COLUMN].mean()),
            "transition_rows": int(labelled[MODE_TRANSITION_COLUMN].sum()),
            "model_summary": self.model.summary(),
        }
        if isinstance(self.model, KMeansContextModel):
            report["silhouette_score"] = self.model.silhouette

        report["output_path"] = str(self.save(labelled))
        atomic_write_json(
            report, self.config.paths.metrics / "context_modeling.json"
        )
        self.logger.info("Context modelling report: %s", {
            k: v for k, v in report.items() if k != "model_summary"
        })
        return labelled, report


def main() -> None:
    """Run operating-mode detection standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    detector = OperatingModeDetector(config)
    labelled, report = detector.run()
    logger.info(
        "context_clusters.csv written: %d rows, %d regime(s)",
        report["rows"],
        report["n_modes"],
    )
    print(
        labelled[
            get_identity_columns()
            + get_op_setting_columns()
            + [
                get_operating_mode_column(),
                MODE_CONFIDENCE_COLUMN,
                MODE_TRANSITION_COLUMN,
            ]
        ]
        .head(12)
        .to_string()
    )


if __name__ == "__main__":
    main()
