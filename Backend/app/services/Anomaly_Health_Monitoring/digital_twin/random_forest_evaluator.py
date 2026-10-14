"""
Extended evaluation for the Random Forest twin.

Complements the generic training metrics with two forest-specific
diagnostics:

* **Out-of-bag-style generalisation check** -- since scikit-learn's
  multi-output ``RandomForestRegressor`` supports ``oob_score_`` only for
  single-output targets, this module instead compares train-set accuracy
  against held-out engine accuracy directly, which surfaces overfitting to
  specific engines' idiosyncrasies (manufacturing bias, unusual noise) rather
  than genuine operating-context patterns.
* **Tree-count sensitivity** -- refits the forest at a few smaller tree counts
  to show whether the configured ``n_estimators`` is actually buying accuracy
  or just spending compute.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import (
    RandomForestTwin,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    engine_wise_split,
    get_logger,
    read_csv,
)


class RandomForestEvaluator:
    """Extended diagnostics for a trained Random Forest Digital Twin."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Configuration supplying output paths and the random seed.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def check_overfitting(self, twin: RandomForestTwin, frame: pd.DataFrame) -> Dict[str, Any]:
        """Compare training-engine accuracy against held-out-engine accuracy.

        A large gap indicates the forest has memorised specific engines rather
        than learning the operating-context -> sensor relationship in general.

        Args:
            twin: A trained :class:`RandomForestTwin`.
            frame: Scaled, context-labelled data.

        Returns:
            A report of train vs. validation RMSE and the resulting gap.
        """
        healthy = twin.select_healthy_region(frame)
        train_frame, val_frame = engine_wise_split(
            healthy,
            unit_column="unit",
            validation_fraction=self.config.data.validation_engine_fraction,
            random_state=self.config.data.random_state,
        )
        if val_frame.empty:
            val_frame = train_frame

        x_train, _ = twin.build_design_matrix(train_frame, n_modes=twin.n_modes)
        y_train, _ = twin.build_targets(train_frame)
        x_val, _ = twin.build_design_matrix(val_frame, n_modes=twin.n_modes)
        y_val, _ = twin.build_targets(val_frame)

        train_pred = twin._predict_estimator(twin.model, x_train)  # noqa: SLF001
        val_pred = twin._predict_estimator(twin.model, x_val)  # noqa: SLF001

        train_metrics = twin._score(y_train, train_pred)  # noqa: SLF001
        val_metrics = twin._score(y_val, val_pred)  # noqa: SLF001

        gap = val_metrics["rmse"] - train_metrics["rmse"]
        return {
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "generalisation_gap": gap,
            "overfitting_suspected": bool(gap > 0.5 * max(train_metrics["rmse"], 1e-9)),
        }

    def tree_count_sensitivity(
        self, frame: pd.DataFrame, candidates: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Refit at smaller tree counts to check where accuracy plateaus.

        Args:
            frame: Scaled, context-labelled data.
            candidates: Tree counts to try; defaults to a spread below the
                configured value.

        Returns:
            A mapping of tree count to validation RMSE.
        """
        configured = int(self.config.twin.random_forest.get("n_estimators", 200))
        options = candidates or sorted(
            {max(10, configured // 4), max(20, configured // 2), configured}
        )

        probe = RandomForestTwin(self.config)
        healthy = probe.select_healthy_region(frame)
        probe.n_modes = (
            int(frame["operating_mode"].max()) + 1
            if "operating_mode" in frame.columns
            else 1
        )
        train_frame, val_frame = engine_wise_split(
            healthy,
            unit_column="unit",
            validation_fraction=self.config.data.validation_engine_fraction,
            random_state=self.config.data.random_state,
        )
        if val_frame.empty:
            val_frame = train_frame

        x_train, feature_names = probe.build_design_matrix(train_frame, n_modes=probe.n_modes)
        probe.feature_names = feature_names
        y_train, target_names = probe.build_targets(train_frame)
        probe.target_names = target_names
        x_val, _ = probe.build_design_matrix(val_frame, n_modes=probe.n_modes)
        y_val, _ = probe.build_targets(val_frame)

        results: Dict[str, float] = {}
        base_params = dict(self.config.twin.random_forest)
        for count in options:
            params = dict(base_params)
            params["n_estimators"] = int(count)
            model = RandomForestRegressor(**params).fit(x_train, y_train)
            predictions = model.predict(x_val)
            rmse = float(np.sqrt(np.mean((y_val - predictions) ** 2)))
            results[str(count)] = rmse

        self.logger.info("Tree-count sensitivity: %s", results)
        return results

    def run(self, frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run the extended random forest evaluation.

        Args:
            frame: Optional in-memory input; otherwise ``scaled_features.csv``
                is read from disk.

        Returns:
            The combined evaluation report.
        """
        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.scaled_features_csv),
                required_columns=get_identity_columns(),
                stage="random_forest_evaluator",
            )
        twin = RandomForestTwin(self.config).load()
        overfitting = self.check_overfitting(twin, source)
        sensitivity = self.tree_count_sensitivity(source)

        report: Dict[str, Any] = {
            "overfitting_check": overfitting,
            "tree_count_sensitivity": sensitivity,
        }
        atomic_write_json(report, self.config.paths.metrics / "random_forest_evaluation.json")
        self.logger.info("Random forest evaluation: %s", overfitting)
        return report


def main() -> None:
    """Run the random forest evaluator standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    evaluator = RandomForestEvaluator(config)
    report = evaluator.run()
    logger.info("Generalisation gap: %.5f", report["overfitting_check"]["generalisation_gap"])
    logger.info("Tree-count sensitivity: %s", report["tree_count_sensitivity"])


if __name__ == "__main__":
    main()
