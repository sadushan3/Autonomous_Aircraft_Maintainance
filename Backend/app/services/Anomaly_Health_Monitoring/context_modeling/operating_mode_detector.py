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


MODE_CONFIDENCE_COLUMN = "mode_confidence"


MODE_POSTERIOR_COLUMN = "mode_posterior"


MODE_TRANSITION_COLUMN = "is_mode_transition"


CYCLES_SINCE_CHANGE_COLUMN = "cycles_since_mode_change"


class OperatingModeDetector:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.mode_column = get_operating_mode_column()
        self.algorithm = str(self.config.context.algorithm).lower().strip()
        self.model: Optional[Any] = None


    def _build_model(self) -> Any:


        if self.algorithm == "kmeans":
            return KMeansContextModel(self.config)
        if self.algorithm == "gmm":
            return GMMContextModel(self.config)
        raise AnomalyHealthError(
            f"Unknown context algorithm '{self.algorithm}'; expected 'kmeans' or 'gmm'."
        )


    def fit(self, frame: pd.DataFrame) -> "OperatingModeDetector":


        self.model = self._build_model()
        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope
        self.model.fit(training_scope(frame, self.config))
        return self

    def load(self) -> "OperatingModeDetector":


        self.model = self._build_model()
        self.model.load()
        return self

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:


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


        result = frame.copy()
        guard = max(int(self.config.context.mode_transition_guard), 0)

        changed = (
            result.groupby("unit", sort=False)[self.mode_column]
            .transform(lambda series: series.ne(series.shift()).fillna(False))
            .astype(bool)
        )

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


    def verify_mode_count(self, frame: pd.DataFrame) -> Dict[str, Any]:


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


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.context_clusters_csv
        )
        verify_identity_columns(frame, context="context_clusters output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


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
            from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, prepare_training_split, training_scope, record_training_scope
            if load_training_split(self.config) is None:
                prepare_training_split(source, self.config)
            fitting = training_scope(source, self.config, required=True)
            self.fit(fitting)
            record_training_scope("context", fitting, self.config)
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
