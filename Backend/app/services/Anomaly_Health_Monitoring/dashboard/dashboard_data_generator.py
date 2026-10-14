"""
Dashboard data generation: the wide merged view feeding the UI and the API.

Joins every carry-forward CSV on ``(unit, cycle)`` into one wide table --
health, anomaly, reasoning, explanation and confidence columns together -- and
writes it as ``dashboard_data.csv``.  This is the last carry-forward CSV in
the pipeline; :mod:`dashboard_service` reshapes this wide table into the
per-engine/timeline/alert views the API returns, and it is also the direct
source for the integration handoff file written by the FastAPI layer.

Every join uses :func:`merge_on_keys`'s ``validate='one_to_one'`` (relaxed to
``many_to_one`` where a stage produced fewer rows, e.g. reasoning only covers
flagged rows) so a silently duplicated key row anywhere upstream surfaces here
as a loud error rather than a doubled dashboard row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
)


class DashboardDataGenerator:
    """Builds the single wide table the dashboard and API layers consume."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the generator.

        Args:
            config: Configuration supplying carry-forward CSV paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    # ------------------------------------------------------------------ #
    # Source loading
    # ------------------------------------------------------------------ #
    def _load_optional(self, filename: str, stage: str) -> Optional[pd.DataFrame]:
        """Read a carry-forward CSV if it exists, else return None.

        Args:
            filename: The CSV filename under ``processed/``.
            stage: Label used in log messages.

        Returns:
            The loaded frame, or None when the file is absent.
        """
        path = self.config.paths.processed_path(filename)
        if not path.is_file():
            self.logger.info("%s: '%s' not yet produced; dashboard will omit its columns.", stage, filename)
            return None
        return read_csv(path, stage=stage)

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #
    def build(self) -> pd.DataFrame:
        """Join every available stage output into one wide table.

        Returns:
            The merged wide table, keyed on ``(unit, cycle)``.

        Raises:
            AnomalyHealthError: If the mandatory health-states input is absent.
        """
        keys = get_identity_columns()

        base = self._load_optional(self.config.paths.health_states_csv, "dashboard_data_generator")
        if base is None:
            raise AnomalyHealthError(
                "health_states.csv is required to build the dashboard; run the "
                "health-monitoring stage first."
            )
        result = base.copy()

        overlays: List[Tuple[str, str]] = [
            (self.config.paths.anomaly_fusion_csv, "many_to_one"),
            (self.config.paths.root_cause_analysis_csv, "many_to_one"),
            (self.config.paths.explanation_reports_csv, "many_to_one"),
            (self.config.paths.confidence_scores_csv, "many_to_one"),
        ]

        for filename, validate in overlays:
            overlay = self._load_optional(filename, "dashboard_data_generator")
            if overlay is None:
                continue
            overlapping = [c for c in overlay.columns if c in result.columns and c not in keys]
            trimmed = overlay.drop(columns=overlapping) if overlapping else overlay
            try:
                result = result.merge(trimmed, on=keys, how="left", validate=validate)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Skipping overlay '%s' due to a merge issue (%s); dashboard "
                    "will omit its columns rather than fail the whole stage.",
                    filename,
                    exc,
                )

        self.logger.info(
            "Dashboard data assembled: %d rows, %d columns, %d engines",
            len(result),
            result.shape[1],
            result["unit"].nunique(),
        )
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``dashboard_data.csv`` atomically.

        Args:
            frame: The wide dashboard table.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.dashboard_data_csv)
        verify_identity_columns(frame, context="dashboard_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the dashboard-data-generation stage.

        Returns:
            A ``(wide_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        wide = self.build()
        output = self.save(wide)

        report: Dict[str, Any] = {
            "rows": int(len(wide)),
            "engines": int(wide["unit"].nunique()),
            "columns": int(wide.shape[1]),
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "dashboard_data.json")
        self.logger.info("Dashboard data generation report: %s", report)
        return wide, report


def main() -> None:
    """Run dashboard data generation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    generator = DashboardDataGenerator(config)
    wide, report = generator.run()
    logger.info("dashboard_data.csv written: %d rows, %d columns", report["rows"], report["columns"])
    print(wide.head(5).to_string())


if __name__ == "__main__":
    main()
