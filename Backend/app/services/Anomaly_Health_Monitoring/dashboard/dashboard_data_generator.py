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


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def _load_optional(self, filename: str, stage: str) -> Optional[pd.DataFrame]:


        path = self.config.paths.processed_path(filename)
        if not path.is_file():
            self.logger.info("%s: '%s' not yet produced; dashboard will omit its columns.", stage, filename)
            return None
        return read_csv(path, stage=stage)


    def build(self) -> pd.DataFrame:


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
            except Exception as exc:
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


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.dashboard_data_csv)
        verify_identity_columns(frame, context="dashboard_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def run(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:


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

    config = get_config()
    logger = get_logger(__name__, config)
    generator = DashboardDataGenerator(config)
    wide, report = generator.run()
    logger.info("dashboard_data.csv written: %d rows, %d columns", report["rows"], report["columns"])
    print(wide.head(5).to_string())


if __name__ == "__main__":
    main()
