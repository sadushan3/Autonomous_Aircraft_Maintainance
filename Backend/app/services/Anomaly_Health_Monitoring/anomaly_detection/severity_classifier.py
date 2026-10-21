from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import json
from pathlib import Path
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


ANOMALY_SCORE_COLUMN = "anomaly_score"


THRESHOLD_USED_COLUMN = "threshold_used"


ANOMALY_TYPE_COLUMN = "anomaly_type"


SEVERITY_COLUMN = "severity"

_ORDER = ["low", "medium", "high", "critical"]


_RELATIVE_BAND_STARTS: Dict[str, float] = {
    "low": 0.0,
    "medium": 0.25,
    "high": 0.55,
    "critical": 0.85,
}


_BAND_PERCENTILES: Dict[str, float] = {"medium": 50.0, "high": 80.0, "critical": 95.0}


SEVERITY_BANDS_FILENAME = "severity_bands.json"


class SeverityClassifier:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.band_starts = self._load_bands()


    def _bands_path(self) -> Path:

        return Path(self.config.paths.model_path(SEVERITY_BANDS_FILENAME))

    def _load_bands(self) -> Dict[str, float]:

        path = self._bands_path()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                bands = {k: float(v) for k, v in stored.get("band_starts", {}).items()}
                if set(bands) >= {"medium", "high", "critical"}:
                    bands.setdefault("low", 0.0)
                    values = [bands[key] for key in ("low", "medium", "high", "critical")]
                    if all(np.isfinite(values)) and 0 <= values[0] and values[-1] < 1 and all(b > a for a, b in zip(values, values[1:])):
                        return bands
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                self.logger.warning(
                    "Severity band file at %s is unreadable; using default fractions.",
                    path,
                )
        return dict(_RELATIVE_BAND_STARTS)

    def calibrate(self, frame: pd.DataFrame) -> Dict[str, float]:


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
        frame = training_scope(frame, self.config)
        record_training_scope("severity", frame, self.config)
        if ANOMALY_TYPE_COLUMN in frame.columns:
            faults = frame[frame[ANOMALY_TYPE_COLUMN] == "fault"]
        else:
            faults = frame
        if faults.empty:
            self.logger.warning(
                "No fault rows to calibrate severity bands; keeping defaults."
            )
            return dict(self.band_starts)

        thresholds = (
            faults[THRESHOLD_USED_COLUMN].astype(float)
            if THRESHOLD_USED_COLUMN in faults.columns
            else pd.Series(
                float(self.config.anomaly.fusion_threshold), index=faults.index
            )
        )
        headroom = (1.0 - thresholds).clip(lower=1e-9)
        position = (
            (faults[ANOMALY_SCORE_COLUMN].astype(float) - thresholds) / headroom
        ).clip(0.0, 1.0)

        bands = {"low": 0.0}
        for name, pctl in _BAND_PERCENTILES.items():
            bands[name] = float(np.percentile(position.to_numpy(), pctl))


        ordered = [bands["low"], bands["medium"], bands["high"], bands["critical"]]
        if any(b <= a for a, b in zip(ordered, ordered[1:])):
            self.logger.warning(
                "Calibrated severity bands are not strictly increasing (%s); "
                "keeping the default fractions.",
                {k: round(v, 4) for k, v in bands.items()},
            )
            bands = dict(_RELATIVE_BAND_STARTS)


        path = self._bands_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "band_starts": bands,
            "percentiles": _BAND_PERCENTILES,
            "fault_rows": int(len(faults)),
            "note": (
                "Positions are (score - threshold) / (1 - threshold) over fault "
                "rows. Frozen at calibration time and reloaded at inference so a "
                "row's severity never depends on the batch it arrives in."
            ),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.band_starts = bands
        self.logger.info(
            "Severity bands calibrated on %d fault row(s): %s",
            len(faults),
            {k: round(v, 4) for k, v in bands.items()},
        )
        return bands

    def _classify_row(self, score: float, threshold: float) -> str:


        if score < threshold:
            return "none"
        headroom = max(1.0 - threshold, 1e-9)
        position = (score - threshold) / headroom
        if position >= self.band_starts["critical"]:
            return "critical"
        if position >= self.band_starts["high"]:
            return "high"
        if position >= self.band_starts["medium"]:
            return "medium"
        return "low"

    def classify(self, frame: pd.DataFrame) -> pd.DataFrame:


        result = frame.copy()
        if THRESHOLD_USED_COLUMN in result.columns:
            thresholds = result[THRESHOLD_USED_COLUMN].astype(float)
        else:
            thresholds = pd.Series(
                float(self.config.anomaly.fusion_threshold), index=result.index
            )

        severity = pd.Series(
            [
                self._classify_row(score, threshold)
                for score, threshold in zip(result[ANOMALY_SCORE_COLUMN], thresholds)
            ],
            index=result.index,
        )

        if ANOMALY_TYPE_COLUMN in result.columns:
            is_operational = result[ANOMALY_TYPE_COLUMN] == "operational"
            severity = severity.where(~is_operational, "none")

        result[SEVERITY_COLUMN] = severity
        distribution = result[SEVERITY_COLUMN].value_counts().to_dict()
        self.logger.info("Severity distribution: %s", distribution)
        return result

    def run(self, frame: pd.DataFrame, calibrate: bool = False) -> pd.DataFrame:


        if calibrate:
            self.calibrate(frame)
        return self.classify(frame)


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.anomaly_fusion_csv),
        required_columns=get_identity_columns() + [ANOMALY_SCORE_COLUMN],
        stage="severity_classifier",
    )
    classifier = SeverityClassifier(config)
    classified = classifier.run(frame)
    logger.info("Severity distribution: %s", classified[SEVERITY_COLUMN].value_counts().to_dict())
    print(classified.head(10).to_string())


if __name__ == "__main__":
    main()
