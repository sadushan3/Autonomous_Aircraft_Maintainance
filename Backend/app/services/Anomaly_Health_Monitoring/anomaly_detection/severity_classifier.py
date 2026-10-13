"""
Severity classification for fused anomalies.

Severity bands are expressed as *fractions of the way from the calibrated
fusion threshold up to the maximum possible score (1.0)*, not as absolute cuts
in raw score space.  This ties severity to the same anchor the anomaly flag
itself uses: whatever the fused score happens to mean on a given run (its
scale depends on how the underlying detectors are calibrated), the flagging
threshold is always "worse than ~99% of healthy behaviour", and severity asks
how much further past that anchor a row sits.  Fixed absolute cuts (e.g.
"critical starts at 0.9") silently assume a specific score scale and go stale
-- and in this component's case, inconsistent with ``anomaly_flag`` -- the
moment that scale changes, which is exactly what happened here: detector
scores were min-max normalised before, so 0.5-0.9 was a meaningful high band,
but under the current percentile-rank normalisation the healthy population
itself commonly produces fused scores in that range, which used to classify
un-flagged healthy rows as "medium"/"high"/"critical" severity.

Rows classified as ``anomaly_type='operational'`` by the fusion gate are
always reported as severity ``'none'`` regardless of their raw score --
severity is a statement about fault seriousness, and an operational detection
is, by definition, not a fault. Rows below the flagging threshold are always
``'none'`` too, since severity describes a flagged anomaly's seriousness, not
a description of ordinary variation among rows nobody flagged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import json
from pathlib import Path
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger

#: Fused anomaly score column, mirrored from anomaly_fusion.py.
#: Kept as a local string constant (rather than imported) to avoid a circular
#: import: anomaly_fusion.py is the orchestrator that calls this module.
ANOMALY_SCORE_COLUMN = "anomaly_score"

#: Per-row calibrated flagging threshold, mirrored from anomaly_fusion.py.
THRESHOLD_USED_COLUMN = "threshold_used"

#: 'fault' | 'operational' | 'none', mirrored from anomaly_fusion.py.
ANOMALY_TYPE_COLUMN = "anomaly_type"

#: Severity label column.
SEVERITY_COLUMN = "severity"

_ORDER = ["low", "medium", "high", "critical"]

#: How far past the flagging threshold (as a fraction of the remaining
#: distance to a perfect score of 1.0) each band starts. Used only when no
#: calibrated bands have been persisted yet -- see ``_load_bands``.
#:
#: These fixed fractions do not survive contact with a tight headroom. Once
#: the fusion threshold sits at 0.9563 the headroom is only 0.0437, fused
#: scores saturate near 1.0, and the fractions put 59% of every fault alert
#: in 'critical' -- useless for triage -- while producing a severity scale
#: that is NOT monotone in remaining life. Measured 2026-08-24 on 38,742
#: pooled fault rows: low 44.8 / medium 55.9 / high 46.1 / critical 28.4 mean
#: RUL, i.e. 'low' engines were closer to failure than 'medium' ones.
_RELATIVE_BAND_STARTS: Dict[str, float] = {
    "low": 0.0,
    "medium": 0.25,
    "high": 0.55,
    "critical": 0.85,
}

#: Percentiles of the fault-row position distribution used to calibrate the
#: bands, chosen so 'critical' is the top 5% of alerts rather than the
#: majority. Calibrated cuts reorder severity correctly by remaining life:
#: low 46.1 / medium 30.9 / high 22.4 / critical 10.7 mean RUL.
_BAND_PERCENTILES: Dict[str, float] = {"medium": 50.0, "high": 80.0, "critical": 95.0}

#: Where calibrated bands are persisted. Calibration happens once, offline,
#: and the result is frozen and reloaded at inference -- severity must never
#: be computed relative to whatever batch happens to be in front of it, for
#: the same reason the fusion threshold is frozen: batch-relative scoring
#: makes a row's label depend on its neighbours.
SEVERITY_BANDS_FILENAME = "severity_bands.json"


class SeverityClassifier:
    """Buckets the fused anomaly score into a severity label relative to the
    calibrated flagging threshold."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the classifier.

        Args:
            config: Configuration (accepted for interface consistency; the
                relative band fractions are fixed, since they express a
                position between "just flagged" and "worst possible", not a
                raw score value that would need per-deployment tuning).
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.band_starts = self._load_bands()

    # ------------------------------------------------------------------ #
    # Calibrated bands
    # ------------------------------------------------------------------ #
    def _bands_path(self) -> Path:
        """Location of the persisted band calibration."""
        return Path(self.config.paths.model_path(SEVERITY_BANDS_FILENAME))

    def _load_bands(self) -> Dict[str, float]:
        """Return the frozen band starts, falling back to the fixed fractions."""
        path = self._bands_path()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                bands = {k: float(v) for k, v in stored.get("band_starts", {}).items()}
                if set(bands) >= {"medium", "high", "critical"}:
                    bands.setdefault("low", 0.0)
                    return bands
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                self.logger.warning(
                    "Severity band file at %s is unreadable; using default fractions.",
                    path,
                )
        return dict(_RELATIVE_BAND_STARTS)

    def calibrate(self, frame: pd.DataFrame) -> Dict[str, float]:
        """Fit and persist band starts from the fault rows in ``frame``.

        Severity answers "how far past the alarm did this go, relative to the
        other alarms this detector raises". That is a percentile question, so
        the cuts are taken from the observed distribution of positions rather
        than assumed. Only ``anomaly_type == 'fault'`` rows participate:
        operational detections are labelled ``'none'`` by design and would
        otherwise drag the percentiles toward benign regime changes.

        Args:
            frame: Fused frame carrying score, threshold and anomaly type.

        Returns:
            The calibrated band starts, also written to ``models_dir``.
        """
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

        # Guard against a degenerate distribution collapsing the bands: if the
        # cuts are not strictly increasing the percentiles have landed on top
        # of one another, and the fixed fractions are the safer fallback.
        ordered = [bands["low"], bands["medium"], bands["high"], bands["critical"]]
        if any(b <= a for a, b in zip(ordered, ordered[1:])):
            self.logger.warning(
                "Calibrated severity bands are not strictly increasing (%s); "
                "keeping the default fractions.",
                {k: round(v, 4) for k, v in bands.items()},
            )
            return dict(_RELATIVE_BAND_STARTS)

        # Round BEFORE persisting and adopt the rounded values in memory, so the
        # calibrating run and every later reload classify identically. Keeping
        # full precision here while writing 6dp shifted 9 rows between bands
        # between the fitting run and the next load.
        bands = {k: round(v, 6) for k, v in bands.items()}

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
        """Classify a single anomaly score relative to its flagging threshold.

        Args:
            score: Fused anomaly score in [0, 1].
            threshold: The calibrated flagging threshold in force for this row.

        Returns:
            One of ``'none'``, ``'low'``, ``'medium'``, ``'high'``, ``'critical'``.
        """
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
        """Attach severity labels to a fused-anomaly frame.

        Args:
            frame: Frame carrying ``anomaly_score``, ``anomaly_type`` and
                (when available) ``threshold_used``; falls back to
                ``config.anomaly.fusion_threshold`` when the latter is absent.

        Returns:
            The frame with ``severity`` added.
        """
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
        """Execute severity classification as an in-memory transform.

        This module contributes columns to ``anomaly_fusion.csv`` rather than
        owning a carry-forward file of its own; the caller (typically
        :mod:`anomaly_fusion` or the pipeline stage script) persists the
        result.

        Args:
            frame: Fused-anomaly frame.
            calibrate: Refit the band cuts from this frame before classifying.
                Only ever true on a retraining run. At inference the frozen
                cuts are reloaded instead, so a row's severity depends on the
                detector's calibration and not on the batch it arrived in.

        Returns:
            The classified frame.
        """
        if calibrate:
            self.calibrate(frame)
        return self.classify(frame)


def main() -> None:
    """Run severity classification standalone against ``anomaly_fusion.csv``."""
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
