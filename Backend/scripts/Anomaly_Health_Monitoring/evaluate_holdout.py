"""Deterministic, versioned held-out evaluation for the Anomaly & Health component.

Rebuilds ``metrics/holdout_anomaly_evaluation.json``. The original producer of
that file was not in the repository -- the component's strongest evidence
(held-out AUC, false-alarm rate, detection lead time) existed as a JSON blob
that nothing could regenerate, so it could neither be audited nor used to catch
a regression. This script is that missing producer.

Protocol
--------
Scores are joined to ground truth and split by the ``split`` column the reader
stamps on every row:

* ``train`` rows are run-to-failure trajectories, so ``RUL`` reaches 0.
* ``test`` rows are censored -- the engine had not failed when observation
  stopped -- so most never enter the critical band at all. This is why the
  train/test detection rates are not comparable and are never averaged here.

Every figure is computed against ``RUL``, which the reader derives from cycle
counts (plus ``RUL_FDxxx.txt`` for test units). It is therefore independent of
the residuals every detector consumes -- unlike ``health_state``, which is
derived from the same residuals and yields a meaningless AUC of exactly 1.0.

Band definitions are explicit, recorded in the output, and overridable:

* healthy  : ``RUL >= --healthy-rul``   (default 125)
* critical : ``RUL <= --critical-rul``  (default 30)

Metric definitions that are easy to get wrong
---------------------------------------------
``detection_rate``
    Engines flagged at least once *while inside the critical band*, divided by
    all engines. On censored test data most engines never reach that band, so
    this is bounded above by the fraction that do. ``engines_reaching_critical``
    is reported alongside it so the number is interpretable rather than merely
    low -- the original file reported 0.4597 with no such context.
``lead_time``
    ``RUL`` at each engine's first flagged cycle: how much life remained when
    the system first raised its hand. Comparable across censored and
    uncensored trajectories, unlike "cycles before the last observation".

Determinism
-----------
No sampling, no shuffling, no model fitting. Groupby is sorted, engines are
iterated in sorted order, and every emitted float is rounded to a fixed
precision. Re-running against the same inputs produces a byte-identical
``metrics`` block; only the ``provenance`` timestamp changes.

Versioning
----------
``SCHEMA_VERSION`` covers the shape and meaning of the ``metrics`` block; bump
the minor when adding a field, the major when changing an existing definition.
The ``provenance`` block records SHA-256 of every input, the resolved band
definitions, and library versions, so a stored result can always be traced to
what produced it.

Usage
-----
    python -m scripts.Anomaly_Health_Monitoring.evaluate_holdout
    python -m scripts.Anomaly_Health_Monitoring.evaluate_holdout --healthy-rul 120
    python -m scripts.Anomaly_Health_Monitoring.evaluate_holdout --check   # CI gate

``--check`` re-evaluates and exits non-zero if any headline metric has moved
by more than ``--tolerance`` from the stored file, which turns this into a
regression gate rather than a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

#: Shape/meaning of the ``metrics`` block. Bump on any definition change.
SCHEMA_VERSION = "1.0.0"

#: Fixed rounding so repeated runs are byte-identical.
ROUND = 6

DEFAULT_SCORES = _BACKEND_ROOT / "processed" / "dashboard_data.csv"
DEFAULT_LABELS = _BACKEND_ROOT / "processed" / "cleaned_data.csv"
DEFAULT_OUT = _BACKEND_ROOT / "metrics" / "holdout_anomaly_evaluation.json"

IDENTITY = ["unit", "cycle"]

#: Headline metrics guarded by --check.
GUARDED = (
    "auc_healthy_vs_critical",
    "false_alarm_rate_healthy",
    "detection_rate",
    "overall_flag_rate",
    "lead_time_median",
)


def sha256(path: Path) -> Optional[str]:
    """Content hash of a file, or None if it is missing."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r(value: Any) -> Any:
    """Round floats to a fixed precision; pass everything else through."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), ROUND)
    return value


def load_frames(scores_path: Path, labels_path: Path) -> pd.DataFrame:
    """Join scored rows to ground-truth labels on the identity keys.

    Raises:
        SystemExit: If a required column or file is missing, naming which.
    """
    for path in (scores_path, labels_path):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    scores = pd.read_csv(scores_path, low_memory=False)
    label_cols = IDENTITY + ["RUL", "split", "subset_id"]
    labels = pd.read_csv(labels_path, usecols=lambda c: c in label_cols, low_memory=False)

    missing = [c for c in label_cols if c not in labels.columns]
    if missing:
        raise SystemExit(f"{labels_path.name} lacks required column(s): {missing}")
    for col in ("anomaly_score", "anomaly_flag"):
        if col not in scores.columns:
            raise SystemExit(f"{scores_path.name} lacks required column: {col}")

    merged = scores.merge(labels, on=IDENTITY, how="inner", validate="one_to_one")
    if merged.empty:
        raise SystemExit("scores and labels share no (unit, cycle) rows -- are they from the same run?")
    merged["anomaly_flag"] = merged["anomaly_flag"].fillna(False).astype(bool)
    return merged.sort_values(IDENTITY).reset_index(drop=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Standardised mean difference with a pooled standard deviation."""
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled <= 0:
        return None
    return float((a.mean() - b.mean()) / pooled)


def split_metrics(frame: pd.DataFrame, healthy_rul: int, critical_rul: int) -> Dict[str, Any]:
    """Compute the full metric block for one split."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    out: Dict[str, Any] = {
        "rows": int(len(frame)),
        "engines": int(frame["unit"].nunique()),
    }
    if frame.empty:
        return out

    score = frame["anomaly_score"].astype(float).to_numpy()
    rul = frame["RUL"].astype(float).to_numpy()
    flag = frame["anomaly_flag"].to_numpy()

    healthy = rul >= healthy_rul
    critical = rul <= critical_rul
    out["healthy_rows"] = int(healthy.sum())
    out["critical_rows"] = int(critical.sum())

    # AUC over the two labelled bands only; rows between them are unlabelled.
    both = healthy | critical
    if healthy.any() and critical.any():
        out["auc_healthy_vs_critical"] = r(
            roc_auc_score(critical[both].astype(int), score[both])
        )
        out["cohens_d"] = r(cohens_d(score[critical], score[healthy]))
    else:
        out["auc_healthy_vs_critical"] = None
        out["cohens_d"] = None

    out["spearman_score_vs_rul"] = r(spearmanr(score, rul).correlation)
    out["threshold_used"] = r(
        float(frame["threshold_used"].iloc[0]) if "threshold_used" in frame.columns else None
    )
    out["false_alarm_rate_healthy"] = r(float(flag[healthy].mean()) if healthy.any() else None)
    out["flag_rate_critical"] = r(float(flag[critical].mean()) if critical.any() else None)
    out["overall_flag_rate"] = r(float(flag.mean()))

    # Per-engine detection and lead time, in sorted engine order.
    reaching = 0
    detected = 0
    leads: List[float] = []
    for _, group in frame.groupby("unit", sort=True):
        g_rul = group["RUL"].astype(float).to_numpy()
        g_flag = group["anomaly_flag"].to_numpy()
        in_band = g_rul <= critical_rul
        if in_band.any():
            reaching += 1
            if g_flag[in_band].any():
                detected += 1
        if g_flag.any():
            leads.append(float(g_rul[g_flag][0]))

    engines = out["engines"]
    out["engines_reaching_critical"] = int(reaching)
    out["detection_rate"] = r(detected / engines if engines else None)
    out["detection_rate_of_reaching"] = r(detected / reaching if reaching else None)
    out["engines_flagged"] = int(len(leads))
    if leads:
        arr = np.asarray(sorted(leads))
        out["lead_time_median"] = r(float(np.median(arr)))
        out["lead_time_p25"] = r(float(np.percentile(arr, 25)))
        out["lead_time_p75"] = r(float(np.percentile(arr, 75)))
    else:
        out["lead_time_median"] = out["lead_time_p25"] = out["lead_time_p75"] = None

    if "anomaly_type" in frame.columns:
        counts = frame.loc[flag, "anomaly_type"].value_counts().to_dict()
        out["type_split"] = {str(k): int(v) for k, v in sorted(counts.items())}
    if "severity" in frame.columns:
        means = frame.groupby("severity", sort=True)["RUL"].mean().to_dict()
        out["severity_mean_rul"] = {str(k): r(float(v)) for k, v in means.items()}
    return out


def per_subset(frame: pd.DataFrame, healthy_rul: int, critical_rul: int) -> Dict[str, Any]:
    """Headline metrics broken out per FD subset, in sorted order."""
    out: Dict[str, Any] = {}
    if "subset_id" not in frame.columns:
        return out
    for subset, group in frame.groupby("subset_id", sort=True):
        m = split_metrics(group, healthy_rul, critical_rul)
        out[str(subset)] = {
            "engines": m.get("engines"),
            "auc": m.get("auc_healthy_vs_critical"),
            "false_alarm": m.get("false_alarm_rate_healthy"),
            "detection": m.get("detection_rate"),
            "lead_time_median": m.get("lead_time_median"),
        }
    return out


def build(frame: pd.DataFrame, healthy_rul: int, critical_rul: int) -> Dict[str, Any]:
    """Assemble the full metrics block from the joined frame."""
    metrics: Dict[str, Any] = {}
    for split in ("train", "test"):
        part = frame[frame["split"] == split]
        if len(part):
            metrics[split] = split_metrics(part, healthy_rul, critical_rul)
    test = frame[frame["split"] == "test"]
    if len(test):
        metrics["test_per_subset"] = per_subset(test, healthy_rul, critical_rul)
    if not metrics:
        metrics["all"] = split_metrics(frame, healthy_rul, critical_rul)
        metrics["_note"] = (
            "No 'train'/'test' values found in the split column; the scored run "
            "covered a single split. Set data.split='both' and re-run the "
            "pipeline for a true held-out comparison."
        )
    return metrics


def compare(stored: Dict[str, Any], fresh: Dict[str, Any], tol: float) -> List[str]:
    """Return human-readable drift lines for guarded metrics."""
    drifts: List[str] = []
    for split in ("train", "test", "all"):
        a, b = stored.get(split), fresh.get(split)
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        for key in GUARDED:
            old, new = a.get(key), b.get(key)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                if abs(float(new) - float(old)) > tol:
                    drifts.append(f"{split}.{key}: {old} -> {new} (delta {float(new)-float(old):+.6f})")
    return drifts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--healthy-rul", type=int, default=125)
    ap.add_argument("--critical-rul", type=int, default=30)
    ap.add_argument("--check", action="store_true",
                    help="Compare against the stored file and exit non-zero on drift.")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()

    scores_path, labels_path, out_path = Path(args.scores), Path(args.labels), Path(args.out)
    frame = load_frames(scores_path, labels_path)
    metrics = build(frame, args.healthy_rul, args.critical_rul)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "scripts/Anomaly_Health_Monitoring/evaluate_holdout.py",
            "deterministic": True,
            "inputs": {
                "scores": {"path": str(scores_path), "sha256": sha256(scores_path)},
                "labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
            },
            "band_definitions": {
                "healthy_rul_min": args.healthy_rul,
                "critical_rul_max": args.critical_rul,
                "note": "RUL comes from cycle counts, independent of the residuals the detectors consume.",
            },
            "rows_joined": int(len(frame)),
            "splits_present": sorted(frame["split"].dropna().unique().tolist()),
            "subsets_present": sorted(frame["subset_id"].dropna().astype(str).unique().tolist())
            if "subset_id" in frame.columns else [],
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "metrics": metrics,
    }

    if args.check:
        if not out_path.exists():
            print(f"--check: no stored file at {out_path}")
            return 1
        stored = json.loads(out_path.read_text(encoding="utf-8"))
        drifts = compare(stored.get("metrics", stored), metrics, args.tolerance)
        if drifts:
            print(f"REGRESSION: {len(drifts)} guarded metric(s) moved beyond {args.tolerance}:")
            for line in drifts:
                print(f"  {line}")
            return 1
        print(f"OK: guarded metrics within {args.tolerance} of the stored result.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}  (schema {SCHEMA_VERSION})")
    for split, block in metrics.items():
        if isinstance(block, dict) and "rows" in block:
            print(
                f"  {split:6s} rows={block['rows']:>7} engines={block['engines']:>4} "
                f"AUC={block.get('auc_healthy_vs_critical')} "
                f"FA={block.get('false_alarm_rate_healthy')} "
                f"det={block.get('detection_rate')} "
                f"lead_med={block.get('lead_time_median')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
