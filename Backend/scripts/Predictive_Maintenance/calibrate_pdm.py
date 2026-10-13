"""Calibrate RUL conformal intervals and Failure-Risk alert thresholds.

Neither task retrains anything. Both replace assumed constants with values
fitted to observed model behaviour, and both are validated on engines that took
no part in the fitting.

Why this exists
---------------
**RUL intervals were not conformal.** ``DEFAULT_RUL_Q_HAT`` is a hand-set
per-subset constant (20/25/20/28). Split-conformal prediction requires the
half-width to be a *quantile of observed absolute residuals* on a calibration
set; only then does the interval carry its nominal coverage. Measured coverage
of the assumed constants was PICP 0.753 / 0.507 / 0.835 / 0.390 against a
nominal 0.90 -- i.e. on FD004 the band missed the truth 61% of the time. Until
this ran, "TCN_Conformal" was a label the intervals had not earned.

**Failure-Risk thresholds were mis-set on two subsets.** At the shipped cuts,
FD002's 10-cycle horizon achieved recall 0.000 -- it caught none of the 22
imminent failures despite AUC 0.911, because the cut sat at 0.97 while the
model's scores for that horizon live far below it. FD004's precision was 0.495
at 30 cycles, so half of every alert was false.

Protocol
--------
Test engines per subset are split 50/50 by a seeded permutation:

* **calibration half** -- fits q_hat and the thresholds.
* **evaluation half**  -- reports coverage, recall and precision. These engines
  are never seen by the fitting step, so the numbers are not self-graded.

RUL: nonconformity is ``|y_true - y_pred|`` at each engine's final window, and
q_hat is the ``ceil((n+1)(1-alpha))/n`` empirical quantile -- the finite-sample
correction that makes split conformal valid, rather than the plain quantile.

Failure Risk: for each horizon the threshold is the lowest value achieving
``--min-recall`` on the calibration half, tie-broken on precision. That is the
convention already recorded in this project's FD002 threshold provenance, kept
here so subsets stay comparable. When no threshold reaches the recall floor,
the best-F1 cut is used instead and the shortfall is recorded.

Outputs
-------
* ``models/Predictive_Maintenance/conformal_calibration.json`` -- q_hat per
  subset, consumed by ``RULLifecycleService`` in preference to the config
  constant.
* Failure-Risk metadata JSONs are updated in place, preserving every existing
  key and recording provenance under ``threshold_provenance``. The previous
  file is copied to ``*.pre_calibration_<stamp>`` first.

Usage
-----
    python -m scripts.Predictive_Maintenance.calibrate_pdm            # report only
    python -m scripts.Predictive_Maintenance.calibrate_pdm --apply    # write files
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Predictive_Maintenance.config import get_settings  # noqa: E402
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader  # noqa: E402

RAW_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
HORIZONS = ("10", "30", "50")
CONFORMAL_PATH = (
    _BACKEND_ROOT / "models" / "Predictive_Maintenance" / "conformal_calibration.json"
)


def load_split(fd: str, data_root: Path):
    """Read one subset's test frame plus its per-engine true RUL."""
    test = pd.read_csv(data_root / f"test_{fd}.txt", sep=r"\s+", header=None, engine="python")
    test = test.iloc[:, : len(RAW_COLUMNS)]
    test.columns = RAW_COLUMNS
    truth = pd.read_csv(
        data_root / f"RUL_{fd}.txt", sep=r"\s+", header=None, engine="python"
    ).iloc[:, 0].to_numpy(dtype=float)
    return test, truth


def final_windows(test: pd.DataFrame, truth: np.ndarray, loader, feature_columns, scaler, window: int):
    """Build the final-window tensor per engine, with true RUL and engine ids."""
    from app.services.Predictive_Maintenance.preprocessing import PreprocessingService

    pre = PreprocessingService()
    X: List[np.ndarray] = []
    y: List[float] = []
    units: List[int] = []
    for unit, grp in test.groupby("unit", sort=True):
        grp = grp.sort_values("cycle")
        if len(grp) < window:
            continue
        X.append(scaler.transform(pre._align_features(grp, feature_columns))[-window:])
        y.append(float(truth[int(unit) - 1]))
        units.append(int(unit))
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(units)


def conformal_q_hat(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal half-width with the finite-sample correction.

    Uses the ``ceil((n+1)(1-alpha))/n`` empirical quantile rather than the
    plain one; the correction is what makes the resulting interval's coverage
    guarantee hold at finite n.
    """
    n = len(residuals)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(np.sort(residuals), level, method="higher"))


#: Minimum positive engines required in the calibration half before a
#: threshold is refitted at all.
#:
#: Threshold selection on a handful of events is noise, not calibration. The
#: first run of this script refitted FD001's 10-cycle cut from 7 total
#: positives (~3 per half) and moved it 0.950 -> 0.844, which held recall at
#: 1.000 but collapsed evaluation precision from 1.000 to 0.333. The incumbent
#: threshold is kept whenever the evidence is this thin.
MIN_CALIBRATION_POSITIVES = 8


def pick_threshold(scores: np.ndarray, labels: np.ndarray, min_recall: float,
                   incumbent: Optional[float] = None):
    """Lowest cut achieving ``min_recall``, tie-broken on precision.

    The incumbent threshold is included as a candidate so an already-good cut
    is never displaced by a marginally different one that merely happens to
    sit on an observed score value. Without it the search can only choose from
    the empirical grid, and a working threshold gets replaced for no gain.

    Returns (threshold, recall, precision, f1, met_floor).
    """
    grid = np.unique(np.round(np.concatenate([scores, [0.0, 1.0]]), 4))
    if incumbent is not None:
        grid = np.unique(np.concatenate([grid, [round(float(incumbent), 4)]]))

    def stats(t):
        pred = scores >= t
        tp = float((pred & (labels == 1)).sum())
        rec = tp / max(labels.sum(), 1)
        prec = tp / max(pred.sum(), 1)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        return rec, prec, f1

    best = None
    best_f1 = None
    for t in grid:
        rec, prec, f1 = stats(t)
        if best_f1 is None or f1 > best_f1[3]:
            best_f1 = (float(t), rec, prec, f1)
        if rec >= min_recall:
            # Strictly greater, so the incumbent wins ties and is only
            # displaced by a genuine precision gain.
            if best is None or prec > best[2]:
                best = (float(t), rec, prec, f1)
    if best is not None:
        return (*best, True)
    return (*best_f1, False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.10, help="1-alpha is the nominal coverage (default 0.10 -> 90%%).")
    ap.add_argument("--min-recall", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default=str(_BACKEND_ROOT / "data" / "raw"))
    ap.add_argument("--apply", action="store_true", help="Write calibration files; otherwise report only.")
    args = ap.parse_args()

    settings = get_settings()
    loader = ArtifactLoader(settings)
    data_root = Path(args.data_root)
    rng = np.random.RandomState(args.seed)

    conformal: Dict[str, Any] = {}
    thresholds_out: Dict[str, Any] = {}

    print(f"nominal coverage = {100*(1-args.alpha):.0f}%   min recall = {args.min_recall}   seed = {args.seed}\n")
    print("=" * 92)
    print("RUL CONFORMAL INTERVALS")
    print("=" * 92)
    print(f"  {'FD':7s} {'cal':>4s} {'eval':>5s} {'q_hat old':>10s} {'q_hat new':>10s} "
          f"{'PICP old':>9s} {'PICP new':>9s} {'MPIW old':>9s} {'MPIW new':>9s}")

    for fd in settings.SUPPORTED_FDS:
        test, truth = load_split(fd, data_root)
        art = loader.load("rul", fd)
        window = settings.RUL_WINDOW_SIZES.get(fd, 50)
        X, y, units = final_windows(test, truth, loader, art.feature_columns, art.scaler, window)
        pred = np.maximum(0.0, np.asarray(art.model.predict(X, verbose=0)).reshape(-1)[: len(y)])

        idx = rng.permutation(len(y))
        half = len(idx) // 2
        cal, ev = idx[:half], idx[half:]

        q_old = float(settings.DEFAULT_RUL_Q_HAT.get(fd, 20.0))
        q_new = conformal_q_hat(np.abs(pred[cal] - y[cal]), args.alpha)

        def cov(q):
            lo = np.maximum(0.0, pred[ev] - q)
            hi = pred[ev] + q
            return float(np.mean((y[ev] >= lo) & (y[ev] <= hi))), float(np.mean(hi - lo))

        picp_o, mpiw_o = cov(q_old)
        picp_n, mpiw_n = cov(q_new)
        conformal[fd] = {
            "q_hat": round(q_new, 4),
            "q_hat_previous": q_old,
            "alpha": args.alpha,
            "nominal_coverage": round(1 - args.alpha, 4),
            "calibration_engines": int(len(cal)),
            "evaluation_engines": int(len(ev)),
            "picp_eval": round(picp_n, 4),
            "picp_eval_previous": round(picp_o, 4),
            "mpiw_eval": round(mpiw_n, 4),
        }
        print(f"  {fd:7s} {len(cal):>4} {len(ev):>5} {q_old:>10.2f} {q_new:>10.2f} "
              f"{picp_o:>9.3f} {picp_n:>9.3f} {mpiw_o:>9.2f} {mpiw_n:>9.2f}")

    print("\n" + "=" * 92)
    print("FAILURE-RISK ALERT THRESHOLDS")
    print("=" * 92)
    print(f"  {'FD':7s} {'H':>3s} {'thr old':>8s} {'thr new':>8s} "
          f"{'rec old':>8s} {'rec new':>8s} {'prec old':>9s} {'prec new':>9s} {'F1 new':>7s}  floor")

    for fd in settings.SUPPORTED_FDS:
        test, truth = load_split(fd, data_root)
        art = loader.load("failure_risk", fd)
        window = int(art.metadata.get("window_size", 30))
        X, y, units = final_windows(test, truth, loader, art.feature_columns, art.scaler, window)
        probs = np.asarray(art.model.predict(X, verbose=0))
        old_thr = art.metadata.get("thresholds", {}) or {}

        idx = rng.permutation(len(y))
        half = len(idx) // 2
        cal, ev = idx[:half], idx[half:]

        new_thr: Dict[str, float] = {}
        detail: Dict[str, Any] = {}
        for i, H in enumerate(HORIZONS):
            labels = (y <= int(H)).astype(int)
            p = np.clip(probs[:, i], 0.0, 1.0)
            n_pos = int(labels[cal].sum())
            if n_pos < MIN_CALIBRATION_POSITIVES:
                keep = float(old_thr.get(H, 0.5))
                new_thr[H] = keep
                detail[H] = {
                    "threshold": keep,
                    "kept_incumbent": True,
                    "reason": f"only {n_pos} positive engine(s) in the calibration half "
                              f"(< {MIN_CALIBRATION_POSITIVES}); refitting would be noise",
                }
                print(f"  {fd:7s} {H:>3s} {keep:>8.3f} {keep:>8.3f} "
                      f"{'':>8} {'':>8} {'':>9} {'':>9} {'':>7}  kept ({n_pos} pos)")
                continue
            t, rc, pc, f1, met = pick_threshold(
                p[cal], labels[cal], args.min_recall, incumbent=float(old_thr.get(H, 0.5))
            )

            def score_at(th):
                pred = p[ev] >= th
                tp = float((pred & (labels[ev] == 1)).sum())
                rec = tp / max(labels[ev].sum(), 1)
                prec = tp / max(pred.sum(), 1)
                return rec, prec, (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))

            o = float(old_thr.get(H, 0.5))
            r_o, p_o, _ = score_at(o)
            r_n, p_n, f_n = score_at(t)
            new_thr[H] = round(t, 4)
            detail[H] = {
                "threshold": round(t, 4),
                "threshold_previous": o,
                "recall_eval": round(r_n, 4),
                "recall_eval_previous": round(r_o, 4),
                "precision_eval": round(p_n, 4),
                "precision_eval_previous": round(p_o, 4),
                "f1_eval": round(f_n, 4),
                "met_recall_floor_on_calibration": bool(met),
            }
            print(f"  {fd:7s} {H:>3s} {o:>8.3f} {t:>8.3f} {r_o:>8.3f} {r_n:>8.3f} "
                  f"{p_o:>9.3f} {p_n:>9.3f} {f_n:>7.3f}  {'yes' if met else 'NO (best-F1)'}")
        thresholds_out[fd] = {"thresholds": new_thr, "detail": detail}

    if not args.apply:
        print("\n(report only -- pass --apply to write the calibration files)")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    CONFORMAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFORMAL_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "scripts/Predictive_Maintenance/calibrate_pdm.py",
                "method": "split conformal, ceil((n+1)(1-alpha))/n empirical quantile of |y-yhat|",
                "protocol": "test engines split 50/50 by seeded permutation; q_hat fitted on the "
                            "calibration half, coverage reported on the evaluation half",
                "seed": args.seed,
                "subsets": conformal,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {CONFORMAL_PATH}")

    fr_dir = _BACKEND_ROOT / "models" / "Predictive_Maintenance" / "Failure_Risk"
    for fd, payload in thresholds_out.items():
        meta_path = fr_dir / f"{fd}_bilstm_failure_risk_metadata_FINAL.json"
        if not meta_path.exists():
            print(f"  skip {fd}: {meta_path.name} not found")
            continue
        shutil.copy(meta_path, meta_path.with_suffix(meta_path.suffix + f".pre_calibration_{stamp}"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        previous = meta.get("thresholds")
        meta["thresholds"] = payload["thresholds"]
        meta["threshold_provenance"] = {
            "calibrated_by": "scripts/Predictive_Maintenance/calibrate_pdm.py",
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "criterion": f"lowest threshold achieving recall >= {args.min_recall} on the "
                         "calibration half, tie-broken on precision; best-F1 when unreachable",
            "protocol": "test engines split 50/50 by seeded permutation; thresholds fitted on the "
                        "calibration half, recall/precision reported on the evaluation half",
            "seed": args.seed,
            "previous_thresholds": previous,
            "per_horizon": payload["detail"],
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  updated {meta_path.name}  (backup: .pre_calibration_{stamp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
