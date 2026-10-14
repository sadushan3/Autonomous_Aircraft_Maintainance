"""
One command that runs the whole system and tells you plainly what happened.

    python scripts/run_pipeline.py --input data/raw/test_FD002.txt

It drives the same four agents the API drives, in the same order, by calling
the same service functions -- Anomaly & Health, then RUL + Failure Risk, then
the Maintenance Supervisor fusion -- and then, when ground-truth RUL labels are
available, scores every agent against them.

Why this exists
---------------
The exchange CSVs are the contract between the four agents, and nothing about
them announces when they went stale. On 2026-08-25 the failure-risk thresholds
in ``supervisor_decisions.csv`` were ``0.97/0.81/0.61`` while the model's own
recalibrated metadata said ``0.0106/0.0151/0.0603``: the CSVs had been written
on Aug 23, the metadata was rewritten on Aug 24, and nothing flagged the gap.
The effect was that not one of the 22 engines within 10 cycles of failure was
flagged -- recall 0.00 -- with a model that scores 0.68 when read against its
own thresholds.

So this runner does three things the bare endpoint does not:

1. **Pre-flight provenance.** Before scoring anything it prints which model
   artefact each subset will use, what threshold each horizon resolves to, and
   where that number came from (artefact metadata / adaptive store / config
   default). A wrong threshold is visible before you spend an hour computing
   with it.
2. **A staleness guard.** If the existing exchange CSVs are older than the
   artefacts that produced them, it says so up front -- that is exactly the
   failure above.
3. **A run manifest.** ``data/exchange/pipeline_run.json`` records the input,
   the stride, the resolved thresholds and the row counts, so a later reader
   can tell what any given decision table was actually computed from.

Coverage
--------
``--stride`` controls how many cycles per engine get a real model call.
Stride 1 (the default here) scores every eligible cycle, which is what makes
the output *complete*: at stride 200 the previous run produced genuine
predictions for 416 of 33,991 rows and the Supervisor forward-filled the other
98.8% with a constant. Those filled rows measure R2 -0.81 against ground truth
while the genuine ones measure +0.40 -- the fill, not the model, was the error.
Stride 1 is slower; ``--stride N`` and ``--no-uncertainty`` are the levers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd  # noqa: E402

RAW_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

HORIZONS = ("10", "30", "50")


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

WIDTH = 78


def rule(char: str = "-") -> None:
    print(char * WIDTH)


def banner(text: str) -> None:
    rule("=")
    print(text)
    rule("=")


def stage(index: int, total: int, name: str) -> None:
    print()
    rule()
    print(f"STAGE {index}/{total}  {name}")
    rule()


def hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def infer_subset(path: Path) -> Optional[str]:
    """Read the FD subset out of a C-MAPSS filename (``test_FD002.txt`` -> FD002)."""
    name = path.stem.upper()
    for candidate in ("FD001", "FD002", "FD003", "FD004"):
        if candidate in name:
            return candidate
    return None


def resolve_thresholds(subset: str) -> Dict[str, Tuple[float, str]]:
    """Resolve each horizon's effective threshold and say where it came from.

    Mirrors ``FailureRiskService._thresholds``: the artefact metadata wins over
    the flat config default, and a stored adaptive value wins over both. The
    point of duplicating the precedence here is to make it visible BEFORE the
    run rather than inferable from the output afterwards.

    Returns:
        ``{horizon: (value, provenance)}``.
    """
    from app.config.Predictive_Maintenance.config import get_settings
    from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader

    settings = get_settings()
    loader = ArtifactLoader(settings)

    metadata: Dict[str, Any] = {}
    try:
        metadata = loader.load("failure_risk", subset).metadata or {}
    except Exception:  # noqa: BLE001 - absence is reported, not raised
        metadata = {}

    meta_thresholds = metadata.get("thresholds")
    config_defaults = settings.FAILURE_RISK_DEFAULT_THRESHOLDS.get(subset, {})

    store: Dict[str, Any] = {}
    store_path = (
        BACKEND_ROOT / "data" / "exchange" / "Predictive_Maintenance" / "adaptive_thresholds.json"
    )
    if store_path.exists():
        try:
            store = json.loads(store_path.read_text(encoding="utf-8")).get(subset, {}) or {}
        except Exception:  # noqa: BLE001
            store = {}

    resolved: Dict[str, Tuple[float, str]] = {}
    for horizon in HORIZONS:
        if isinstance(meta_thresholds, dict) and horizon in meta_thresholds:
            value, source = float(meta_thresholds[horizon]), "artefact metadata"
        elif horizon in config_defaults:
            value, source = float(config_defaults[horizon]), "config default"
        else:
            value, source = 0.5, "hardcoded fallback"

        entry = store.get(horizon)
        if isinstance(entry, dict) and "current" in entry:
            value, source = float(entry["current"]), "adaptive store (operator feedback)"

        resolved[horizon] = (value, source)
    return resolved


def preflight(input_path: Path, subset: str, exchange_dir: Path, stride: int) -> Dict[str, Any]:
    """Print artefact provenance and warn when existing outputs predate the models."""
    from app.config.Predictive_Maintenance.config import get_settings

    settings = get_settings()
    banner("PRE-FLIGHT")
    print(f"  input          : {input_path}")
    print(f"  subset         : {subset}")
    print(f"  stride         : {stride}" + ("  (every eligible cycle)" if stride == 1 else ""))
    print(f"  RUL window     : {settings.RUL_WINDOW_SIZES.get(subset, 50)} cycles")
    print(f"  risk window    : {settings.FAILURE_RISK_WINDOW_SIZE} cycles")

    artefact_dir = BACKEND_ROOT / "models" / "Predictive_Maintenance"
    newest_artefact = 0.0
    print("\n  Model artefacts")
    for label, pattern in (
        ("RUL", f"RUL_Lifecycle/{subset}_*.keras"),
        ("Failure Risk", f"Failure_Risk/{subset}_*.keras"),
        ("Risk metadata", f"Failure_Risk/{subset}_*metadata*.json"),
    ):
        matches = sorted(artefact_dir.glob(pattern))
        if not matches:
            print(f"    {label:14} MISSING  ({pattern})")
            continue
        for match in matches:
            mtime = match.stat().st_mtime
            newest_artefact = max(newest_artefact, mtime)
            stamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print(f"    {label:14} {match.name:52} {stamp}")

    thresholds = resolve_thresholds(subset)
    print("\n  Failure-risk thresholds that WILL be applied")
    for horizon in HORIZONS:
        value, source = thresholds[horizon]
        print(f"    <= {horizon:>2} cycles : {value:<10.4f} from {source}")

    stale: List[str] = []
    for name in ("supervisor_decisions.csv", "failure_risk_predictions.csv", "rul_predictions.csv"):
        path = exchange_dir / name
        if path.exists() and newest_artefact and path.stat().st_mtime < newest_artefact:
            stale.append(name)
    if stale:
        print("\n  [STALE] These existing outputs predate the model artefacts above:")
        for name in stale:
            print(f"          {name}")
        print("          They are about to be regenerated, which is the fix.")

    return {
        "thresholds": {h: {"value": v, "source": s} for h, (v, s) in thresholds.items()},
        "stale_outputs": stale,
    }


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------


def spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(values: List[float]) -> List[int]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0] * len(values)
        for position, index in enumerate(order):
            out[index] = position
        return out

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def validate(decisions_csv: Path, raw_path: Path, rul_truth_path: Path) -> Optional[Dict[str, Any]]:
    """Score every agent against C-MAPSS ground truth.

    Only meaningful for a *test* split, where ``RUL_FDxxx.txt`` gives the true
    remaining life at each engine's last observed cycle; true RUL at any earlier
    cycle is that value plus the cycles still to come.
    """
    if not (decisions_csv.exists() and rul_truth_path.exists()):
        return None

    frame = pd.read_csv(decisions_csv)
    last_cycle: Dict[str, int] = {}
    for line in raw_path.read_text().splitlines():
        parts = line.split()
        if parts:
            unit = str(int(float(parts[0])))
            last_cycle[unit] = max(last_cycle.get(unit, 0), int(float(parts[1])))

    truth_at_last = [int(float(x)) for x in rul_truth_path.read_text().split()]
    truth = {str(i + 1): truth_at_last[i] for i in range(len(truth_at_last))}
    if not set(truth) & set(last_cycle):
        return None

    frame["engine_key"] = frame["engine_id"].astype(int).astype(str)
    frame = frame[frame["engine_key"].isin(truth)].copy()
    if frame.empty:
        return None
    frame["true_rul"] = [
        truth[e] + (last_cycle[e] - int(c))
        for e, c in zip(frame["engine_key"], frame["cycle"])
    ]
    frame["true_capped"] = frame["true_rul"].clip(upper=125)

    final = frame.sort_values("cycle").groupby("engine_key").tail(1)

    report: Dict[str, Any] = {"engines": int(final.shape[0]), "rows": int(frame.shape[0])}

    # --- RUL ---
    valid = final.dropna(subset=["predicted_rul"])
    if not valid.empty:
        error = valid["predicted_rul"] - valid["true_capped"]
        ss_tot = ((valid["true_capped"] - valid["true_capped"].mean()) ** 2).sum()
        report["rul"] = {
            "mae": float(error.abs().mean()),
            "rmse": float(math.sqrt((error ** 2).mean())),
            "r2": float(1 - (error ** 2).sum() / ss_tot) if ss_tot else float("nan"),
            "bias": float(error.mean()),
        }
        if {"lower_bound", "upper_bound"}.issubset(valid.columns):
            inside = (
                (valid["lower_bound"] <= valid["true_rul"])
                & (valid["true_rul"] <= valid["upper_bound"])
            )
            report["rul"]["interval_coverage"] = float(inside.mean())

    # --- Failure risk ---
    risk: Dict[str, Any] = {}
    for horizon in HORIZONS:
        column = f"predicted_fail_in_{horizon}"
        if column not in final.columns:
            continue
        predicted = final[column].astype(str).str.lower().isin(["true", "1", "1.0"])
        actual = final["true_rul"] <= int(horizon)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        risk[horizon] = {
            "actual": tp + fn,
            "flagged": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    if risk:
        report["failure_risk"] = risk

    # --- Anomaly & health ---
    if "health_index" in frame.columns:
        health = frame.dropna(subset=["health_index"])
        within: List[float] = []
        for _, group in health.groupby("engine_key"):
            if len(group) > 10:
                value = spearman(
                    group["health_index"].tolist(), group["true_rul"].tolist()
                )
                if math.isfinite(value):
                    within.append(value)
        anomaly: Dict[str, Any] = {
            "final_cycle_spearman": spearman(
                final["health_index"].tolist(), final["true_rul"].tolist()
            ),
            "within_engine_median_spearman": (
                sorted(within)[len(within) // 2] if within else float("nan")
            ),
        }
        if "is_anomalous" in health.columns:
            flagged = health["is_anomalous"].astype(str).str.lower().isin(["true", "1", "1.0"])
            healthy_rows = health["true_rul"] > 80
            anomaly["flag_rate"] = float(flagged.mean())
            anomaly["false_alarm_rate_on_healthy"] = (
                float(flagged[healthy_rows].mean()) if healthy_rows.any() else float("nan")
            )
            if flagged.any():
                anomaly["mean_true_rul_flagged"] = float(health.loc[flagged, "true_rul"].mean())
            if (~flagged).any():
                anomaly["mean_true_rul_unflagged"] = float(
                    health.loc[~flagged, "true_rul"].mean()
                )
        report["anomaly_health"] = anomaly

    # --- Supervisor ---
    if "final_decision" in final.columns:
        by_decision = {
            str(name): {
                "engines": int(len(group)),
                "mean_true_rul": float(group["true_rul"].mean()),
                "min_true_rul": int(group["true_rul"].min()),
            }
            for name, group in final.groupby("final_decision")
        }
        unsafe = final[
            (final["true_rul"] <= 15)
            & (final["final_decision"].isin(["continue_operation", "monitor_closely"]))
        ]
        overkill = final[
            (final["true_rul"] > 100) & (final["final_decision"] == "immediate_maintenance")
        ]
        report["supervisor"] = {
            "by_decision": by_decision,
            "dangerous_misses": int(len(unsafe)),
            "dangerous_miss_engines": [int(e) for e in unsafe["engine_id"].tolist()[:10]],
            "false_urgency": int(len(overkill)),
        }

    return report


def print_validation(report: Dict[str, Any]) -> None:
    print(f"  Scored against ground truth: {report['engines']} engines, {report['rows']:,} rows\n")

    rul = report.get("rul")
    if rul:
        print("  RUL agent (final cycle per engine, truth capped at 125)")
        print(
            f"    MAE {fmt(rul['mae'])} cycles   RMSE {fmt(rul['rmse'])}   "
            f"R2 {fmt(rul['r2'], 3)}   bias {fmt(rul['bias'])}"
        )
        if "interval_coverage" in rul:
            print(f"    prediction-interval coverage {rul['interval_coverage']:.1%}")

    risk = report.get("failure_risk")
    if risk:
        print("\n  Failure-risk agent (final cycle per engine)")
        print(f"    {'horizon':<10}{'actual':>8}{'flagged':>9}{'TP':>5}{'FP':>5}{'FN':>5}"
              f"{'precision':>11}{'recall':>9}")
        for horizon in HORIZONS:
            entry = risk.get(horizon)
            if not entry:
                continue
            print(
                f"    <= {horizon:<7}{entry['actual']:>8}{entry['flagged']:>9}"
                f"{entry['tp']:>5}{entry['fp']:>5}{entry['fn']:>5}"
                f"{fmt(entry['precision']):>11}{fmt(entry['recall']):>9}"
            )

    anomaly = report.get("anomaly_health")
    if anomaly:
        print("\n  Anomaly & health agent")
        print(
            f"    health vs true RUL: final-cycle rho {fmt(anomaly['final_cycle_spearman'], 3)}, "
            f"within-engine median rho {fmt(anomaly['within_engine_median_spearman'], 3)}"
        )
        if "false_alarm_rate_on_healthy" in anomaly:
            print(
                f"    flag rate {anomaly['flag_rate']:.1%}   "
                f"false alarms on true-RUL>80 rows {anomaly['false_alarm_rate_on_healthy']:.1%}"
            )
        if "mean_true_rul_flagged" in anomaly:
            print(
                f"    mean true RUL: flagged {fmt(anomaly['mean_true_rul_flagged'], 1)} vs "
                f"unflagged {fmt(anomaly.get('mean_true_rul_unflagged'), 1)}"
            )

    supervisor = report.get("supervisor")
    if supervisor:
        print("\n  Maintenance supervisor (decisions should order by true RUL)")
        ordered = sorted(
            supervisor["by_decision"].items(), key=lambda kv: -kv[1]["mean_true_rul"]
        )
        for decision, entry in ordered:
            print(
                f"    {decision:<24} n={entry['engines']:>4}   "
                f"mean true RUL {entry['mean_true_rul']:>6.1f}   min {entry['min_true_rul']:>4}"
            )
        print(
            f"\n    dangerous misses (true RUL<=15 on continue/monitor): "
            f"{supervisor['dangerous_misses']}"
        )
        if supervisor["dangerous_miss_engines"]:
            print(f"      engines: {supervisor['dangerous_miss_engines']}")
        print(f"    false urgency (true RUL>100 on immediate): {supervisor['false_urgency']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full four-agent pipeline and report what it produced.",
    )
    parser.add_argument("--input", required=True, help="C-MAPSS .txt/.csv to score.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Score every Nth eligible cycle for RUL/Failure Risk. 1 (default) "
            "scores every cycle, so no row is a forward-filled placeholder."
        ),
    )
    parser.add_argument(
        "--no-uncertainty",
        action="store_true",
        help="Skip the Monte-Carlo dropout passes (much faster, zero uncertainty columns).",
    )
    parser.add_argument(
        "--truth",
        default=None,
        help="RUL_FDxxx.txt for ground-truth validation. Auto-detected beside --input.",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Run the agents but skip stage 4."
    )
    parser.add_argument(
        "--skip-anomaly",
        action="store_true",
        help=(
            "Reuse the anomaly outputs already in the exchange directory instead "
            "of re-running stage 1. Only valid when the last stage-1 run used "
            "THIS same input file -- the fusion joins on (subset, engine, cycle), "
            "so anomaly rows from a different file silently drop out of the merge."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (BACKEND_ROOT / input_path).resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    subset = infer_subset(input_path) or "FD001"

    from app.config.Anomaly_Health_Monitoring.config import get_config as get_ahm_config
    from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.data_loader import DataLoader
    from app.services.Integration.rul_failure_batch_service import run_rul_failure_batch
    from app.services.Integration.supervisor_bridge import (
        prepare_anomaly_agent_csv,
        run_supervisor_bridge,
    )

    ahm_config = get_ahm_config()
    exchange_dir = ahm_config.paths.data_exchange

    started = time.perf_counter()
    banner(f"AEROTWIN PIPELINE  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    preflight_report = preflight(input_path, subset, exchange_dir, args.stride)

    total_stages = 3 if args.skip_validation else 4
    manifest: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "fd_subset": subset,
        "stride": args.stride,
        "include_uncertainty": not args.no_uncertainty,
        "thresholds": preflight_report["thresholds"],
        "stale_outputs_replaced": preflight_report["stale_outputs"],
        "stages": {},
    }

    # --- Stage 1: Anomaly & Health -----------------------------------------
    stage(1, total_stages, "Anomaly & Health  (detection, health index, reasoning)")
    t0 = time.perf_counter()
    if args.skip_anomaly:
        existing = exchange_dir / ahm_config.paths.exchange_predictions_csv
        if not existing.exists():
            print(f"  --skip-anomaly given but {existing.name} is not there. Run stage 1 first.")
            return 2
        stamp = datetime.fromtimestamp(existing.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  SKIPPED -- reusing {existing.name} written {stamp}")
        manifest["stages"]["anomaly_health"] = {"status": "reused", "source": str(existing)}
        anomaly_report = {"status": "success", "reused": True}
    else:
        pipeline_module_path = (
            BACKEND_ROOT
            / "app"
            / "services"
            / "Anomaly_Health_Monitoring"
            / "pipeline"
            / "10_full_pipeline.py"
        )
        spec = importlib.util.spec_from_file_location("ahm_full_pipeline_cli", pipeline_module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ahm_full_pipeline_cli"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        anomaly_report = module.run_full_pipeline(
            ahm_config, retrain=False, include_shap=False, source_csv=str(input_path)
        )
        manifest["stages"]["anomaly_health"] = {
            "status": anomaly_report.get("status"),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"  status {anomaly_report.get('status')}  ({hms(time.perf_counter() - t0)})")
    if anomaly_report.get("status") != "success":
        print("  Anomaly & Health did not succeed; stopping before fusion.")
        return 1

    # --- Stage 2: RUL + Failure Risk ---------------------------------------
    stage(2, total_stages, "RUL + Failure Risk  (per-cycle model inference)")
    t0 = time.perf_counter()
    frame = DataLoader(ahm_config).load(source_csv=input_path)
    batch_report = run_rul_failure_batch(
        frame,
        output_dir=exchange_dir,
        stride=args.stride,
        include_uncertainty=not args.no_uncertainty,
    )
    manifest["stages"]["predictive_maintenance"] = {
        "status": batch_report.get("status"),
        "rul_rows": batch_report.get("rul_rows"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        f"  status {batch_report.get('status')}  "
        f"{batch_report.get('rul_rows')} scored point(s)  ({hms(time.perf_counter() - t0)})"
    )

    # --- Stage 3: Supervisor fusion ----------------------------------------
    stage(3, total_stages, "Maintenance Supervisor  (fusion + decisions)")
    t0 = time.perf_counter()
    engine_fd_subset: Dict[str, str] = {}
    if "subset_id" in frame.columns:
        engine_fd_subset = {
            str(int(unit)): str(value).upper()
            for unit, value in frame.groupby("unit")["subset_id"].first().items()
            if pd.notna(value)
        }
    anomaly_for_supervisor = exchange_dir / "anomaly_agent_outputs.csv"
    prepare_anomaly_agent_csv(
        exchange_dir / ahm_config.paths.exchange_predictions_csv,
        anomaly_for_supervisor,
        engine_fd_subset=engine_fd_subset,
    )
    decisions_csv = exchange_dir / "supervisor_decisions.csv"
    supervisor_report = run_supervisor_bridge(
        rul_csv=Path(batch_report["rul_predictions_csv"]),
        failure_risk_csv=Path(batch_report["failure_risk_predictions_csv"]),
        anomaly_csv=anomaly_for_supervisor,
        decisions_output_csv=decisions_csv,
    )
    manifest["stages"]["maintenance_supervisor"] = {
        "status": supervisor_report.get("status"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        f"  status {supervisor_report.get('status')}  ({hms(time.perf_counter() - t0)})"
    )

    # --- Coverage ----------------------------------------------------------
    coverage: Dict[str, Any] = {}
    if decisions_csv.exists():
        decisions = pd.read_csv(decisions_csv)
        genuine = pd.read_csv(batch_report["rul_predictions_csv"])
        keys = set(zip(genuine["engine_id"].astype(str), genuine["cycle"].astype(int)))
        rows = len(decisions)
        scored = sum(
            1
            for e, c in zip(decisions["engine_id"].astype(str), decisions["cycle"].astype(int))
            if (e, c) in keys
        )
        coverage = {
            "decision_rows": rows,
            "rows_with_a_real_rul_call": scored,
            "fraction": scored / rows if rows else 0.0,
            "engines": int(decisions["engine_id"].nunique()),
            "engines_without_any_real_call": int(
                decisions["engine_id"].nunique() - genuine["engine_id"].nunique()
            ),
        }
        manifest["coverage"] = coverage

    # --- Stage 4: Validation -----------------------------------------------
    validation: Optional[Dict[str, Any]] = None
    if not args.skip_validation:
        stage(4, total_stages, "Validation against C-MAPSS ground truth")
        truth_path = Path(args.truth) if args.truth else input_path.parent / f"RUL_{subset}.txt"
        if not truth_path.exists():
            print(f"  No ground-truth file at {truth_path} -- skipping.")
            print("  (Only test splits have one; a train split runs to failure instead.)")
        else:
            validation = validate(decisions_csv, input_path, truth_path)
            if validation is None:
                print("  Ground truth did not line up with these engines -- skipping.")
            else:
                print_validation(validation)
                manifest["validation"] = validation

    # --- Summary -----------------------------------------------------------
    elapsed = time.perf_counter() - started
    print()
    banner("SUMMARY")
    print(f"  input              {input_path.name}  ({subset})")
    if coverage:
        print(
            f"  decision rows      {coverage['decision_rows']:,} "
            f"across {coverage['engines']} engines"
        )
        print(
            f"  real model calls   {coverage['rows_with_a_real_rul_call']:,} "
            f"({coverage['fraction']:.1%} of rows)"
        )
        if coverage["fraction"] < 0.999:
            print(
                "                     the remainder are forward-filled by the "
                "Supervisor; re-run at --stride 1 for full coverage"
            )
        if coverage["engines_without_any_real_call"]:
            print(
                f"  engines never scored {coverage['engines_without_any_real_call']} "
                f"(shorter than the {subset} window)"
            )
    print("  thresholds applied " + ", ".join(
        f"<={h}: {preflight_report['thresholds'][h]['value']:.4f}" for h in HORIZONS
    ))
    print(f"  total time         {hms(elapsed)}")

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_seconds"] = round(elapsed, 1)
    manifest_path = exchange_dir / "pipeline_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"  manifest           {manifest_path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
