from __future__ import annotations

import functools
import json
import logging
import os
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config

T = TypeVar("T")

_LOGGING_CONFIGURED = False


class AnomalyHealthError(Exception):
    pass


class StageExecutionError(AnomalyHealthError):
    pass


class ArtifactNotFoundError(AnomalyHealthError):
    pass


def configure_logging(config: Optional[Config] = None, force: bool = False) -> None:


    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    cfg = config or get_config()
    cfg.ensure_directories()

    level = getattr(logging, str(cfg.mlops.log_level).upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-46s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if cfg.mlops.log_to_file:
        file_handler = logging.FileHandler(
            cfg.paths.logs / cfg.mlops.log_filename, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


    for noisy in ("matplotlib", "numba", "h5py", "urllib3", "git", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str, config: Optional[Config] = None) -> logging.Logger:


    configure_logging(config)
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, label: str) -> Iterator[None]:


    started = time.perf_counter()
    logger.info("START %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info("END   %s (%.2fs)", label, elapsed)


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:


    os.replace(tmp_path, final_path)


def atomic_write_csv(
    frame: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    float_format: Optional[str] = None,
    config: Optional[Config] = None,
) -> Path:


    cfg = config or get_config()
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp")

    fmt = float_format if float_format is not None else cfg.mlops.csv_float_format
    try:
        frame.to_csv(tmp_path, index=index, float_format=fmt)
        _atomic_replace(tmp_path, final_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise AnomalyHealthError(f"Failed to write CSV {final_path}: {exc}") from exc
    return final_path


def atomic_write_json(
    payload: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:


    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=_json_default)
        _atomic_replace(tmp_path, final_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise AnomalyHealthError(f"Failed to write JSON {final_path}: {exc}") from exc
    return final_path


def _json_default(value: Any) -> Any:


    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def read_json(path: str | Path, default: Optional[Any] = None) -> Any:


    file_path = Path(path)
    if not file_path.is_file():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger(__name__).warning("Could not read JSON %s: %s", file_path, exc)
        return default


def read_csv(
    path: str | Path,
    required_columns: Optional[Sequence[str]] = None,
    stage: str = "unknown stage",
) -> pd.DataFrame:


    file_path = Path(path)
    if not file_path.is_file():
        raise ArtifactNotFoundError(
            f"{stage}: required input '{file_path}' does not exist. "
            f"Run the preceding pipeline stage first."
        )
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        raise AnomalyHealthError(f"{stage}: could not parse {file_path}: {exc}") from exc

    if required_columns:
        missing = [c for c in required_columns if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(
                f"{stage}: '{file_path}' is missing column(s) {missing}"
            )
    return frame


def save_model(model: Any, path: str | Path, config: Optional[Config] = None) -> Path:


    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(final_path.name + ".tmp" + final_path.suffix)

    try:
        if final_path.suffix == ".keras":
            model.save(tmp_path)
        else:
            import joblib

            joblib.dump(model, tmp_path)
        _atomic_replace(tmp_path, final_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise AnomalyHealthError(f"Failed to save model {final_path}: {exc}") from exc
    return final_path


def load_model(path: str | Path) -> Any:


    file_path = Path(path)
    if not file_path.exists():
        raise ArtifactNotFoundError(
            f"Model artefact '{file_path}' not found. Train the corresponding "
            f"stage before running inference."
        )
    try:
        if file_path.suffix == ".keras":
            from tensorflow import keras

            return keras.models.load_model(file_path)
        import joblib

        return joblib.load(file_path)
    except Exception as exc:
        raise AnomalyHealthError(f"Failed to load model {file_path}: {exc}") from exc


def stage_guard(stage_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:


    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = get_logger(func.__module__)
            started = time.perf_counter()
            logger.info("=" * 78)
            logger.info("STAGE START : %s", stage_name)
            logger.info("=" * 78)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.exception(
                    "STAGE FAILED: %s after %.2fs -- %s. "
                    "No prior artefacts were modified.",
                    stage_name,
                    elapsed,
                    exc,
                )
                raise StageExecutionError(f"{stage_name} failed: {exc}") from exc
            elapsed = time.perf_counter() - started
            logger.info("STAGE OK    : %s (%.2fs)", stage_name, elapsed)
            return result

        return wrapper

    return decorator


def run_stage_cli(stage_callable: Callable[[], Any], stage_name: str) -> None:


    logger = get_logger(__name__)
    try:
        stage_callable()
    except Exception as exc:
        logger.error("Stage '%s' terminated with an error: %s", stage_name, exc)
        sys.exit(1)


def merge_on_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    how: str = "left",
    validate: Optional[str] = "one_to_one",
    suffixes: tuple = ("", "_dup"),
) -> pd.DataFrame:


    key_list = list(keys)
    overlapping = [
        c for c in right.columns if c in left.columns and c not in key_list
    ]
    right_trimmed = right.drop(columns=overlapping) if overlapping else right
    try:
        merged = left.merge(
            right_trimmed, on=key_list, how=how, validate=validate, suffixes=suffixes
        )
    except Exception as exc:
        raise AnomalyHealthError(
            f"Merge on {key_list} failed ({exc}). This usually means a stage "
            f"emitted duplicate (unit, cycle) rows."
        ) from exc
    return merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])


def robust_zscore(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:


    values = frame.loc[:, list(columns)].astype(float)
    median = values.median()
    mad = (values - median).abs().median()

    scale = (mad * 1.4826).replace(0.0, np.nan)
    scale = scale.fillna(values.std().replace(0.0, 1.0)).replace(0.0, 1.0)
    return (values - median) / scale


def minmax_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:


    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=float)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high - low < eps:
        return np.zeros_like(array, dtype=float)
    scaled = (array - low) / (high - low)
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0)


def empirical_percentile_normalize(
    reference_sorted: np.ndarray, values: np.ndarray
) -> np.ndarray:


    array = np.asarray(values, dtype=float)
    if reference_sorted.size == 0:
        return np.zeros_like(array, dtype=float)
    array = np.nan_to_num(array, nan=float(reference_sorted[0]))
    ranks = np.searchsorted(reference_sorted, array, side="right")
    return np.clip(ranks / float(reference_sorted.size), 0.0, 1.0)


def percentile_threshold(values: np.ndarray, percentile: float) -> float:


    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, percentile))


def rolling_slope(series: pd.Series, window: int) -> pd.Series:


    if window < 2:
        return pd.Series(np.zeros(len(series)), index=series.index)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float((x_centered**2).sum())
    if denominator == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    def _slope(window_values: np.ndarray) -> float:
        y = np.asarray(window_values, dtype=float)
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y, nan=float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0)
        return float((x_centered * (y - y.mean())).sum() / denominator)

    return (
        series.astype(float)
        .rolling(window=window, min_periods=window)
        .apply(_slope, raw=True)
        .fillna(0.0)
    )


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fallback: float = 0.0,
) -> np.ndarray:


    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(num, den)
    return np.where(np.isfinite(result), result, fallback)


def engine_wise_split(
    frame: pd.DataFrame,
    unit_column: str = "unit",
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple:


    units = np.sort(frame[unit_column].unique())
    if units.size == 0:
        return frame.copy(), frame.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    shuffled = units.copy()
    rng.shuffle(shuffled)

    n_validation = int(round(len(shuffled) * validation_fraction))
    n_validation = min(max(n_validation, 1 if len(shuffled) > 1 else 0), len(shuffled) - 1) \
        if len(shuffled) > 1 else 0

    validation_units = set(shuffled[:n_validation].tolist())
    is_validation = frame[unit_column].isin(validation_units)
    return frame.loc[~is_validation].copy(), frame.loc[is_validation].copy()


class ExperimentTracker:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.ExperimentTracker", self.config)
        self._mlflow: Any = None
        self._active = False

        if not self.config.mlops.enable_mlflow:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.paths.mlruns.as_uri())
            mlflow.set_experiment(self.config.mlops.mlflow_experiment_name)
            self._mlflow = mlflow
        except Exception as exc:
            self.logger.warning("MLflow unavailable, falling back to file logging: %s", exc)
            self._mlflow = None

    @contextmanager
    def run(self, run_name: str) -> Iterator["ExperimentTracker"]:


        if self._mlflow is not None:
            try:
                with self._mlflow.start_run(run_name=run_name):
                    self._active = True
                    yield self
                    self._active = False
                return
            except Exception as exc:
                self.logger.warning("MLflow run '%s' failed to start: %s", run_name, exc)
                self._active = False
        self.logger.info("Tracking run (local): %s", run_name)
        yield self

    def log_params(self, params: Dict[str, Any]) -> None:


        if self._mlflow is not None and self._active:
            try:
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                return
            except Exception as exc:
                self.logger.debug("MLflow log_params failed: %s", exc)
        self.logger.info("params: %s", json.dumps(params, default=str)[:2000])

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:


        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))
        }
        if self._mlflow is not None and self._active and clean:
            try:
                self._mlflow.log_metrics(clean, step=step)
                return
            except Exception as exc:
                self.logger.debug("MLflow log_metrics failed: %s", exc)
        self.logger.info("metrics: %s", json.dumps(clean, default=str)[:2000])

    def save_metrics(self, name: str, metrics: Dict[str, Any]) -> Path:


        payload = dict(metrics)
        payload["_recorded_at"] = datetime.now(timezone.utc).isoformat()
        return atomic_write_json(payload, self.config.paths.metrics / f"{name}.json")


class ModelRegistry:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.ModelRegistry", self.config)
        self.path = self.config.paths.models / self.config.paths.model_registry_json

    def load(self) -> Dict[str, Any]:


        return read_json(self.path, default={"models": {}}) or {"models": {}}

    def register(
        self,
        name: str,
        artifact_path: str | Path,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:


        registry = self.load()
        entry = {
            "name": name,
            "version": self.config.mlops.model_version_tag,
            "run_id": uuid.uuid4().hex[:12],
            "artifact_path": str(artifact_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "params": params or {},
            "extra": extra or {},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        registry.setdefault("models", {})[name] = entry
        atomic_write_json(registry, self.path)
        self.logger.info("Registered model '%s' -> %s", name, artifact_path)
        return entry

    def get(self, name: str) -> Optional[Dict[str, Any]]:


        return self.load().get("models", {}).get(name)


def utc_now_iso() -> str:


    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:


    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def summarize_frame(frame: pd.DataFrame, name: str) -> Dict[str, Any]:


    summary: Dict[str, Any] = {
        "name": name,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
    }
    if "unit" in frame.columns:
        summary["engines"] = int(frame["unit"].nunique())
    if "cycle" in frame.columns and len(frame):
        summary["max_cycle"] = int(frame["cycle"].max())
    return summary


def list_available_columns(frame: pd.DataFrame, wanted: Sequence[str]) -> List[str]:


    return [name for name in wanted if name in frame.columns]


def main() -> None:

    cfg = get_config()
    logger = get_logger(__name__, cfg)
    cfg.ensure_directories()

    demo = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "value": [10.0, 12.0, 9.0, 30.0],
        }
    )
    target = cfg.paths.processed / "_utils_smoke_test.csv"
    atomic_write_csv(demo, target)
    reloaded = read_csv(target, required_columns=["unit", "cycle"], stage="utils smoke test")
    logger.info("round-trip summary: %s", summarize_frame(reloaded, "demo"))
    logger.info("robust z:\n%s", robust_zscore(reloaded, ["value"]))
    train, validation = engine_wise_split(reloaded, validation_fraction=0.5)
    logger.info("engine-wise split -> train=%d rows, val=%d rows", len(train), len(validation))
    target.unlink(missing_ok=True)

    tracker = ExperimentTracker(cfg)
    with tracker.run("utils_smoke_test") as run:
        run.log_params({"demo": True})
        run.log_metrics({"rows": float(len(demo))})
    logger.info("utils smoke test complete")


if __name__ == "__main__":
    main()
