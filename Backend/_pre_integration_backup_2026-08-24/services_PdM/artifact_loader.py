from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.config.Predictive_Maintenance.config import Settings
from app.utils.Predictive_Maintenance.errors import (
    ArtifactNotFoundError,
    InvalidFDSubsetError,
    PredictionError,
)

AgentType = Literal["rul", "failure_risk"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactPaths:
    fd_subset: str
    agent_type: AgentType
    folder: Path
    model: Optional[Path]
    scaler: Optional[Path]
    feature_columns: Optional[Path]
    metadata: Optional[Path] = None
    manifest: Optional[Path] = None
    searched_locations: List[Path] = field(default_factory=list)


@dataclass
class LoadedArtifacts:
    paths: ArtifactPaths
    model: Any
    scaler: Any
    feature_columns: List[str]
    metadata: Dict[str, Any]


class ArtifactLoader:
    """Discovers and lazily loads model artifacts from the configured root."""

    RUL_PATTERNS = {
        "model": ["*best_tcn*.keras", "*tcn*.keras"],
        "scaler": ["*scaler*.pkl"],
        "feature_columns": ["*feature_columns*.json"],
        "manifest": ["*manifest*.json"],
    }
    FAILURE_RISK_PATTERNS = {
        "model": ["*bilstm*.keras", "*failure_risk*.keras"],
        "scaler": ["*scaler*.pkl"],
        "feature_columns": ["*feature_columns*.json"],
        "metadata": ["*metadata*.json"],
        "manifest": ["*manifest*.json"],
    }

    # Tried in priority order. HistGradientBoosting checkpoints exported from some
    # scikit-learn/numpy combinations fail to unpickle here (RandomState/PCG64
    # bit-generator mismatch), so it is listed last and load_baseline_model()
    # skips any candidate that raises instead of hard-failing the whole agent.
    BASELINE_MODEL_PATTERNS = [
        ("RandomForest", "*RandomForest_baseline*.pkl"),
        ("LinearRegression", "*LinearRegression_baseline*.pkl"),
        ("HistGradientBoosting", "*HistGradientBoosting_baseline*.pkl"),
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.project_root = Path(__file__).resolve().parents[2]
        self.artifact_root = self._resolve_configured_root(settings.ARTIFACT_ROOT)
        self._cache: Dict[str, LoadedArtifacts] = {}
        self._baseline_cache: Dict[str, Any] = {}

    def validate_fd_subset(self, fd_subset: str) -> str:
        normalized = fd_subset.upper()
        if normalized not in self.settings.SUPPORTED_FDS:
            raise InvalidFDSubsetError(
                f"Unsupported FD subset '{fd_subset}'. Supported values: "
                f"{', '.join(self.settings.SUPPORTED_FDS)}."
            )
        return normalized

    def discover(self, agent_type: AgentType, fd_subset: str) -> ArtifactPaths:
        fd_subset = self.validate_fd_subset(fd_subset)
        patterns = (
            self.RUL_PATTERNS if agent_type == "rul" else self.FAILURE_RISK_PATTERNS
        )

        found: Dict[str, Optional[Path]] = {}
        searched_locations = self._candidate_folders(agent_type, fd_subset)
        for key, key_patterns in patterns.items():
            found[key] = self._find_first_match(
                folders=searched_locations,
                patterns=key_patterns,
                fd_subset=fd_subset,
            )

        folder = self._best_folder(found, searched_locations)
        logger.info(
            "Artifact discovery %s/%s: model=%s scaler=%s feature_columns=%s metadata=%s",
            agent_type,
            fd_subset,
            found.get("model"),
            found.get("scaler"),
            found.get("feature_columns"),
            found.get("metadata"),
        )

        return ArtifactPaths(
            fd_subset=fd_subset,
            agent_type=agent_type,
            folder=folder,
            model=found.get("model"),
            scaler=found.get("scaler"),
            feature_columns=found.get("feature_columns"),
            metadata=found.get("metadata"),
            manifest=found.get("manifest"),
            searched_locations=searched_locations,
        )

    def load(self, agent_type: AgentType, fd_subset: str) -> LoadedArtifacts:
        fd_subset = self.validate_fd_subset(fd_subset)
        cache_key = f"{agent_type}:{fd_subset}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        paths = self.discover(agent_type, fd_subset)
        self._raise_if_required_missing(paths)

        try:
            model = self._load_keras_model(paths.model)
            scaler = self._load_joblib(paths.scaler)
            feature_columns = self._load_json_list(paths.feature_columns)
            metadata = self._load_json_dict(paths.metadata) if paths.metadata else {}
        except ArtifactNotFoundError:
            raise
        except Exception as exc:
            raise PredictionError(
                f"Failed to load {agent_type} artifacts for {fd_subset}: {exc}"
            ) from exc

        loaded = LoadedArtifacts(
            paths=paths,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
        )
        self._cache[cache_key] = loaded
        return loaded

    def load_baseline_model(self, fd_subset: str) -> Optional[Dict[str, Any]]:
        """Loads the tabular fallback regressor used by the self-correction gate.

        Tries each baseline checkpoint in priority order and returns the first
        one that both exists and unpickles cleanly, so a single incompatible
        checkpoint (see BASELINE_MODEL_PATTERNS) never disables the fallback
        path for a whole FD subset if a sibling baseline is usable.
        """
        fd_subset = self.validate_fd_subset(fd_subset)
        cache_key = f"baseline:{fd_subset}"
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        folders = self._candidate_folders("rul", fd_subset)
        result: Optional[Dict[str, Any]] = None
        for label, pattern in self.BASELINE_MODEL_PATTERNS:
            path = self._find_first_match(folders=folders, patterns=[pattern], fd_subset=fd_subset)
            if path is None:
                continue
            try:
                model = self._load_joblib(path)
            except Exception as exc:
                logger.warning("Baseline model %s at %s failed to load: %s", label, path, exc)
                continue
            result = {"label": label, "model": model, "path": path}
            break

        self._baseline_cache[cache_key] = result
        return result

    def status(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "rul_lifecycle": [
                self._status_for("rul", fd_subset)
                for fd_subset in self.settings.SUPPORTED_FDS
            ],
            "failure_risk": [
                self._status_for("failure_risk", fd_subset)
                for fd_subset in self.settings.SUPPORTED_FDS
            ],
        }

    def _status_for(self, agent_type: AgentType, fd_subset: str) -> Dict[str, Any]:
        paths = self.discover(agent_type, fd_subset)
        missing = self._missing_required(paths)
        path_map = {
            "folder": str(paths.folder),
            "model": str(paths.model) if paths.model else None,
            "scaler": str(paths.scaler) if paths.scaler else None,
            "feature_columns": str(paths.feature_columns) if paths.feature_columns else None,
            "metadata": str(paths.metadata) if paths.metadata else None,
            "manifest": str(paths.manifest) if paths.manifest else None,
        }
        return {
            "fd_subset": fd_subset,
            "loaded": len(missing) == 0,
            "missing_model": paths.model is None,
            "missing_scaler": paths.scaler is None,
            "missing_feature_columns": paths.feature_columns is None,
            "missing_metadata": paths.metadata is None,
            "missing": missing,
            "paths": path_map,
            "searched_locations": [str(path) for path in paths.searched_locations or []],
        }

    def _missing_required(self, paths: ArtifactPaths) -> List[str]:
        missing = []
        if paths.model is None:
            missing.append("model")
        if paths.scaler is None:
            missing.append("scaler")
        if paths.feature_columns is None:
            missing.append("feature_columns")
        return missing

    def _raise_if_required_missing(self, paths: ArtifactPaths) -> None:
        missing = self._missing_required(paths)
        if not missing:
            return

        patterns = (
            self.RUL_PATTERNS if paths.agent_type == "rul" else self.FAILURE_RISK_PATTERNS
        )
        searched = ", ".join(
            f"{name}={patterns_for_key}"
            for name, patterns_for_key in patterns.items()
        )
        locations = ", ".join(str(path) for path in paths.searched_locations or [])
        raise ArtifactNotFoundError(
            f"Missing required {paths.agent_type} artifact(s) for {paths.fd_subset}: "
            f"{', '.join(missing)}. Searched locations: {locations}. "
            f"Patterns: {searched}."
        )

    @staticmethod
    def _load_keras_model(path: Optional[Path]) -> Any:
        if path is None:
            raise ArtifactNotFoundError("Keras model path is missing.")
        try:
            import tensorflow as tf
        except Exception as exc:
            raise PredictionError(
                "TensorFlow is required to load .keras model artifacts. "
                "Install dependencies from requirements.txt."
            ) from exc
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as first_error:
            class MCDropout(tf.keras.layers.Dropout):
                def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
                    return super().call(inputs, training=training)

            class CompatDense(tf.keras.layers.Dense):
                @classmethod
                def from_config(cls, config: Dict[str, Any]) -> Any:
                    sanitized = dict(config)
                    sanitized.pop("quantization_config", None)
                    return super().from_config(sanitized)

            custom_objects = {
                "Dense": CompatDense,
                "MCDropout": MCDropout,
                "MCConcreteDropout": MCDropout,
            }
            original_dense_from_config = tf.keras.layers.Dense.from_config

            def dense_from_config_without_quantization(
                cls: Any, config: Dict[str, Any]
            ) -> Any:
                sanitized = dict(config)
                sanitized.pop("quantization_config", None)
                return original_dense_from_config(sanitized)

            tf.keras.layers.Dense.from_config = classmethod(
                dense_from_config_without_quantization
            )
            try:
                return tf.keras.models.load_model(
                    path,
                    compile=False,
                    custom_objects=custom_objects,
                )
            except Exception as second_error:
                raise PredictionError(
                    f"Failed to deserialize Keras model '{path}'. First error: "
                    f"{first_error}. Retry with custom_objects failed: {second_error}."
                ) from second_error
            finally:
                tf.keras.layers.Dense.from_config = original_dense_from_config

    def _resolve_configured_root(self, configured_root: str) -> Path:
        root = Path(configured_root)
        if root.is_absolute():
            return root
        return self.project_root / root

    def _candidate_roots(self) -> List[Path]:
        raw_roots = [
            self.artifact_root,
            self.project_root / "models",
            self.project_root.parent / "models",
            Path.cwd() / "artifacts",
            Path.cwd() / "models",
        ]
        seen: set[Path] = set()
        roots: List[Path] = []
        for root in raw_roots:
            resolved = root.resolve()
            if resolved not in seen:
                roots.append(resolved)
                seen.add(resolved)
        return roots

    def _candidate_folders(self, agent_type: AgentType, fd_subset: str) -> List[Path]:
        agent_dir = "RUL_Lifecycle" if agent_type == "rul" else "Failure_Risk"
        folders: List[Path] = []
        for root in self._candidate_roots():
            component_folder = root / agent_dir
            folders.append(component_folder / fd_subset)
            folders.append(component_folder)
        return self._dedupe_paths(folders)

    @staticmethod
    def _dedupe_paths(paths: List[Path]) -> List[Path]:
        seen: set[Path] = set()
        deduped: List[Path] = []
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                deduped.append(resolved)
                seen.add(resolved)
        return deduped

    @staticmethod
    def _find_first_match(
        folders: List[Path],
        patterns: List[str],
        fd_subset: str,
    ) -> Optional[Path]:
        for folder in folders:
            if not folder.exists():
                continue
            matches: List[Path] = []
            for pattern in patterns:
                matches.extend(folder.glob(pattern))
            fd_matches = [
                match
                for match in sorted(matches)
                if fd_subset.lower() in match.name.lower()
            ]
            if fd_matches:
                return fd_matches[0]
            if matches and folder.name.upper() == fd_subset:
                return sorted(matches)[0]
        return None

    @staticmethod
    def _best_folder(
        found: Dict[str, Optional[Path]],
        searched_locations: List[Path],
    ) -> Path:
        for path in found.values():
            if path is not None:
                return path.parent
        return searched_locations[0] if searched_locations else Path(".")

    @staticmethod
    def _load_joblib(path: Optional[Path]) -> Any:
        if path is None:
            raise ArtifactNotFoundError("Scaler path is missing.")
        try:
            import joblib
        except Exception as exc:
            raise PredictionError(
                "joblib is required to load scaler artifacts. "
                "Install dependencies from requirements.txt."
            ) from exc
        return joblib.load(path)

    @staticmethod
    def _load_json_list(path: Optional[Path]) -> List[str]:
        if path is None:
            raise ArtifactNotFoundError("Feature columns JSON path is missing.")
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise PredictionError(f"Feature columns file '{path}' must contain a JSON string list.")
        return data

    @staticmethod
    def _load_json_dict(path: Optional[Path]) -> Dict[str, Any]:
        if path is None:
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise PredictionError(f"Metadata file '{path}' must contain a JSON object.")
        return data
