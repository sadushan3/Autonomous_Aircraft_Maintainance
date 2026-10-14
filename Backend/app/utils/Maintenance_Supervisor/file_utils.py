"""
file_utils.py

Safe file-handling utilities for the Autonomous Maintenance Supervisor Agent.

This module provides:

- Directory creation
- File existence validation
- Safe CSV reading and writing
- Safe JSON reading and writing
- Parquet reading and writing
- Joblib model persistence
- Atomic file replacement
- File hashing
- Timestamped backups
- Dataset metadata collection

The functions in this file are shared by preprocessing, training, evaluation,
explainability, confidence scoring, decision fusion, and API inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import pandas as pd


# =============================================================================
# IMPORT BOOTSTRAP
# =============================================================================

try:
    from app.utils.Maintenance_Supervisor.logger import get_logger
except ModuleNotFoundError:
    CURRENT_FILE = Path(__file__).resolve()
    BACKEND_ROOT = CURRENT_FILE.parents[3]

    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    from app.utils.Maintenance_Supervisor.logger import get_logger


LOGGER = get_logger()


# =============================================================================
# GENERAL PATH UTILITIES
# =============================================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Resolved directory path.
    """

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Create the parent directory of a file path.

    Parameters
    ----------
    path:
        Target file path.

    Returns
    -------
    Path
        Resolved file path.
    """

    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def validate_file_exists(
    path: str | Path,
    description: str = "File",
) -> Path:
    """
    Validate that a file exists.

    Parameters
    ----------
    path:
        File path to validate.
    description:
        Human-readable file description.

    Returns
    -------
    Path
        Resolved file path.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    IsADirectoryError
        If the path is a directory.
    """

    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"{description} is not a file: {file_path}"
        )

    return file_path


def validate_directory_exists(
    path: str | Path,
    description: str = "Directory",
) -> Path:
    """
    Validate that a directory exists.

    Parameters
    ----------
    path:
        Directory path to validate.
    description:
        Human-readable directory description.

    Returns
    -------
    Path
        Resolved directory path.
    """

    directory = Path(path).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"{description} is not a directory: {directory}"
        )

    return directory


def is_file_empty(path: str | Path) -> bool:
    """
    Return True when a file exists but contains zero bytes.
    """

    file_path = validate_file_exists(path)

    return file_path.stat().st_size == 0


# =============================================================================
# ATOMIC TEMPORARY FILE SUPPORT
# =============================================================================

def _temporary_file_path(
    target_path: Path,
    suffix: str | None = None,
) -> Path:
    """
    Create a temporary file path in the same directory as the target.

    Keeping the temporary file in the same directory helps ensure that
    os.replace remains atomic on the same filesystem.
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_suffix = suffix or target_path.suffix

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}_",
        suffix=file_suffix,
        dir=target_path.parent,
    )

    os.close(file_descriptor)

    return Path(temp_name)


def atomic_replace(
    temporary_path: str | Path,
    target_path: str | Path,
) -> Path:
    """
    Atomically replace a target file with a temporary file.

    Existing target files are replaced only after the temporary file has
    been written successfully.
    """

    temp_path = validate_file_exists(
        temporary_path,
        description="Temporary file",
    )

    final_path = ensure_parent_directory(target_path)

    os.replace(temp_path, final_path)

    return final_path


# =============================================================================
# CSV UTILITIES
# =============================================================================

def read_csv(
    path: str | Path,
    required_columns: Sequence[str] | None = None,
    usecols: Sequence[str] | None = None,
    dtype: Mapping[str, Any] | None = None,
    parse_dates: Sequence[str] | None = None,
    low_memory: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Safely read a CSV file.

    Parameters
    ----------
    path:
        CSV file path.
    required_columns:
        Columns that must exist after loading.
    usecols:
        Optional subset of columns to load.
    dtype:
        Optional pandas dtype mapping.
    parse_dates:
        Optional date columns.
    low_memory:
        Passed to pandas.read_csv.
    kwargs:
        Additional pandas.read_csv arguments.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """

    file_path = validate_file_exists(
        path,
        description="CSV file",
    )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"CSV file is empty: {file_path}"
        )

    LOGGER.info(f"Reading CSV file: {file_path}")

    try:
        dataframe = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=low_memory,
            **kwargs,
        )
    except Exception:
        LOGGER.exception(
            f"Failed to read CSV file: {file_path}"
        )
        raise

    if required_columns:
        validate_required_columns(
            dataframe=dataframe,
            required_columns=required_columns,
            dataset_name=file_path.name,
        )

    LOGGER.info(
        "CSV loaded successfully | "
        f"rows={len(dataframe):,} | "
        f"columns={len(dataframe.columns):,}"
    )

    return dataframe


def read_csv_in_chunks(
    path: str | Path,
    chunk_size: int,
    required_columns: Sequence[str] | None = None,
    usecols: Sequence[str] | None = None,
    dtype: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Iterable[pd.DataFrame]:
    """
    Yield a CSV dataset in chunks.

    Parameters
    ----------
    path:
        CSV file path.
    chunk_size:
        Number of rows per chunk.
    required_columns:
        Required columns for every chunk.
    usecols:
        Optional columns to read.
    dtype:
        Optional dtype mapping.
    kwargs:
        Additional pandas.read_csv arguments.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    file_path = validate_file_exists(
        path,
        description="CSV file",
    )

    LOGGER.info(
        f"Reading CSV in chunks: {file_path} | "
        f"chunk_size={chunk_size:,}"
    )

    reader = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
        **kwargs,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        if required_columns:
            validate_required_columns(
                dataframe=chunk,
                required_columns=required_columns,
                dataset_name=f"{file_path.name} chunk {chunk_number}",
            )

        LOGGER.info(
            f"Loaded chunk {chunk_number:,} | "
            f"rows={len(chunk):,}"
        )

        yield chunk


def write_csv_atomic(
    dataframe: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> Path:
    """
    Write a dataframe to CSV atomically.

    The existing output file remains untouched if writing fails.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    target_path = ensure_parent_directory(path)
    temporary_path = _temporary_file_path(
        target_path,
        suffix=".csv",
    )

    LOGGER.info(
        f"Writing CSV atomically: {target_path} | "
        f"rows={len(dataframe):,} | "
        f"columns={len(dataframe.columns):,}"
    )

    try:
        dataframe.to_csv(
            temporary_path,
            index=index,
            encoding=encoding,
            **kwargs,
        )

        if temporary_path.stat().st_size == 0:
            raise IOError(
                "Temporary CSV file was created but is empty."
            )

        atomic_replace(
            temporary_path=temporary_path,
            target_path=target_path,
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

        LOGGER.exception(
            f"Failed to write CSV file: {target_path}"
        )
        raise

    LOGGER.info(
        f"CSV saved successfully: {target_path}"
    )

    return target_path


# =============================================================================
# JSON UTILITIES
# =============================================================================

def read_json(
    path: str | Path,
) -> Any:
    """
    Read a JSON file.
    """

    file_path = validate_file_exists(
        path,
        description="JSON file",
    )

    LOGGER.info(f"Reading JSON file: {file_path}")

    try:
        with file_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        LOGGER.exception(
            f"Failed to read JSON file: {file_path}"
        )
        raise


def write_json_atomic(
    data: Any,
    path: str | Path,
    indent: int = 4,
    sort_keys: bool = False,
) -> Path:
    """
    Write JSON data atomically.
    """

    target_path = ensure_parent_directory(path)
    temporary_path = _temporary_file_path(
        target_path,
        suffix=".json",
    )

    LOGGER.info(
        f"Writing JSON atomically: {target_path}"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=False,
                default=_json_serializer,
            )

        if temporary_path.stat().st_size == 0:
            raise IOError(
                "Temporary JSON file was created but is empty."
            )

        atomic_replace(
            temporary_path=temporary_path,
            target_path=target_path,
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)

        LOGGER.exception(
            f"Failed to write JSON file: {target_path}"
        )
        raise

    LOGGER.info(
        f"JSON saved successfully: {target_path}"
    )

    return target_path


def _json_serializer(value: Any) -> Any:
    """
    Convert common non-JSON-native objects into serializable values.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


# =============================================================================
# PARQUET UTILITIES
# =============================================================================

def read_parquet(
    path: str | Path,
    required_columns: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a Parquet file safely.
    """

    file_path = validate_file_exists(
        path,
        description="Parquet file",
    )

    LOGGER.info(f"Reading Parquet file: {file_path}")

    try:
        dataframe = pd.read_parquet(
            file_path,
            columns=columns,
            **kwargs,
        )

    except ImportError as exception:
        raise ImportError(
            "Parquet support requires pyarrow or fastparquet. "
            "Install pyarrow using: pip install pyarrow"
        ) from exception

    except Exception:
        LOGGER.exception(
            f"Failed to read Parquet file: {file_path}"
        )
        raise

    if required_columns:
        validate_required_columns(
            dataframe=dataframe,
            required_columns=required_columns,
            dataset_name=file_path.name,
        )

    LOGGER.info(
        "Parquet loaded successfully | "
        f"rows={len(dataframe):,} | "
        f"columns={len(dataframe.columns):,}"
    )

    return dataframe


def write_parquet_atomic(
    dataframe: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    compression: str = "snappy",
    **kwargs: Any,
) -> Path:
    """
    Write a dataframe to Parquet atomically.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    target_path = ensure_parent_directory(path)
    temporary_path = _temporary_file_path(
        target_path,
        suffix=".parquet",
    )

    LOGGER.info(
        f"Writing Parquet atomically: {target_path} | "
        f"rows={len(dataframe):,} | "
        f"columns={len(dataframe.columns):,}"
    )

    try:
        dataframe.to_parquet(
            temporary_path,
            index=index,
            compression=compression,
            **kwargs,
        )

        if temporary_path.stat().st_size == 0:
            raise IOError(
                "Temporary Parquet file was created but is empty."
            )

        atomic_replace(
            temporary_path=temporary_path,
            target_path=target_path,
        )

    except ImportError as exception:
        temporary_path.unlink(missing_ok=True)

        raise ImportError(
            "Parquet support requires pyarrow or fastparquet. "
            "Install pyarrow using: pip install pyarrow"
        ) from exception

    except Exception:
        temporary_path.unlink(missing_ok=True)

        LOGGER.exception(
            f"Failed to write Parquet file: {target_path}"
        )
        raise

    LOGGER.info(
        f"Parquet saved successfully: {target_path}"
    )

    return target_path


# =============================================================================
# JOBLIB MODEL UTILITIES
# =============================================================================

def save_joblib_atomic(
    object_to_save: Any,
    path: str | Path,
    compress: int = 3,
) -> Path:
    """
    Save a Python object using joblib with atomic replacement.
    """

    target_path = ensure_parent_directory(path)
    temporary_path = _temporary_file_path(
        target_path,
        suffix=target_path.suffix or ".joblib",
    )

    LOGGER.info(
        f"Saving joblib artifact: {target_path}"
    )

    try:
        joblib.dump(
            object_to_save,
            temporary_path,
            compress=compress,
        )

        if temporary_path.stat().st_size == 0:
            raise IOError(
                "Temporary joblib artifact is empty."
            )

        atomic_replace(
            temporary_path=temporary_path,
            target_path=target_path,
        )

    except Exception:
        temporary_path.unlink(missing_ok=True)

        LOGGER.exception(
            f"Failed to save joblib artifact: {target_path}"
        )
        raise

    LOGGER.info(
        f"Joblib artifact saved successfully: {target_path}"
    )

    return target_path


def load_joblib(
    path: str | Path,
) -> Any:
    """
    Load a joblib artifact.
    """

    file_path = validate_file_exists(
        path,
        description="Joblib artifact",
    )

    LOGGER.info(
        f"Loading joblib artifact: {file_path}"
    )

    try:
        artifact = joblib.load(file_path)

    except Exception:
        LOGGER.exception(
            f"Failed to load joblib artifact: {file_path}"
        )
        raise

    LOGGER.info(
        f"Joblib artifact loaded successfully: {file_path}"
    )

    return artifact


# =============================================================================
# DATAFRAME VALIDATION
# =============================================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that required columns are present in a dataframe.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    missing_columns = sorted(
        set(required_columns).difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def validate_no_duplicate_columns(
    dataframe: pd.DataFrame,
    dataset_name: str = "dataset",
) -> None:
    """
    Validate that a dataframe does not contain duplicate column names.
    """

    duplicate_mask = dataframe.columns.duplicated()

    if duplicate_mask.any():
        duplicate_columns = dataframe.columns[
            duplicate_mask
        ].tolist()

        raise ValueError(
            f"{dataset_name} contains duplicate column names: "
            f"{duplicate_columns}"
        )


# =============================================================================
# BACKUP AND VERSIONING
# =============================================================================

def create_timestamped_backup(
    path: str | Path,
    backup_directory: str | Path | None = None,
) -> Path:
    """
    Create a timestamped backup copy of a file.

    The source file is not modified.
    """

    source_path = validate_file_exists(
        path,
        description="Source file",
    )

    if backup_directory is None:
        backup_root = source_path.parent / "backups"
    else:
        backup_root = Path(
            backup_directory
        ).expanduser().resolve()

    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup_name = (
        f"{source_path.stem}_{timestamp}"
        f"{source_path.suffix}"
    )

    backup_path = backup_root / backup_name

    shutil.copy2(
        source_path,
        backup_path,
    )

    LOGGER.info(
        f"Backup created: {backup_path}"
    )

    return backup_path


# =============================================================================
# FILE HASHING AND METADATA
# =============================================================================

def calculate_sha256(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the SHA-256 checksum of a file.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    file_path = validate_file_exists(path)

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_metadata(
    path: str | Path,
    include_hash: bool = True,
) -> dict[str, Any]:
    """
    Return useful metadata for a file.
    """

    file_path = validate_file_exists(path)
    file_stat = file_path.stat()

    metadata: dict[str, Any] = {
        "name": file_path.name,
        "path": str(file_path),
        "suffix": file_path.suffix,
        "size_bytes": file_stat.st_size,
        "size_megabytes": round(
            file_stat.st_size / (1024 ** 2),
            6,
        ),
        "modified_utc": datetime.fromtimestamp(
            file_stat.st_mtime,
            timezone.utc,
        ).isoformat(),
    }

    if include_hash:
        metadata["sha256"] = calculate_sha256(
            file_path
        )

    return metadata


def get_dataframe_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Collect high-level dataframe metadata.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    memory_bytes = int(
        dataframe.memory_usage(
            index=True,
            deep=True,
        ).sum()
    )

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "memory_bytes": memory_bytes,
        "memory_megabytes": round(
            memory_bytes / (1024 ** 2),
            6,
        ),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
    }


# =============================================================================
# TEXT FILE UTILITIES
# =============================================================================

def write_text_atomic(
    content: str,
    path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """
    Write a text file atomically.
    """

    if not isinstance(content, str):
        raise TypeError(
            "content must be a string."
        )

    target_path = ensure_parent_directory(path)
    temporary_path = _temporary_file_path(
        target_path,
        suffix=target_path.suffix or ".txt",
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding=encoding,
        ) as file:
            file.write(content)

        atomic_replace(
            temporary_path=temporary_path,
            target_path=target_path,
        )

    except Exception:
        temporary_path.unlink(missing_ok=True)

        LOGGER.exception(
            f"Failed to write text file: {target_path}"
        )
        raise

    LOGGER.info(
        f"Text file saved successfully: {target_path}"
    )

    return target_path


def read_text(
    path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
    Read a text file.
    """

    file_path = validate_file_exists(path)

    with file_path.open(
        mode="r",
        encoding=encoding,
    ) as file:
        return file.read()


# =============================================================================
# TEST ENTRY POINT
# =============================================================================

def _run_self_test() -> None:
    """
    Run a small self-test for the utility functions.
    """

    LOGGER.info("=" * 72)
    LOGGER.info("FILE UTILS SELF-TEST")
    LOGGER.info("=" * 72)

    test_root = Path(__file__).resolve().parent / "_file_utils_test"
    ensure_directory(test_root)

    test_dataframe = pd.DataFrame(
        {
            "engine_id": ["ENG_001", "ENG_001", "ENG_002"],
            "cycle": [1, 2, 1],
            "predicted_rul": [120.0, 119.0, 95.0],
        }
    )

    csv_path = test_root / "test_data.csv"
    json_path = test_root / "test_metadata.json"
    joblib_path = test_root / "test_object.joblib"
    text_path = test_root / "test_report.txt"

    write_csv_atomic(
        dataframe=test_dataframe,
        path=csv_path,
    )

    loaded_dataframe = read_csv(
        path=csv_path,
        required_columns=[
            "engine_id",
            "cycle",
            "predicted_rul",
        ],
    )

    metadata = get_dataframe_metadata(
        loaded_dataframe
    )

    write_json_atomic(
        data=metadata,
        path=json_path,
    )

    save_joblib_atomic(
        object_to_save={
            "message": "joblib test successful",
            "rows": len(loaded_dataframe),
        },
        path=joblib_path,
    )

    loaded_object = load_joblib(
        joblib_path
    )

    write_text_atomic(
        content="File utilities self-test completed successfully.",
        path=text_path,
    )

    LOGGER.info(
        f"Loaded joblib object: {loaded_object}"
    )

    LOGGER.info(
        f"CSV SHA-256: {calculate_sha256(csv_path)}"
    )

    LOGGER.info(
        "File utilities self-test completed successfully."
    )


if __name__ == "__main__":
    _run_self_test()