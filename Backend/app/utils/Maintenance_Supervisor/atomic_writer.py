"""
atomic_writer.py

Transactional and atomic file-writing utilities for the Autonomous
Maintenance Supervisor Agent.

Purpose
-------
A machine-learning pipeline may generate several related outputs, such as:

- predictions.csv
- metrics.json
- feature_columns.json
- trained_model.joblib

If one write fails midway, the pipeline should not leave a mixture of old and
new files. This module provides transaction-style writing:

1. Write every output to a temporary file.
2. Validate the temporary outputs.
3. Back up existing destination files.
4. Atomically replace all destination files.
5. Roll back if any commit operation fails.

Supported formats
-----------------
- CSV
- JSON
- text
- bytes
- Joblib
- Parquet

The module is compatible with Windows, Linux, and macOS.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
# CUSTOM EXCEPTIONS
# =============================================================================

class AtomicWriteError(RuntimeError):
    """
    Raised when an atomic file-writing operation fails.
    """


class TransactionStateError(RuntimeError):
    """
    Raised when an invalid transaction operation is attempted.
    """


# =============================================================================
# TRANSACTION ITEM
# =============================================================================

@dataclass
class TransactionItem:
    """
    Represents one staged file in an atomic transaction.

    Attributes
    ----------
    target_path:
        Final destination path.
    temporary_path:
        Temporary file created during staging.
    backup_path:
        Backup of an existing target file, if one existed.
    committed:
        Whether the temporary file has replaced the destination.
    """

    target_path: Path
    temporary_path: Path
    backup_path: Path | None = None
    committed: bool = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _resolve_target(path: str | Path) -> Path:
    """
    Resolve a destination path and create its parent directory.
    """

    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    return target_path


def _create_temp_path(
    target_path: Path,
    suffix: str | None = None,
) -> Path:
    """
    Create a temporary file in the same directory as the destination.

    Using the same directory helps ensure that os.replace is atomic because the
    temporary and destination files are located on the same filesystem.
    """

    selected_suffix = suffix or target_path.suffix or ".tmp"

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}_staged_",
        suffix=selected_suffix,
        dir=target_path.parent,
    )

    os.close(file_descriptor)

    return Path(temporary_name)


def _validate_staged_file(
    path: Path,
    allow_empty: bool = False,
) -> None:
    """
    Validate that a staged file exists and is suitable for committing.
    """

    if not path.exists():
        raise AtomicWriteError(
            f"Staged file was not created: {path}"
        )

    if not path.is_file():
        raise AtomicWriteError(
            f"Staged path is not a file: {path}"
        )

    if not allow_empty and path.stat().st_size == 0:
        raise AtomicWriteError(
            f"Staged file is empty: {path}"
        )


def _safe_delete(path: Path | None) -> None:
    """
    Delete a file without raising an exception.
    """

    if path is None:
        return

    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError:
        LOGGER.warning(
            f"Could not remove temporary file: {path}"
        )


# =============================================================================
# ATOMIC TRANSACTION
# =============================================================================

class AtomicWriteTransaction:
    """
    Manage several file writes as one transaction.

    Example
    -------
    with AtomicWriteTransaction() as transaction:
        transaction.stage_csv(dataframe, "predictions.csv")
        transaction.stage_json(metrics, "metrics.json")

    The context manager automatically commits when no exception occurs.
    If staging or committing fails, previous destination files are restored.
    """

    def __init__(
        self,
        transaction_name: str | None = None,
        keep_backups: bool = False,
    ) -> None:
        """
        Initialize an atomic write transaction.

        Parameters
        ----------
        transaction_name:
            Optional human-readable transaction name.
        keep_backups:
            Keep destination backups after a successful commit.
            Normally False because backups are only needed for rollback.
        """

        self.transaction_id = uuid.uuid4().hex
        self.transaction_name = (
            transaction_name
            or f"transaction_{self.transaction_id[:8]}"
        )
        self.keep_backups = keep_backups

        self._items: list[TransactionItem] = []
        self._committed = False
        self._rolled_back = False
        self._closed = False

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    def __enter__(self) -> "AtomicWriteTransaction":
        LOGGER.info(
            f"Atomic transaction started: {self.transaction_name}"
        )
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> bool:
        """
        Commit when the context exits normally.

        Roll back when an exception occurs.
        """

        if exception_type is not None:
            LOGGER.error(
                f"Atomic transaction failed before commit: "
                f"{self.transaction_name}"
            )
            self.rollback()
            self.close()
            return False

        try:
            self.commit()
        except Exception:
            self.rollback()
            self.close()
            raise

        self.close()
        return False

    # =========================================================================
    # TRANSACTION STATE
    # =========================================================================

    def _ensure_open(self) -> None:
        """
        Ensure that the transaction can still accept staged files.
        """

        if self._closed:
            raise TransactionStateError(
                "The transaction is already closed."
            )

        if self._committed:
            raise TransactionStateError(
                "The transaction has already been committed."
            )

        if self._rolled_back:
            raise TransactionStateError(
                "The transaction has already been rolled back."
            )

    @property
    def staged_count(self) -> int:
        """
        Number of staged files.
        """

        return len(self._items)

    @property
    def committed(self) -> bool:
        """
        Whether the transaction has committed successfully.
        """

        return self._committed

    # =========================================================================
    # GENERIC STAGING
    # =========================================================================

    def stage_custom(
        self,
        target_path: str | Path,
        writer: Callable[[Path], None],
        suffix: str | None = None,
        allow_empty: bool = False,
    ) -> Path:
        """
        Stage a file using a custom writer function.

        Parameters
        ----------
        target_path:
            Final destination path.
        writer:
            Callable that receives a temporary Path and writes content to it.
        suffix:
            Optional temporary-file extension.
        allow_empty:
            Permit an empty output file.

        Returns
        -------
        Path
            Final destination path that will be used after commit.
        """

        self._ensure_open()

        destination = _resolve_target(target_path)

        if any(
            item.target_path == destination
            for item in self._items
        ):
            raise AtomicWriteError(
                f"Destination is already staged in this transaction: "
                f"{destination}"
            )

        temporary_path = _create_temp_path(
            destination,
            suffix=suffix,
        )

        LOGGER.info(
            f"Staging output | target={destination}"
        )

        try:
            writer(temporary_path)

            _validate_staged_file(
                temporary_path,
                allow_empty=allow_empty,
            )

            self._items.append(
                TransactionItem(
                    target_path=destination,
                    temporary_path=temporary_path,
                )
            )

        except Exception as exception:
            _safe_delete(temporary_path)

            LOGGER.exception(
                f"Failed to stage output: {destination}"
            )

            raise AtomicWriteError(
                f"Could not stage output: {destination}"
            ) from exception

        return destination

    # =========================================================================
    # CSV
    # =========================================================================

    def stage_csv(
        self,
        dataframe: pd.DataFrame,
        target_path: str | Path,
        index: bool = False,
        encoding: str = "utf-8",
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Stage a pandas DataFrame as a CSV file.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty and not allow_empty:
            raise ValueError(
                "Cannot stage an empty dataframe unless "
                "allow_empty=True."
            )

        return self.stage_custom(
            target_path=target_path,
            suffix=".csv",
            allow_empty=allow_empty,
            writer=lambda temporary_path: dataframe.to_csv(
                temporary_path,
                index=index,
                encoding=encoding,
                **kwargs,
            ),
        )

    # =========================================================================
    # JSON
    # =========================================================================

    def stage_json(
        self,
        data: Any,
        target_path: str | Path,
        indent: int = 4,
        sort_keys: bool = False,
    ) -> Path:
        """
        Stage JSON-compatible data.
        """

        def write_json(temporary_path: Path) -> None:
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
                    default=self._json_serializer,
                )

        return self.stage_custom(
            target_path=target_path,
            suffix=".json",
            writer=write_json,
        )

    @staticmethod
    def _json_serializer(value: Any) -> Any:
        """
        Serialize common objects unsupported by standard JSON.
        """

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, set):
            return sorted(value)

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )

    # =========================================================================
    # TEXT
    # =========================================================================

    def stage_text(
        self,
        content: str,
        target_path: str | Path,
        encoding: str = "utf-8",
        allow_empty: bool = False,
    ) -> Path:
        """
        Stage a text file.
        """

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        def write_text(temporary_path: Path) -> None:
            with temporary_path.open(
                mode="w",
                encoding=encoding,
            ) as file:
                file.write(content)

        return self.stage_custom(
            target_path=target_path,
            suffix=Path(target_path).suffix or ".txt",
            writer=write_text,
            allow_empty=allow_empty,
        )

    # =========================================================================
    # BINARY DATA
    # =========================================================================

    def stage_bytes(
        self,
        content: bytes,
        target_path: str | Path,
        allow_empty: bool = False,
    ) -> Path:
        """
        Stage raw binary data.
        """

        if not isinstance(content, bytes):
            raise TypeError(
                "content must be bytes."
            )

        def write_bytes(temporary_path: Path) -> None:
            with temporary_path.open("wb") as file:
                file.write(content)

        return self.stage_custom(
            target_path=target_path,
            suffix=Path(target_path).suffix or ".bin",
            writer=write_bytes,
            allow_empty=allow_empty,
        )

    # =========================================================================
    # JOBLIB
    # =========================================================================

    def stage_joblib(
        self,
        object_to_save: Any,
        target_path: str | Path,
        compress: int = 3,
    ) -> Path:
        """
        Stage a Python object using joblib.
        """

        return self.stage_custom(
            target_path=target_path,
            suffix=Path(target_path).suffix or ".joblib",
            writer=lambda temporary_path: joblib.dump(
                object_to_save,
                temporary_path,
                compress=compress,
            ),
        )

    # =========================================================================
    # PARQUET
    # =========================================================================

    def stage_parquet(
        self,
        dataframe: pd.DataFrame,
        target_path: str | Path,
        index: bool = False,
        compression: str = "snappy",
        allow_empty: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        Stage a pandas DataFrame as Parquet.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty and not allow_empty:
            raise ValueError(
                "Cannot stage an empty dataframe unless "
                "allow_empty=True."
            )

        def write_parquet(temporary_path: Path) -> None:
            try:
                dataframe.to_parquet(
                    temporary_path,
                    index=index,
                    compression=compression,
                    **kwargs,
                )
            except ImportError as exception:
                raise ImportError(
                    "Parquet output requires pyarrow or fastparquet. "
                    "Install pyarrow using: pip install pyarrow"
                ) from exception

        return self.stage_custom(
            target_path=target_path,
            suffix=".parquet",
            writer=write_parquet,
            allow_empty=allow_empty,
        )

    # =========================================================================
    # COMMIT
    # =========================================================================

    def commit(self) -> None:
        """
        Commit all staged files.

        Existing destination files are temporarily backed up. If any replacement
        fails, all already-replaced files are restored.
        """

        self._ensure_open()

        if not self._items:
            raise TransactionStateError(
                "Cannot commit a transaction with no staged files."
            )

        LOGGER.info(
            f"Committing atomic transaction: {self.transaction_name} | "
            f"files={len(self._items)}"
        )

        try:
            self._create_destination_backups()

            for item in self._items:
                os.replace(
                    item.temporary_path,
                    item.target_path,
                )

                item.committed = True

                LOGGER.info(
                    f"Committed output: {item.target_path}"
                )

        except Exception as exception:
            LOGGER.exception(
                f"Commit failed: {self.transaction_name}"
            )

            self._restore_after_failed_commit()

            raise AtomicWriteError(
                f"Atomic transaction commit failed: "
                f"{self.transaction_name}"
            ) from exception

        self._committed = True

        if not self.keep_backups:
            self._delete_backups()

        LOGGER.info(
            f"Atomic transaction committed successfully: "
            f"{self.transaction_name}"
        )

    def _create_destination_backups(self) -> None:
        """
        Create temporary backups for existing destination files.
        """

        for item in self._items:
            if not item.target_path.exists():
                continue

            backup_path = item.target_path.with_name(
                f".{item.target_path.name}."
                f"{self.transaction_id}.backup"
            )

            shutil.copy2(
                item.target_path,
                backup_path,
            )

            item.backup_path = backup_path

            LOGGER.debug(
                f"Created rollback backup: {backup_path}"
            )

    def _restore_after_failed_commit(self) -> None:
        """
        Restore destination files after an incomplete commit.
        """

        for item in reversed(self._items):
            try:
                if item.committed:
                    if item.backup_path and item.backup_path.exists():
                        os.replace(
                            item.backup_path,
                            item.target_path,
                        )
                    else:
                        _safe_delete(item.target_path)

                elif item.backup_path and item.backup_path.exists():
                    _safe_delete(item.backup_path)

            except Exception:
                LOGGER.exception(
                    f"Rollback restoration failed for: "
                    f"{item.target_path}"
                )

        for item in self._items:
            _safe_delete(item.temporary_path)

    # =========================================================================
    # ROLLBACK
    # =========================================================================

    def rollback(self) -> None:
        """
        Cancel the transaction and remove staged files.

        If a partial commit occurred, destination backups are restored.
        """

        if self._rolled_back:
            return

        if self._committed:
            raise TransactionStateError(
                "A successfully committed transaction cannot be rolled back."
            )

        LOGGER.warning(
            f"Rolling back transaction: {self.transaction_name}"
        )

        self._restore_after_failed_commit()

        self._rolled_back = True

        LOGGER.info(
            f"Transaction rolled back: {self.transaction_name}"
        )

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def _delete_backups(self) -> None:
        """
        Delete rollback backups after a successful commit.
        """

        for item in self._items:
            _safe_delete(item.backup_path)
            item.backup_path = None

    def close(self) -> None:
        """
        Clean up temporary transaction files.
        """

        if self._closed:
            return

        for item in self._items:
            _safe_delete(item.temporary_path)

            if not self.keep_backups:
                _safe_delete(item.backup_path)

        self._closed = True


# =============================================================================
# SINGLE-FILE CONVENIENCE FUNCTIONS
# =============================================================================

def atomic_write_csv(
    dataframe: pd.DataFrame,
    target_path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Atomically write one CSV file.
    """

    with AtomicWriteTransaction(
        transaction_name="single_csv_write"
    ) as transaction:
        result = transaction.stage_csv(
            dataframe=dataframe,
            target_path=target_path,
            index=index,
            **kwargs,
        )

    return result


def atomic_write_json(
    data: Any,
    target_path: str | Path,
    indent: int = 4,
) -> Path:
    """
    Atomically write one JSON file.
    """

    with AtomicWriteTransaction(
        transaction_name="single_json_write"
    ) as transaction:
        result = transaction.stage_json(
            data=data,
            target_path=target_path,
            indent=indent,
        )

    return result


def atomic_write_text(
    content: str,
    target_path: str | Path,
) -> Path:
    """
    Atomically write one text file.
    """

    with AtomicWriteTransaction(
        transaction_name="single_text_write"
    ) as transaction:
        result = transaction.stage_text(
            content=content,
            target_path=target_path,
        )

    return result


def atomic_save_joblib(
    object_to_save: Any,
    target_path: str | Path,
    compress: int = 3,
) -> Path:
    """
    Atomically save one Joblib artifact.
    """

    with AtomicWriteTransaction(
        transaction_name="single_joblib_write"
    ) as transaction:
        result = transaction.stage_joblib(
            object_to_save=object_to_save,
            target_path=target_path,
            compress=compress,
        )

    return result


# =============================================================================
# TRANSACTION MANIFEST
# =============================================================================

def build_transaction_manifest(
    transaction: AtomicWriteTransaction,
) -> dict[str, Any]:
    """
    Create metadata describing a transaction.
    """

    return {
        "transaction_id": transaction.transaction_id,
        "transaction_name": transaction.transaction_name,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "staged_count": transaction.staged_count,
        "committed": transaction.committed,
        "files": [
            {
                "target_path": str(item.target_path),
                "temporary_path": str(item.temporary_path),
                "backup_path": (
                    str(item.backup_path)
                    if item.backup_path
                    else None
                ),
                "committed": item.committed,
            }
            for item in transaction._items
        ],
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test() -> None:
    """
    Run a small transaction test.

    This creates three related files in a test directory and commits all of
    them together.
    """

    LOGGER.info("=" * 72)
    LOGGER.info("ATOMIC WRITER SELF-TEST")
    LOGGER.info("=" * 72)

    test_directory = (
        Path(__file__).resolve().parent
        / "_atomic_writer_test"
    )

    test_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.DataFrame(
        {
            "engine_id": [
                "ENG_001",
                "ENG_002",
                "ENG_003",
            ],
            "supervisor_decision": [
                "CONTINUE_OPERATION",
                "SCHEDULE_MAINTENANCE",
                "IMMEDIATE_SHUTDOWN",
            ],
            "confidence_score": [
                0.96,
                0.84,
                0.93,
            ],
        }
    )

    metrics = {
        "accuracy": 0.94,
        "macro_f1": 0.92,
        "critical_recall": 0.98,
    }

    report = (
        "Atomic writer self-test completed successfully.\n"
        "All related files were written in one transaction.\n"
    )

    with AtomicWriteTransaction(
        transaction_name="atomic_writer_self_test"
    ) as transaction:
        transaction.stage_csv(
            dataframe=predictions,
            target_path=test_directory / "predictions.csv",
        )

        transaction.stage_json(
            data=metrics,
            target_path=test_directory / "metrics.json",
        )

        transaction.stage_text(
            content=report,
            target_path=test_directory / "report.txt",
        )

    expected_files = [
        test_directory / "predictions.csv",
        test_directory / "metrics.json",
        test_directory / "report.txt",
    ]

    missing_files = [
        str(path)
        for path in expected_files
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            f"Atomic writer self-test failed. "
            f"Missing files: {missing_files}"
        )

    LOGGER.info(
        "Atomic writer self-test completed successfully."
    )

    LOGGER.info(
        f"Test outputs saved in: {test_directory}"
    )


if __name__ == "__main__":
    _run_self_test()