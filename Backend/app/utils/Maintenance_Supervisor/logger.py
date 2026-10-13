"""
logger.py

Production logging utility for the Autonomous Maintenance Supervisor Agent.

Features
--------
✓ Console logging
✓ Rotating log files
✓ Timestamped messages
✓ Execution timers
✓ Exception logging
✓ Pipeline stage logging
✓ Thread-safe singleton logger
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from functools import wraps
from pathlib import Path
from typing import Callable

from app.config.Maintenance_Supervisor.supervisor_config import LOGS_ROOT


# ==============================================================================
# Logger Configuration
# ==============================================================================

LOGGER_NAME = "MaintenanceSupervisor"

LOG_FILE = LOGS_ROOT / "supervisor.log"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==============================================================================
# Internal Logger Instance
# ==============================================================================

_logger: logging.Logger | None = None


# ==============================================================================
# Build Logger
# ==============================================================================

def get_logger() -> logging.Logger:
    """
    Returns a singleton logger.

    The logger writes simultaneously to

    • Console
    • Rotating log file

    Returns
    -------
    logging.Logger
    """

    global _logger

    if _logger is not None:
        return _logger

    LOGS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(logging.INFO)

    logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # File
    # ------------------------------------------------------------------

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    _logger = logger

    return logger


# ==============================================================================
# Convenience Functions
# ==============================================================================

def info(message: str) -> None:
    get_logger().info(message)


def warning(message: str) -> None:
    get_logger().warning(message)


def error(message: str) -> None:
    get_logger().error(message)


def critical(message: str) -> None:
    get_logger().critical(message)


def debug(message: str) -> None:
    get_logger().debug(message)


# ==============================================================================
# Pipeline Section Logger
# ==============================================================================

def section(title: str) -> None:
    """
    Prints a pipeline section.

    Example

    ===========================================================
    FEATURE ENGINEERING
    ===========================================================
    """

    line = "=" * 70

    logger = get_logger()

    logger.info("")

    logger.info(line)

    logger.info(title.upper())

    logger.info(line)


# ==============================================================================
# Timing Decorator
# ==============================================================================

def log_execution_time(func: Callable):
    """
    Decorator for execution timing.

    Example

    @log_execution_time
    def train_model():
        ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        logger = get_logger()

        start = time.perf_counter()

        logger.info(
            f"Started : {func.__name__}"
        )

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            f"Finished: {func.__name__} "
            f"({elapsed:.2f} seconds)"
        )

        return result

    return wrapper


# ==============================================================================
# Exception Logger
# ==============================================================================

def log_exception(
    exception: Exception,
    location: str,
) -> None:
    """
    Logs exceptions with traceback.
    """

    logger = get_logger()

    logger.exception(
        f"Exception at {location}: {exception}"
    )


# ==============================================================================
# Experiment Logger
# ==============================================================================

def experiment(
    experiment_name: str,
) -> None:

    logger = get_logger()

    logger.info(
        ""
    )

    logger.info(
        "#" * 70
    )

    logger.info(
        f"EXPERIMENT : {experiment_name}"
    )

    logger.info(
        "#" * 70
    )


# ==============================================================================
# Metrics Logger
# ==============================================================================

def metric(
    name: str,
    value,
) -> None:
    """
    Example

    Accuracy : 0.9542
    """

    get_logger().info(
        f"{name:<35}: {value}"
    )


# ==============================================================================
# Dictionary Logger
# ==============================================================================

def dictionary(
    title: str,
    values: dict,
) -> None:

    logger = get_logger()

    logger.info(title)

    for key, value in values.items():

        logger.info(
            f"    {key:<30}: {value}"
        )


# ==============================================================================
# Separator
# ==============================================================================

def separator() -> None:

    get_logger().info(
        "-" * 70
    )


# ==============================================================================
# Banner
# ==============================================================================

def banner() -> None:

    logger = get_logger()

    logger.info("")

    logger.info("=" * 70)

    logger.info(
        "AUTONOMOUS MAINTENANCE SUPERVISOR AGENT"
    )

    logger.info("=" * 70)

    logger.info("")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    banner()

    section("Logger Test")

    info("Logger initialized successfully.")

    warning("Example warning.")

    error("Example error.")

    metric("Accuracy", 0.9521)

    metric("Macro F1", 0.9445)

    metric("Critical Recall", 0.982)

    dictionary(
        "Model Parameters",
        {
            "n_estimators": 300,
            "max_depth": 20,
            "learning_rate": 0.05,
        },
    )

    separator()

    experiment("Random Forest Baseline")

    info("Experiment completed successfully.")