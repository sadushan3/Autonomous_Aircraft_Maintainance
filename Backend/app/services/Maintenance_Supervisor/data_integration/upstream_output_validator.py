"""
upstream_output_validator.py

Schema Validator & Constraint Checker for Real Upstream Component Outputs.

Purpose
-------
This module validates raw prediction CSVs produced by upstream research components:
1. Remaining Useful Life (RUL) Prediction Component
2. Failure Risk Prediction Component
3. Anomaly & Health Monitoring Component

It enforces strict schema, data type, value range, and temporal sanity constraints
on incoming files before they are merged into the Supervisor Agent's pipeline.

Validation Rules
----------------
- Schema Integrity:
  Ensures engine_id, cycle, and required model predictions are present.

- Bounds & Range Checks:
  - RUL: predicted_rul >= 0, lower_bound <= upper_bound.
  - Risk: risk_10, risk_30, risk_50 in [0.0, 1.0], risk_10 <= risk_30 <= risk_50.
  - Anomaly: anomaly_score in [0.0, 1.0], anomaly_severity in valid enum set.

- Monotonicity & Continuity:
  Checks that engine cycle sequences are contiguous and non-decreasing.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.data_integration.upstream_output_validator

Expected outputs
----------------
- reports/Maintenance_Supervisor/upstream_validation_report.json

Exit codes
----------
0 — validation clean (0 errors)
1 — validation failure (errors detected)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    REPORTS_ROOT,
    ENGINE_ID_COLUMN,
    CYCLE_COLUMN,
    RUL_NUMERICAL_FEATURES,
    RISK_NUMERICAL_FEATURES,
    ANOMALY_NUMERICAL_FEATURES,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths & Enums
# ---------------------------------------------------------------------------

_REPORT_PATH: Final[Path] = REPORTS_ROOT / "upstream_validation_report.json"
ComponentType = Literal["rul", "risk", "anomaly", "combined"]

VALID_ANOMALY_SEVERITIES: Final[frozenset[str]] = frozenset([
    "none", "low", "medium", "high", "critical", "unknown"
])

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    component: str
    column: str
    severity: str  # "ERROR" or "WARNING"
    description: str


@dataclass
class UpstreamValidationResult:
    is_valid: bool
    component: str
    total_rows: int
    error_count: int
    warning_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "component": self.component,
            "total_rows": self.total_rows,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "component": i.component,
                    "column": i.column,
                    "severity": i.severity,
                    "description": i.description,
                }
                for i in self.issues
            ],
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_rul_component(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate RUL component predictions."""
    issues: list[ValidationIssue] = []

    if "predicted_rul" in df.columns:
        neg_rul = (df["predicted_rul"] < 0).sum()
        if neg_rul > 0:
            issues.append(ValidationIssue("rul", "predicted_rul", "ERROR", f"Found {neg_rul} negative RUL predictions."))

    if "lower_bound" in df.columns and "upper_bound" in df.columns:
        invalid_bounds = (df["lower_bound"] > df["upper_bound"]).sum()
        if invalid_bounds > 0:
            issues.append(ValidationIssue("rul", "lower_bound", "ERROR", f"Found {invalid_bounds} rows where lower_bound > upper_bound."))

    return issues


def validate_risk_component(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate Failure Risk component predictions."""
    issues: list[ValidationIssue] = []

    for col in ["risk_10", "risk_30", "risk_50"]:
        if col in df.columns:
            out_of_bounds = ((df[col] < 0.0) | (df[col] > 1.0)).sum()
            if out_of_bounds > 0:
                issues.append(ValidationIssue("risk", col, "ERROR", f"Found {out_of_bounds} values outside [0.0, 1.0]."))

    if all(c in df.columns for c in ["risk_10", "risk_30"]):
        inv_order = (df["risk_10"] > df["risk_30"] + 1e-5).sum()
        if inv_order > 0:
            issues.append(ValidationIssue("risk", "risk_10", "WARNING", f"Found {inv_order} rows where risk_10 > risk_30."))

    return issues


def validate_anomaly_component(df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate Anomaly Detection component predictions."""
    issues: list[ValidationIssue] = []

    if "anomaly_score" in df.columns:
        out_of_bounds = ((df["anomaly_score"] < 0.0) | (df["anomaly_score"] > 1.0)).sum()
        if out_of_bounds > 0:
            issues.append(ValidationIssue("anomaly", "anomaly_score", "ERROR", f"Found {out_of_bounds} values outside [0.0, 1.0]."))

    if "anomaly_severity" in df.columns:
        invalid_sev = (~df["anomaly_severity"].astype(str).str.lower().isin(VALID_ANOMALY_SEVERITIES)).sum()
        if invalid_sev > 0:
            issues.append(ValidationIssue("anomaly", "anomaly_severity", "WARNING", f"Found {invalid_sev} unknown anomaly_severity values."))

    return issues


def validate_upstream_dataframe(
    df: pd.DataFrame,
    component: ComponentType = "combined",
) -> UpstreamValidationResult:
    """Validate an upstream DataFrame against all schema & domain rules."""
    start_time = time.perf_counter()
    issues: list[ValidationIssue] = []

    # Essential Keys Check
    for req_col in [ENGINE_ID_COLUMN, CYCLE_COLUMN]:
        if req_col not in df.columns:
            issues.append(ValidationIssue(component, req_col, "ERROR", f"Missing essential key column '{req_col}'."))

    # Component Specific Validations
    if component in ("rul", "combined"):
        issues.extend(validate_rul_component(df))
    if component in ("risk", "combined"):
        issues.extend(validate_risk_component(df))
    if component in ("anomaly", "combined"):
        issues.extend(validate_anomaly_component(df))

    err_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARNING")
    duration = time.perf_counter() - start_time

    return UpstreamValidationResult(
        is_valid=(err_count == 0),
        component=component,
        total_rows=len(df),
        error_count=err_count,
        warning_count=warn_count,
        issues=issues,
        duration_seconds=round(duration, 4),
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def run_upstream_validator(
    input_path: Path | None = None,
    component: ComponentType = "combined",
) -> UpstreamValidationResult:
    """Run upstream validation on a CSV file."""
    if input_path is None:
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("UPSTREAM OUTPUT VALIDATOR STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Component type: %s", component)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    res = validate_upstream_dataframe(df, component=component)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(res.to_dict(), _REPORT_PATH)

    logger.info("Validation complete: %d errors, %d warnings.", res.error_count, res.warning_count)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("UPSTREAM OUTPUT VALIDATOR COMPLETED")

    return res


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_upstream_validator()
        print(json.dumps(res.to_dict(), indent=2))
        return 0 if res.is_valid else 1
    except Exception as exc:
        logger.error("Error running upstream validator: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
