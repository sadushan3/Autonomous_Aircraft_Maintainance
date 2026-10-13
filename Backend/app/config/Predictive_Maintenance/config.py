from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Path anchoring
# --------------------------------------------------------------------------- #
# config.py lives at: Backend/app/config/Predictive_Maintenance/config.py
# The Backend root is therefore three parents up.
_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parents[3]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "Predictive Maintenance Backend"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    ARTIFACT_ROOT: str = str(BACKEND_ROOT / "models" / "Predictive_Maintenance")
    SUPPORTED_FDS: List[str] = Field(
        default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"]
    )

    FAILURE_RISK_WINDOW_SIZE: int = 30
    RUL_WINDOW_SIZES: Dict[str, int] = Field(
        default_factory=lambda: {
            "FD001": 50,
            "FD002": 70,
            "FD003": 50,
            "FD004": 50,
        }
    )

    FAILURE_RISK_MC_PASSES: int = 20
    DEFAULT_RUL_Q_HAT: Dict[str, float] = Field(
        default_factory=lambda: {
            "FD001": 20.0,
            "FD002": 25.0,
            "FD003": 20.0,
            "FD004": 28.0,
        }
    )
    # Deprecated. This was compared against ``2 * DEFAULT_RUL_Q_HAT`` to detect
    # confidence collapse, but every entry below is exactly twice the matching
    # q_hat, so the comparison was constant-vs-constant and never fired.
    # ``RULLifecycleService`` now tests the interval width against the estimate
    # it brackets instead and no longer reads this key. Retained so existing
    # .env overrides keep parsing; safe to delete once none remain.
    RUL_UNCERTAINTY_WIDTH_THRESHOLD: Dict[str, float] = Field(
        default_factory=lambda: {
            "FD001": 40.0,
            "FD002": 50.0,
            "FD003": 40.0,
            "FD004": 56.0,
        }
    )
    FAILURE_RISK_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "FD001": {"10": 0.50, "30": 0.50, "50": 0.50},
            "FD002": {"10": 0.50, "30": 0.50, "50": 0.50},
            "FD003": {"10": 0.50, "30": 0.50, "50": 0.50},
            "FD004": {"10": 0.50, "30": 0.50, "50": 0.50},
        }
    )

    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
