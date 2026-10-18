from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from app.schemas.Anomaly_Health_Monitoring.common_schema import (
    BaseResponse,
    StrictModel,
)


class SchemaDescription(StrictModel):


    identity_columns: List[str] = Field(..., description="['unit', 'cycle'].")
    op_setting_columns: List[str] = Field(..., description="The 3 operating settings.")
    sensor_columns: List[str] = Field(..., description="The 21 sensor channels.")
    raw_columns: List[str] = Field(..., description="Full 26-column layout in order.")
    n_raw_columns: int = Field(..., description="Always 26.")
    derived_label: str = Field(..., description="Derived label column name ('RUL').")
    valid_subsets: List[str] = Field(..., description="FD001..FD004.")
    subset_characteristics: Dict[str, Dict[str, int]] = Field(
        ..., description="Engine / regime / fault-mode counts per subset."
    )
    notes: str = Field(default="", description="Format caveats for callers.")


class UploadOptions(StrictModel):


    subset_id: Optional[str] = Field(
        default=None,
        description="Subset label to tag the rows with, e.g. 'FD001'. Inferred "
        "from the filename when omitted.",
    )
    split: str = Field(
        default="train",
        description="Which C-MAPSS split the file represents ('train' or 'test').",
    )
    compute_rul: bool = Field(
        default=True,
        description="Derive the RUL label as max cycle per engine minus cycle.",
    )

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:


        lowered = value.lower().strip()
        if lowered not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")
        return lowered


class UploadResponse(BaseResponse):


    filename: str = Field(..., description="Original filename as uploaded.")
    stored_path: str = Field(..., description="Where the file was persisted.")
    detected_format: str = Field(
        ..., description="'cmapss_txt' or 'csv', as auto-detected."
    )
    rows: int = Field(default=0, ge=0, description="Rows parsed.")
    columns: int = Field(default=0, ge=0, description="Columns found in the file.")
    engines: int = Field(default=0, ge=0, description="Distinct engine ids.")
    subsets: List[str] = Field(
        default_factory=list, description="Subset labels present in the data."
    )
    schema_valid: bool = Field(
        default=False, description="True when all 26 required columns were present."
    )
    missing_columns: List[str] = Field(
        default_factory=list,
        description="Required columns that were absent. Non-empty means rejection.",
    )
    ignored_columns: List[str] = Field(
        default_factory=list,
        description="Foreign columns dropped after validation, e.g. "
        "'rul_prediction' or 'failure_risk_score'. Never a failure cause.",
    )
    cycle_range: List[int] = Field(
        default_factory=list, description="[min_cycle, max_cycle] across the file."
    )


class PreprocessRequest(StrictModel):


    subsets: Optional[List[str]] = Field(
        default=None,
        description="Subsets to ingest, e.g. ['FD001'] or ['FD001','FD002',"
        "'FD003','FD004']. Defaults to the configured value.",
    )
    split: Optional[str] = Field(
        default=None, description="'train', 'test' or 'both'. Defaults to config."
    )
    source_csv: Optional[str] = Field(
        default=None,
        description="Explicit path to an already-parsed CSV. When supplied the "
        "raw .txt ingestion is bypassed.",
    )
    drop_constant_sensors: Optional[bool] = Field(
        default=None,
        description="Override whether flat sensors are dropped. Column "
        "alignment is preserved either way.",
    )
    max_engines: Optional[int] = Field(
        default=None,
        ge=0,
        description="Sub-sample this many engines for a fast smoke run. 0 = all.",
    )


class DataQualitySummary(StrictModel):


    rows_in: int = Field(default=0, ge=0)
    rows_out: int = Field(default=0, ge=0)
    duplicate_rows_removed: int = Field(default=0, ge=0)
    missing_values_imputed: int = Field(default=0, ge=0)
    non_finite_values_fixed: int = Field(default=0, ge=0)
    constant_sensors_detected: List[str] = Field(default_factory=list)
    constant_sensors_dropped: List[str] = Field(default_factory=list)
    active_sensors: List[str] = Field(
        default_factory=list, description="Sensors surviving into the models."
    )
    mean_data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)


class PreprocessResponse(BaseResponse):


    rows: int = Field(default=0, ge=0)
    engines: int = Field(default=0, ge=0)
    subsets: List[str] = Field(default_factory=list)
    raw_csv: str = Field(default="", description="Path of raw_data.csv.")
    cleaned_csv: str = Field(default="", description="Path of cleaned_data.csv.")
    features_csv: str = Field(
        default="", description="Path of engineered_features.csv."
    )
    feature_count: int = Field(default=0, ge=0)
    data_quality: Optional[DataQualitySummary] = Field(default=None)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ContextModelingResponse(BaseResponse):


    rows: int = Field(default=0, ge=0)
    algorithm: str = Field(default="", description="'kmeans' or 'gmm'.")
    n_modes: int = Field(default=0, ge=0, description="Operating modes discovered.")
    expected_modes: int = Field(
        default=0, ge=0, description="Modes expected for the loaded subsets."
    )
    silhouette_score: Optional[float] = Field(
        default=None, description="Cluster separation quality, when computed."
    )
    mode_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Row count per operating mode."
    )
    drift_detected: bool = Field(default=False)
    drift_psi: Optional[float] = Field(
        default=None, description="Population stability index of the mode mix."
    )
    context_csv: str = Field(default="", description="Path of context_clusters.csv.")
    scaled_csv: str = Field(default="", description="Path of scaled_features.csv.")


class EngineModeStatus(StrictModel):


    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)
    operating_mode: Optional[int] = Field(default=None)
    mode_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_mode_transition: Optional[bool] = Field(default=None)
    cycles_since_mode_change: Optional[int] = Field(default=None, ge=0)


class ContextDriftStatus(StrictModel):


    drift_detected: bool = Field(default=False)
    psi: float = Field(default=0.0, ge=0.0)
    threshold: float = Field(default=0.0, ge=0.0)
    severity: str = Field(default="none", description="'none', 'moderate' or 'major'.")
    reference_proportions: Dict[str, float] = Field(default_factory=dict)
    current_proportions: Dict[str, float] = Field(default_factory=dict)
    per_mode_shift: Dict[str, float] = Field(default_factory=dict)
    setting_drift: Dict[str, float] = Field(default_factory=dict)
    reason: str = Field(default="")


class ContextStatusResponse(BaseResponse):


    engines: List[EngineModeStatus] = Field(default_factory=list)
    drift: ContextDriftStatus = Field(default_factory=ContextDriftStatus)


class TwinTrainingRequest(StrictModel):


    twins: Optional[List[str]] = Field(
        default=None,
        description="Subset of ['random_forest','xgboost','lightgbm','tensorflow']. "
        "Defaults to all enabled twins.",
    )


class TwinMetrics(StrictModel):


    name: str = Field(..., description="Twin variant name.")
    rmse: float = Field(default=0.0, ge=0.0)
    mae: float = Field(default=0.0, ge=0.0)
    r2: float = Field(default=0.0)
    train_rows: int = Field(default=0, ge=0)
    val_rows: int = Field(default=0, ge=0)
    fit_seconds: float = Field(default=0.0, ge=0.0)
    ensemble_weight: float = Field(
        default=0.0, ge=0.0, description="Normalised inverse-RMSE fusion weight."
    )


class TwinTrainingResponse(BaseResponse):


    twins: List[TwinMetrics] = Field(default_factory=list)
    ensemble_rmse: float = Field(default=0.0, ge=0.0)
    healthy_rows_used: int = Field(
        default=0, ge=0, description="Rows in the healthy training region."
    )
    total_rows: int = Field(default=0, ge=0)
    predictions_csv: str = Field(default="")
    residuals_csv: str = Field(default="")
    residual_validation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output of validate_residuals: are healthy residuals centred?",
    )
