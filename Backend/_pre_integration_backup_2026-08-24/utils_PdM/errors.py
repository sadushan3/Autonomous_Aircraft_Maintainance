from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


class PredictiveMaintenanceError(Exception):
    """Base application exception."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "predictive_maintenance_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ArtifactNotFoundError(PredictiveMaintenanceError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "artifact_not_found"


class InvalidFDSubsetError(PredictiveMaintenanceError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_fd_subset"


class InvalidSequenceError(PredictiveMaintenanceError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_sequence"


class PredictionError(PredictiveMaintenanceError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "prediction_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PredictiveMaintenanceError)
    async def predictive_error_handler(
        request: Request, exc: PredictiveMaintenanceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": "Request validation failed.",
                "details": jsonable_encoder(exc.errors()),
            },
        )
