"""Path-scoped exception handlers for the Predictive Maintenance component.

The upstream standalone service called
``app.utils.Predictive_Maintenance.errors.register_exception_handlers(app)``
in its own ``create_app()``. That function registers two handlers:

* ``PredictiveMaintenanceError`` -- keyed on an exception type only this
  component raises, so it is inherently scoped and safe to register globally.
* ``RequestValidationError`` -- **not** scoped. It remaps FastAPI's default
  422 to a 400 with a different body shape, for *every* route in the app.

Registering the second one verbatim on the shared Backend app would change the
error contract of the Anomaly & Health Monitoring and Maintenance Supervisor
endpoints, which are out of scope for this integration and whose frontends may
depend on the 422. So the validation handler here dispatches on request path:
requests under the Predictive Maintenance mount prefix get the upstream 400
response byte-for-byte, and everything else gets FastAPI's stock 422 body,
exactly as before this module existed.

Without this, the four upstream tests asserting ``status_code == 400`` on an
invalid ``fd_subset`` or a too-short sequence fail against the mounted
component -- both are rejected by pydantic before the service layer runs, so
they surface as 422.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.utils.Predictive_Maintenance.errors import PredictiveMaintenanceError

#: Mount prefix of the Predictive Maintenance router, as wired in routes.py.
PREDICTIVE_MAINTENANCE_PREFIX = "/predictive-maintenance"


def register_predictive_maintenance_handlers(
    app: FastAPI, prefix: str = PREDICTIVE_MAINTENANCE_PREFIX
) -> None:
    """Install the component's error contract without touching sibling routes.

    Args:
        app: The shared FastAPI application.
        prefix: Mount prefix whose requests receive the upstream 400 contract.
    """

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
        if request.url.path.startswith(prefix):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": "Request validation failed.",
                    "details": jsonable_encoder(exc.errors()),
                },
            )
        # Stock FastAPI behaviour for every other component.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(exc.errors())},
        )
