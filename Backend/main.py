"""
FastAPI application entry point.

Wires the Anomaly & Health Monitoring, Predictive Maintenance (RUL /
Failure-Risk), Maintenance Supervisor, and Integration routers into one
FastAPI app, each under its own prefix. The Integration router is the live,
callable end-to-end path: `POST /integration/run-full` runs all three
upstream components against an uploaded file and fuses the result through
the Maintenance Supervisor in one request -- no CSV hand-picking, no
synthetic fallback.

Run with::

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.Anomaly_Health_Monitoring.config import get_config
from app.routers.Anomaly_Health_Monitoring.routes import router as anomaly_health_router
from app.routers.Predictive_Maintenance.routes import router as predictive_maintenance_router
from app.routers.Maintenance_Supervisor.routes import router as maintenance_supervisor_router
from app.routers.Integration.routes import router as integration_router
from app.routers.Predictive_Maintenance.exception_handlers import (
    register_predictive_maintenance_handlers,
)
from app.utils.Anomaly_Health_Monitoring.utils import configure_logging, get_logger

config = get_config()
configure_logging(config)
logger = get_logger(__name__, config)

app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    description=(
        "Explainable Health Monitoring & Anomaly Reasoning Agent -- "
        "Digital-Twin-based predictive maintenance over the NASA C-MAPSS "
        "turbofan degradation dataset."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anomaly_health_router)
app.include_router(predictive_maintenance_router)
app.include_router(maintenance_supervisor_router)
app.include_router(integration_router)

# Restores the upstream Predictive Maintenance error contract (400 + error/message
# body) for that component's routes only. Anomaly & Health and Maintenance
# Supervisor keep FastAPI's stock 422 -- see exception_handlers.py.
register_predictive_maintenance_handlers(app)


def _warm_up_tensorflow() -> None:
    """Import and touch TensorFlow once, on the main thread, before any request lands.

    Both the LSTM autoencoder (Anomaly & Health) and the RUL / Failure-Risk
    Keras models import ``tensorflow`` lazily, on first use, inside whichever
    Starlette threadpool worker happens to be serving that request. Importing
    it once here, on the main thread at startup, moves that one-off cost
    (several seconds, mostly the native extension and Keras submodules) out
    of the first request that needs it.

    This is a latency convenience, nothing more. It is deliberately *not*
    doing any thread-pool pinning: an earlier version of this function set
    TensorFlow's intra/inter-op parallelism to 1 to guard against a suspected
    multi-threaded initialisation deadlock, but that diagnosis was wrong --
    the apparent hang was the uvicorn *reloader parent* sitting idle at 0%
    CPU by design while the real worker child churned through a genuinely
    long batch. Pinning TF to a single thread only slowed that batch down.

    Failure to import TensorFlow does not fail startup: it is logged and the
    app continues, since not every deployment exercises the Predictive
    Maintenance / LSTM-autoencoder code paths.
    """
    try:
        import tensorflow as tf

        # Touch the Keras API surface so its lazy submodules are loaded now,
        # rather than on whichever thread handles the first request needing them.
        _ = tf.keras.models

        logger.info("TensorFlow %s warmed up on startup.", tf.__version__)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "TensorFlow warm-up skipped (%s). LSTM-autoencoder / RUL / "
            "Failure-Risk endpoints may be slower or less stable on first "
            "use if TensorFlow is required but was not pre-initialised.",
            exc,
        )


@app.on_event("startup")
def on_startup() -> None:
    """Ensure every runtime directory exists before the first request lands."""
    config.ensure_directories()
    _warm_up_tensorflow()
    logger.info(
        "%s v%s starting up. Data raw dir: %s",
        config.api.title,
        config.api.version,
        config.paths.data_raw,
    )


@app.get("/", tags=["Health"])
def root() -> dict:
    """Basic liveness endpoint.

    Returns:
        A small status payload.
    """
    return {
        "service": config.api.title,
        "version": config.api.version,
        "status": "ok",
    }


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Kubernetes/uptime-style health check.

    Returns:
        A minimal OK payload.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
