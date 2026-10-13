"""
routes.py

Router re-export shim for the Autonomous Maintenance Supervisor Agent.

The real FastAPI router (all endpoint definitions) lives in
`app.services.Maintenance_Supervisor.service.api_routes`, per this project's
convention of implementing route handlers alongside their service logic.
This module exists only so `main.py` can import a router from the standard
`app.routers.<Component>.routes` location, matching the pattern used by the
other components in this backend (e.g. `app.routers.Anomaly_Health_Monitoring.routes`).

The router already has `prefix="/api/v1/supervisor"` and tags baked in via
`APIRouter(prefix=..., tags=...)` in `api_routes.py`, so it is re-exported
as-is with no further modification.
"""

from __future__ import annotations

from app.services.Maintenance_Supervisor.service.api_routes import router

__all__ = ["router"]
