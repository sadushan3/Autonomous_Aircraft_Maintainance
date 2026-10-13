"""Upstream test, vendored from predictive-maintenance-backend/tests/test_routes.py.

Preserved as shipped apart from two mechanical adaptations, both forced by the
component being mounted into the shared Backend app rather than running as a
standalone service:

  * ``from app.main import app`` -> ``from main import app`` (this repo's app).
  * URL prefix ``/api`` -> ``/predictive-maintenance`` (the mount prefix).

No assertion, fixture or test name has been altered. Requires pytest, which is
not currently installed in this environment:  pip install pytest
"""

from main import app


def test_required_routes_are_registered() -> None:
    registered = {route.path for route in app.routes}
    required = {
        "/predictive-maintenance/health",
        "/predictive-maintenance/artifacts/status",
        "/predictive-maintenance/artifacts/config",
        "/predictive-maintenance/rul/predict",
        "/predictive-maintenance/failure-risk/predict",
        "/predictive-maintenance/predict/combined",
        "/predictive-maintenance/demo/sample-request",
        "/predictive-maintenance/demo/predict",
    }

    assert required.issubset(registered)

