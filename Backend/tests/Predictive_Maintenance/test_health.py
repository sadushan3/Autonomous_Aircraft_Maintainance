"""Upstream test, vendored from predictive-maintenance-backend/tests/test_health.py.

Preserved as shipped apart from two mechanical adaptations, both forced by the
component being mounted into the shared Backend app rather than running as a
standalone service:

  * ``from app.main import app`` -> ``from main import app`` (this repo's app).
  * URL prefix ``/api`` -> ``/predictive-maintenance`` (the mount prefix).

No assertion, fixture or test name has been altered. Requires pytest, which is
not currently installed in this environment:  pip install pytest
"""

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/predictive-maintenance/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "predictive-maintenance-backend"


def test_artifact_status_works_without_artifacts() -> None:
    response = client.get("/predictive-maintenance/artifacts/status")

    assert response.status_code == 200
    body = response.json()
    assert "rul_lifecycle" in body
    assert "failure_risk" in body
    assert len(body["rul_lifecycle"]) == 4

