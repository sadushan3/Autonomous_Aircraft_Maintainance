"""Upstream test, vendored from predictive-maintenance-backend/tests/test_artifacts.py.

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


def test_artifact_status_has_expected_shape() -> None:
    response = client.get("/predictive-maintenance/artifacts/status")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"rul_lifecycle", "failure_risk"}
    assert len(body["rul_lifecycle"]) == 4
    assert len(body["failure_risk"]) == 4
    first = body["rul_lifecycle"][0]
    assert "missing_model" in first
    assert "searched_locations" in first


def test_artifact_config_returns_supported_fds() -> None:
    response = client.get("/predictive-maintenance/artifacts/config")

    assert response.status_code == 200
    assert response.json()["supported_fds"] == ["FD001", "FD002", "FD003", "FD004"]

