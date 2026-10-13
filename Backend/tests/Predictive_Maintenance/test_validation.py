"""Upstream test, vendored from predictive-maintenance-backend/tests/test_validation.py.

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


def make_sequence(length: int = 30) -> list[dict[str, float]]:
    rows = []
    for cycle in range(1, length + 1):
        row = {
            "cycle": cycle,
            "op_setting_1": -0.0007,
            "op_setting_2": -0.0004,
            "op_setting_3": 100.0,
        }
        for idx in range(1, 22):
            row[f"sensor_{idx}"] = 500.0 + idx + cycle * 0.01
        rows.append(row)
    return rows


def test_invalid_fd_returns_client_error() -> None:
    response = client.post(
        "/predictive-maintenance/predict/combined",
        json={"fd_subset": "FD999", "sequence": make_sequence(70)},
    )

    assert response.status_code in {400, 422}


def test_short_sequence_returns_client_error() -> None:
    response = client.post(
        "/predictive-maintenance/predict/combined",
        json={"fd_subset": "FD001", "sequence": make_sequence(5)},
    )

    assert response.status_code in {400, 422}
