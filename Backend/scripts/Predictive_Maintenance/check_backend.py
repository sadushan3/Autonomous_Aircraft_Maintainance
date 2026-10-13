from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.schemas.combined_schema import CombinedPredictionRequest


def main() -> None:
    print("Registered routes:")
    for route in sorted(app.routes, key=lambda item: getattr(item, "path", "")):
        methods = ",".join(sorted(getattr(route, "methods", []) or []))
        print(f"  {methods:20} {route.path}")

    print("\nArtifact discovery:")
    status = app.state.artifact_loader.status()
    pprint(status)

    sample_path = PROJECT_ROOT / "sample_data" / "sample_combined_request.json"
    print(f"\nLoading sample request: {sample_path}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    request = CombinedPredictionRequest(**payload)
    print(
        f"Sample OK: fd_subset={request.fd_subset}, rows={len(request.sequence)}, "
        f"last_cycle={max(row.cycle for row in request.sequence)}"
    )

    required_loaded = [
        item["loaded"]
        for section in ("rul_lifecycle", "failure_risk")
        for item in status[section]
        if item["fd_subset"] == request.fd_subset
    ]
    if not all(required_loaded):
        print("\nPrediction skipped because one or more sample FD artifacts are missing.")
        return

    print("\nRunning direct combined service prediction:")
    try:
        response = app.state.combined_service.predict(request)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return
    pprint(response.model_dump())


if __name__ == "__main__":
    main()
