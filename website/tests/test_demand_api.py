"""End-to-end tests for the locked synthetic demand-demo API."""

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings


def _headers(client: TestClient) -> dict[str, str]:
    token = client.post(
        "/api/auth/token", data={"username": "admin", "password": "admin123"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_locked_demo_serves_synthetic_forecasts_and_rejects_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "demand-data"))
    client = TestClient(create_app())
    headers = _headers(client)

    status = client.get("/api/demand/status", headers=headers)
    assert status.status_code == 200
    payload = status.json()
    assert payload["mode"] == "locked_synthetic_demo"
    assert payload["uploads_enabled"] is False
    assert payload["metrics"]["drugs_trained"] == 30
    assert payload["metrics"]["source_rows"] == 32880

    forecast = client.get(
        "/api/demand/forecasts?drug_name=Amoxicillin%20500%20mg%20capsule",
        headers=headers,
    )
    assert forecast.status_code == 200
    assert forecast.json()["count"] == 14

    upload = {"file": ("daily_sales.csv", "date,drug_name,units_sold", "text/csv")}
    assert client.post("/api/demand/validate", files=upload, headers=headers).status_code == 403
    assert client.post("/api/demand/ingest", files=upload, headers=headers).status_code == 403
