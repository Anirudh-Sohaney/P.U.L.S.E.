"""Account-scoped direct 14-day demand API integration test."""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings


def _register(client: TestClient, username: str) -> dict[str, str]:
    response = client.post("/api/auth/register", json={"username": username, "password": "test-password-123"})
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_uploaded_sales_train_direct_model_and_remain_account_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "demand-data"))
    client = TestClient(create_app())
    alice = _register(client, "alice")
    bob = _register(client, "bob")

    root = Path(__file__).resolve().parents[2]
    sales = pd.read_csv(root / "data" / "synthetic_pharmacy_data" / "arkansas_clinic_daily_pharmacy_sales.csv")
    sales = sales[sales["drug_name"] == "Azithromycin 250 mg tablet"].copy()
    inventory = pd.DataFrame({"drug_name": ["Azithromycin 250 mg tablet"], "on_hand_units": [0]})
    files = {
        "sales_file": ("sales.csv", sales.to_csv(index=False).encode(), "text/csv"),
        "inventory_file": ("inventory.csv", inventory.to_csv(index=False).encode(), "text/csv"),
    }
    trained = client.post("/api/demand/train", files=files, headers=alice)
    assert trained.status_code == 200, trained.text
    metrics = trained.json()
    assert metrics["model_version"] == "per_drug_direct_14d_top5_lagged_v3"
    assert metrics["forecast_target"] == "next_14_calendar_days_total_units"
    assert metrics["signal_candidate_count"] == 1312
    assert metrics["signal_feature_count"] == 5

    status = client.get("/api/demand/status", headers=alice)
    assert status.status_code == 200 and status.json()["ready"] is True
    forecasts = client.get("/api/demand/forecasts", headers=alice)
    assert forecasts.status_code == 200 and forecasts.json()["count"] == 14
    plan = client.get("/api/demand/planning", headers=alice)
    assert plan.status_code == 200 and plan.json()["count"] == 1
    assert abs(sum(row["predicted_units"] for row in forecasts.json()["forecasts"]) - plan.json()["items"][0]["forecast_14d_units"]) < 0.01
    signals = client.get("/api/demand/signals/relevant", headers=alice)
    assert signals.status_code == 200
    signal_payload = signals.json()
    assert signal_payload["at_risk_drugs"] == ["Azithromycin 250 mg tablet"]
    assert len(signal_payload["signals"]) <= 3
    for signal in signal_payload["signals"]:
        assert signal["kind"] in {"demand", "news"}
        assert 0 <= signal["score"] <= 1
        if signal["kind"] == "demand":
            assert len(signal["demand_context"]["points"]) == 14
            assert signal["article"] is None
        else:
            assert signal["demand_context"] is None
            assert signal["article"] and signal["article"]["title"]

    assert client.get("/api/demand/status", headers=bob).json()["ready"] is False
    assert client.get("/api/demand/planning", headers=bob).status_code == 409
