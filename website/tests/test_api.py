"""Tests for the current account and demand API contract."""

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "data"))
    return TestClient(create_app())


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_registration_login_and_account_status(client):
    account = {"username": "pharmacy_one", "password": "secure-password-123"}
    registered = client.post("/api/auth/register", json=account)
    assert registered.status_code == 201
    assert registered.json()["token_type"] == "bearer"
    assert client.post("/api/auth/register", json=account).status_code == 409

    bad_login = client.post("/api/auth/token", data={"username": account["username"], "password": "wrong-password"})
    assert bad_login.status_code == 401
    login = client.post("/api/auth/token", data=account)
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/api/auth/me", headers=headers).json()["username"] == account["username"]
    assert client.get("/api/demand/status", headers=headers).json()["ready"] is False
    assert client.get("/api/demand/forecasts", headers=headers).status_code == 409


def test_demand_endpoints_require_authentication(client):
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/demand/status").status_code == 401
    assert client.get("/api/demand/planning").status_code == 401
