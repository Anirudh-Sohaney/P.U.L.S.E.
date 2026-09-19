"""Tests for backend API endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_token(client):
    """Get an authentication token."""
    response = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_login_success(self, client):
        response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure(self, client):
        response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"

    def test_unauthorized_access(self, client):
        response = client.get("/api/auth/me")
        assert response.status_code == 401


class TestDataEndpoints:
    """Tests for data endpoints."""

    def test_get_schema(self, client, auth_headers):
        response = client.get("/api/data/schema", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "fields" in data
        assert "medication_id" in data["fields"]

    def test_validate_csv(self, client, auth_headers):
        csv_content = """medication_id,medication_name,medication_category,current_inventory,reorder_point,max_capacity,prescription_volume_30d,avg_weekly_demand,supplier_id,supplier_lead_time_days,supplier_reliability_score,historical_shortage_count
MED001,Aspirin,analgesic,100,50,500,30,10,SUP001,7,0.9,2"""
        response = client.post(
            "/api/data/validate",
            files={"file": ("test.csv", csv_content, "text/csv")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "errors" in data

    def test_validate_non_csv_fails(self, client, auth_headers):
        response = client.post(
            "/api/data/validate",
            files={"file": ("test.txt", "content", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400


class TestModelEndpoints:
    """Tests for model endpoints."""

    def test_get_model_info_no_model(self, client, auth_headers):
        response = client.get("/api/model/info", headers=auth_headers)
        assert response.status_code == 404

    def test_get_feature_importance_no_model(self, client, auth_headers):
        response = client.get("/api/model/features", headers=auth_headers)
        assert response.status_code == 404


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_no_model(self, client, auth_headers):
        response = client.post(
            "/api/predictions/predict",
            json={
                "medication_id": "MED001",
                "medication_name": "Aspirin",
                "medication_category": "analgesic",
                "current_inventory": 100,
                "reorder_point": 50,
                "max_capacity": 500,
                "prescription_volume_30d": 30,
                "avg_weekly_demand": 10,
                "supplier_id": "SUP001",
                "supplier_lead_time_days": 7,
                "supplier_reliability_score": 0.9,
                "historical_shortage_count": 2,
            },
            headers=auth_headers,
        )
        assert response.status_code == 503

    def test_get_threshold(self, client, auth_headers):
        response = client.get("/api/predictions/threshold", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "threshold" in data

    def test_update_threshold(self, client, auth_headers):
        response = client.put(
            "/api/predictions/threshold?threshold=0.7",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.7

    def test_invalid_threshold(self, client, auth_headers):
        response = client.put(
            "/api/predictions/threshold?threshold=1.5",
            headers=auth_headers,
        )
        assert response.status_code == 400
