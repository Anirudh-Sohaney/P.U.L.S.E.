"""Public API budgets and client address trust boundary."""

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.config import settings
from backend.public_rate_limit import PublicRateLimiter


def test_bucket_refills_and_bounds_client_state(monkeypatch):
    from backend import public_rate_limit

    now = [100.0]
    monkeypatch.setattr(public_rate_limit.time, "monotonic", lambda: now[0])
    limiter = PublicRateLimiter(capacity=5, refill_per_second=1)
    assert limiter.reserve("first", 5) == 0
    assert limiter.reserve("first", 1) == 1
    now[0] += 1
    assert limiter.reserve("first", 1) == 0


def test_public_budget_cannot_be_evaded_by_untrusted_forwarded_headers(
        tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "TRUSTED_PROXY_IPS", [])
    monkeypatch.setattr(settings, "TRUSTED_PROXY_HOSTNAMES", [])
    monkeypatch.setattr(app_module, "PublicRateLimiter",
                        lambda: PublicRateLimiter(capacity=3, refill_per_second=0.001))
    client = TestClient(app_module.create_app())
    first = client.get("/api/v1/signals/catalog", headers={
        "X-Real-IP": "198.51.100.1", "X-Forwarded-For": "198.51.100.1"})
    assert first.status_code == 200
    denied = client.get("/api/v1/signals/catalog", headers={
        "X-Real-IP": "198.51.100.2", "X-Forwarded-For": "198.51.100.2"})
    assert denied.status_code == 429
    assert int(denied.headers["Retry-After"]) > 0
    assert denied.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/ready").status_code != 429


def test_trusted_proxy_real_ip_gets_separate_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "TRUSTED_PROXY_IPS", ["testclient"])
    monkeypatch.setattr(settings, "TRUSTED_PROXY_HOSTNAMES", [])
    monkeypatch.setattr(app_module, "PublicRateLimiter",
                        lambda: PublicRateLimiter(capacity=3, refill_per_second=0.001))
    client = TestClient(app_module.create_app())
    assert client.get("/api/v1/signals/catalog", headers={
        "X-Real-IP": "198.51.100.1"}).status_code == 200
    assert client.get("/api/v1/signals/catalog", headers={
        "X-Real-IP": "198.51.100.2"}).status_code == 200
    assert client.get("/api/v1/signals/catalog", headers={
        "X-Real-IP": "198.51.100.1"}).status_code == 429
