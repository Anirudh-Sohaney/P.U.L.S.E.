"""Interactive API documentation loads executable assets from this origin."""

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app


def test_docs_use_pinned_local_assets_and_strict_script_policy():
    client = TestClient(create_app())
    response = client.get("/docs")
    assert response.status_code == 200
    assert "cdn.jsdelivr.net" not in response.text
    assert "<script src=\"/api-doc-assets/swagger-ui-bundle.js\"" in response.text
    assert "<script src=\"/api-doc-assets/swagger-init.js\"" in response.text
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert "unsafe-inline" not in response.headers["content-security-policy"]
    assert client.get("/redoc", follow_redirects=False).headers["location"] == "/docs"
    bundle = client.get("/api-doc-assets/swagger-ui-bundle.js")
    assert bundle.status_code == 200
    assert hashlib.sha256(bundle.content).hexdigest() == (
        "050bc415ee7048dcd881682678f720264e7da5e373f7461d7c58c755305255f7")
    assert Path(__file__).resolve().parents[1].joinpath(
        "backend/docs_assets/LICENSE").is_file()
