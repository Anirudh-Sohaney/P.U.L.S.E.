"""Boot the API against a private temporary copy of a verified backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import sqlite3
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings
from backend.refresh_cms_partd_source import source_snapshot
from backend.signal_store import backfill_catalog_source_urls, catalog_rows
from scripts.backup_databases import verify_backup
from scripts.audit_signal_store import audit as audit_signal_store
from scripts.import_signal_catalog import DEFAULT_HISTORY, DEFAULT_HISTORY_SHA256


def check_restored_auth(client: TestClient, origin: str) -> None:
    """Exercise authentication only against the disposable restored database."""
    account = {"username": "restore_" + secrets.token_hex(8),
               "password": secrets.token_urlsafe(32)}
    headers = {"Origin": origin, "Sec-Fetch-Site": "same-origin"}
    registered = client.post("/api/auth/register", json=account, headers=headers)
    logged_in = client.post("/api/auth/session", data=account, headers=headers)
    current = client.get("/api/auth/me")
    if (registered.status_code != 201 or logged_in.status_code != 200
            or current.status_code != 200
            or current.json().get("username") != account["username"]):
        raise ValueError("Restored authentication did not register and authenticate an account")
    csrf = client.cookies.get("pulse_csrf")
    signed_out = client.post("/api/auth/logout-all", headers={
        **headers, "X-CSRF-Token": csrf or ""})
    if signed_out.status_code != 200 or client.get("/api/auth/me").status_code != 401:
        raise ValueError("Restored authentication did not revoke its session")


def repair_historical_source_urls(database: Path) -> int:
    """Apply the narrow provenance migration to a temporary restored copy."""
    if hashlib.sha256(DEFAULT_HISTORY.read_bytes()).hexdigest() != DEFAULT_HISTORY_SHA256:
        raise ValueError("Bundled historical catalog checksum mismatch")
    _, observations, rejected = catalog_rows(DEFAULT_HISTORY)
    if rejected:
        raise ValueError("Bundled historical catalog has invalid rows")
    with sqlite3.connect(database) as db:
        return backfill_catalog_source_urls(db, observations)


def rehearse(snapshot: Path, *, expected_catalog: int = 1312,
             expected_historical_records: int = 4097) -> dict:
    manifest = verify_backup(snapshot)
    if "cms_partd_source_manifest.json" not in manifest["files"]:
        raise ValueError("Snapshot does not include the CMS source pair")
    with tempfile.TemporaryDirectory(prefix="pulse-restore-") as temporary:
        restored = Path(temporary)
        restored.chmod(0o700)
        for name in manifest["files"]:
            shutil.copy2(snapshot / name, restored / name)
        repaired_urls = repair_historical_source_urls(restored / "signals.sqlite3")
        signal_audit = audit_signal_store(restored / "signals.sqlite3",
                                          expected_definitions=expected_catalog)
        historical_records = signal_audit["records_by_source_kind"].get(
            "historical_catalog", 0)
        if historical_records < expected_historical_records:
            raise ValueError("Restored backup has only "
                             f"{historical_records} historical catalog records; "
                             f"expected at least {expected_historical_records}")
        with sqlite3.connect(restored / "accounts.sqlite3") as accounts:
            restored_accounts = accounts.execute("SELECT COUNT(username) FROM accounts").fetchone()[0]
        previous_data_path = settings.DATA_PATH
        try:
            settings.DATA_PATH = str(restored)
            cms_source, cms_manifest = source_snapshot()
            origin = settings.PUBLIC_ORIGIN or "http://testserver"
            with TestClient(create_app(), base_url=origin) as client:
                ready_response = client.get("/ready")
                catalog_response = client.get("/api/v1/signals/catalog?limit=1500")
                ranking_response = client.get("/api/v1/signals/demand/drugs?limit=1")
                check_restored_auth(client, origin)
            ready = ready_response.json()
            catalog = catalog_response.json()
            ranking = ranking_response.json()
            ranked_rows = ranking.get("drugs", [])
            if (ready_response.status_code != 200 or not ready["ready"]
                    or ready["signal_definitions"] < expected_catalog
                    or catalog_response.status_code != 200
                    or catalog["count"] < expected_catalog
                    or ranking_response.status_code != 200
                    or ranking["total"] < 1
                    or any(not row.get("id") for row in ranked_rows)):
                raise ValueError("Restored API did not satisfy its catalog and ranking checks")
            return {"catalog_signals": catalog["count"],
                    "restored_accounts": restored_accounts,
                    "authentication_verified": True,
                    "audited_signal_records": signal_audit["records"],
                    "audited_news_article_versions": signal_audit.get("news_article_versions", 0),
                    "historical_catalog_records": historical_records,
                    "repaired_historical_source_urls": repaired_urls,
                    "ranked_drugs": ranking["total"],
                    "demand_target_year": ranking.get("target_year"),
                    "cms_source_year": max(cms_manifest["years"]),
                    "cms_source_file": cms_source.name}
        finally:
            settings.DATA_PATH = previous_data_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expected-catalog", type=int, default=1312)
    parser.add_argument("--expected-historical-records", type=int, default=4097)
    args = parser.parse_args()
    print(json.dumps(rehearse(args.snapshot, expected_catalog=args.expected_catalog,
                             expected_historical_records=args.expected_historical_records),
                     indent=2))


if __name__ == "__main__":
    main()
