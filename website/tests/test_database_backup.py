"""Online backups are restorable, private, and reject corruption."""

import sqlite3
import stat
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.backup_databases import (
    BackupAlreadyRunning, create_backup, exclusive_backup, verify_backup,
)
from backend.backup_worker import backup_due
from backend.config import settings
from backend.refresh_cms_partd_source import source_snapshot
from scripts import rehearse_restore


@pytest.mark.parametrize("production", [False, True])
def test_restored_authentication_probe_supports_local_and_secure_cookies(
        tmp_path, monkeypatch, production):
    from fastapi.testclient import TestClient
    from backend.app import create_app

    origin = "https://pulse.example.com" if production else "http://testserver"
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "ENVIRONMENT", "production" if production else "development")
    monkeypatch.setattr(settings, "COOKIE_SECURE", production)
    monkeypatch.setattr(settings, "PUBLIC_ORIGIN", origin if production else "")
    with TestClient(create_app(), base_url=origin) as client:
        rehearse_restore.check_restored_auth(client, origin)
    with sqlite3.connect(tmp_path / "accounts.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_online_backup_restores_committed_rows_and_rejects_corruption(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        with sqlite3.connect(source / name) as db:
            db.execute("CREATE TABLE sample (value TEXT)")
            db.execute("INSERT INTO sample VALUES (?)", (name,))
    writer = sqlite3.connect(source / "signals.sqlite3")
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("INSERT INTO sample VALUES ('uncommitted')")
    try:
        snapshot = create_backup(source, tmp_path / "backups")
    finally:
        writer.rollback()
        writer.close()
    assert verify_backup(snapshot)["files"]["signals.sqlite3"]["bytes"] > 0
    assert stat.S_IMODE(snapshot.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o700
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        assert stat.S_IMODE((snapshot / name).stat().st_mode) == 0o600
        with sqlite3.connect(snapshot / name) as restored:
            assert restored.execute("SELECT value FROM sample").fetchall() == [(name,)]
    with (snapshot / "signals.sqlite3").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_backup(snapshot)


def test_snapshot_lock_rejects_a_second_backup_process(tmp_path):
    root = tmp_path / "backups"
    environment = os.environ.copy()
    with exclusive_backup(root):
        assert stat.S_IMODE((root / ".backup.lock").stat().st_mode) == 0o600
        with pytest.raises(BackupAlreadyRunning):
            create_backup(tmp_path / "missing-data", root)
        code = ("from pathlib import Path; import sys; "
                "from scripts.backup_databases import create_backup; "
                "create_backup(Path(sys.argv[1]), Path(sys.argv[2]))")
        result = subprocess.run([sys.executable, "-c", code,
                                 str(tmp_path / "missing-data"), str(root)],
                                cwd=Path(__file__).resolve().parents[1],
                                env=environment, capture_output=True, text=True,
                                timeout=10, check=False)
        assert result.returncode != 0
        assert "Another PULSE backup" in result.stderr
    with exclusive_backup(root):
        pass


def test_backup_retention_keeps_latest_committed_snapshots(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        with sqlite3.connect(source / name) as db:
            db.execute("CREATE TABLE sample (value INTEGER)")
    root = tmp_path / "backups"
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    snapshots = [create_backup(source, root, keep=2, now=start + timedelta(days=day))
                 for day in range(3)]
    assert not snapshots[0].exists()
    assert all(path.exists() for path in snapshots[1:])


def test_backup_restores_the_committed_cms_source_pair(tmp_path, monkeypatch):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        with sqlite3.connect(source / name) as db:
            db.execute("CREATE TABLE sample (value INTEGER)")
    cms_source = tmp_path / "cms.csv.gz"
    cms_source.write_bytes(gzip.compress(b"year,state,drug_key,demand_claims\n"
                                         b"2024,arkansas,Drug A,20\n"))
    manifest = {"years": [2024], "source_urls": ["https://data.cms.gov/partd.csv"]}
    snapshot = create_backup(source, tmp_path / "backups", cms_source=cms_source,
                             cms_manifest=manifest)
    verified = verify_backup(snapshot)
    cms_name = next(name for name in verified["files"] if name.startswith("cms_partd_claims."))
    restored = tmp_path / "restored"
    restored.mkdir()
    shutil.copy2(snapshot / cms_name, restored / cms_name)
    shutil.copy2(snapshot / "cms_partd_source_manifest.json",
                 restored / "cms_partd_source_manifest.json")
    monkeypatch.setattr(settings, "DATA_PATH", str(restored))
    actual_source, actual_manifest = source_snapshot()
    assert actual_source.read_bytes() == cms_source.read_bytes()
    assert actual_manifest["years"] == [2024]


def test_backup_worker_catches_up_and_replaces_corrupt_snapshot(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        with sqlite3.connect(source / name) as db:
            db.execute("CREATE TABLE sample (value INTEGER)")
    backups = source / "backups"
    scheduled = datetime(2026, 10, 3, 4, tzinfo=timezone.utc)
    assert backup_due(backups, scheduled + timedelta(minutes=1))
    create_backup(source, backups, now=scheduled + timedelta(minutes=2))
    assert backup_due(backups, scheduled + timedelta(minutes=3))
    cms_source = tmp_path / "cms.csv.gz"
    cms_source.write_bytes(gzip.compress(b"year,state,drug_key,demand_claims\n"
                                         b"2024,arkansas,Drug A,20\n"))
    snapshot = create_backup(source, backups, now=scheduled + timedelta(minutes=4),
                             cms_source=cms_source, cms_manifest={"years": [2024]})
    assert not backup_due(backups, scheduled + timedelta(hours=1))
    assert backup_due(backups, scheduled + timedelta(days=1))
    with (snapshot / "accounts.sqlite3").open("ab") as stream:
        stream.write(b"corrupt")
    assert backup_due(backups, scheduled + timedelta(hours=1))


def test_backup_rejects_ranking_from_a_different_cms_source_generation(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    with sqlite3.connect(source / "accounts.sqlite3") as db:
        db.execute("CREATE TABLE sample (value INTEGER)")
    with sqlite3.connect(source / "signals.sqlite3") as db:
        db.execute("""CREATE TABLE demand_forecast_runs
            (source_sha256 TEXT, source_year INTEGER, target_year INTEGER,
             created_at TEXT, method TEXT)""")
        db.execute("""INSERT INTO demand_forecast_runs VALUES
            ('old-source-digest', 2024, 2026, '2026-10-04T16:00:00+00:00',
             'cms_partd_two_year_persistence_v2')""")
    cms = tmp_path / "cms.csv.gz"
    cms.write_bytes(gzip.compress(b"year,state,drug_key,demand_claims\n"
                                  b"2024,arkansas,Drug A,20\n"))
    digest = hashlib.sha256(cms.read_bytes()).hexdigest()
    cms_manifest = {"years": [2024], "normalized_sha256": digest}
    backups = tmp_path / "backups"
    with pytest.raises(ValueError, match="no committed CMS source pair"):
        create_backup(source, backups)
    with pytest.raises(ValueError, match="ranking and CMS source generation disagree"):
        create_backup(source, backups, cms_source=cms, cms_manifest=cms_manifest)
    assert not [path for path in backups.iterdir() if path.is_dir()]
    with sqlite3.connect(source / "signals.sqlite3") as db:
        db.execute("UPDATE demand_forecast_runs SET source_sha256=?", (digest,))
    snapshot = create_backup(source, backups, cms_source=cms,
                             cms_manifest=cms_manifest)
    assert verify_backup(snapshot)["files"]["signals.sqlite3"]["bytes"] > 0


def test_restore_rehearsal_rejects_incomplete_historical_catalog_before_api_start(
        tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "signals.sqlite3").touch()
    monkeypatch.setattr(rehearse_restore, "verify_backup",
                        lambda _: {"files": {"signals.sqlite3": {},
                                             "cms_partd_source_manifest.json": {}}})
    (snapshot / "cms_partd_source_manifest.json").touch()
    monkeypatch.setattr(rehearse_restore, "audit_signal_store",
                        lambda *args, **kwargs: {
                            "records_by_source_kind": {"historical_catalog": 4096}})
    monkeypatch.setattr(rehearse_restore, "repair_historical_source_urls",
                        lambda _: 0)
    monkeypatch.setattr(rehearse_restore, "create_app",
                        lambda: pytest.fail("API must not start with incomplete history"))
    with pytest.raises(ValueError, match="only 4096 historical catalog records"):
        rehearse_restore.rehearse(snapshot)
