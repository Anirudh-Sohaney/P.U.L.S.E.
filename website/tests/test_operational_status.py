"""Container probes must detect stalled work and invalid offsite copies."""

from datetime import datetime, timezone

from scripts import operational_status as status


def test_worker_probe_reports_stale_heartbeat(monkeypatch):
    monkeypatch.setattr(status, "freshness", lambda: {
        "worker_last_check_at": "2026-10-04T00:00:00+00:00",
        "worker_recent": False,
        "failed_sources": ["gdelt_recent_news"],
    })
    report = status.worker_status()
    assert report["healthy"] is False
    assert report["reason"] == "refresh_worker_heartbeat_stale_or_missing"
    assert report["failed_sources"] == ["gdelt_recent_news"]


def test_source_probe_flags_failed_refresh(monkeypatch):
    monkeypatch.setattr(status, "REFRESH_TASKS", ((lambda: {}, ("gdelt_recent_news",)),))
    monkeypatch.setattr(status, "freshness", lambda: {
        "source_checks": [{"source_name": "gdelt_recent_news", "last_status": "failed",
                           "last_attempt_at": "2026-10-05T12:00:00+00:00"}],
        "failed_sources": ["gdelt_recent_news"],
    })
    report = status.sources_status(datetime(2026, 10, 5, 13, tzinfo=timezone.utc))
    assert report["healthy"] is False
    assert report["reason"] == "upstream_source_refresh_failed"


def test_source_probe_flags_unattempted_source(monkeypatch):
    monkeypatch.setattr(status, "REFRESH_TASKS", ((lambda: {}, ("fda_drugs_rss",)),))
    monkeypatch.setattr(status, "freshness", lambda: {
        "source_checks": [], "failed_sources": [],
    })
    report = status.sources_status()
    assert report["healthy"] is False
    assert report["missing_sources"] == ["fda_drugs_rss"]


def test_source_probe_flags_old_success(monkeypatch):
    monkeypatch.setattr(status, "REFRESH_TASKS", ((lambda: {}, ("fda_drugs_rss",)),))
    monkeypatch.setattr(status, "freshness", lambda: {
        "source_checks": [{"source_name": "fda_drugs_rss",
                           "last_attempt_at": "2026-10-04T16:01:00+00:00"}],
        "failed_sources": [],
    })
    report = status.sources_status(datetime(2026, 10, 5, 17, tzinfo=timezone.utc))
    assert report["healthy"] is False
    assert report["stale_sources"] == ["fda_drugs_rss"]


def test_backup_probe_checks_encrypted_copy_against_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "backups" / "20261005T040001.000000Z"
    snapshot.mkdir(parents=True)
    export_dir = tmp_path / "offsite"
    export_dir.mkdir()
    (export_dir / f"{snapshot.name}.enc").write_bytes(b"test")
    key_file = tmp_path / "key"
    key_file.write_text("test")
    monkeypatch.setattr(status, "latest_verified_backup", lambda *_: snapshot)
    monkeypatch.setattr(status, "load_key", lambda *_: b"test")
    monkeypatch.setattr(status, "verify_backup", lambda *_: {"files": {"db": "source"}})
    monkeypatch.setattr(status, "verify_encrypted", lambda *_: {"files": {"db": "other"}})
    monkeypatch.setenv("PULSE_BACKUP_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("PULSE_BACKUP_KEY_FILE", str(key_file))
    report = status.backup_status(datetime(2026, 10, 5, 5, tzinfo=timezone.utc))
    assert report["healthy"] is False
    assert report["reason"] == "encrypted_export_invalid:ValueError"

    monkeypatch.setattr(status, "verify_encrypted", lambda *_: {"files": {"db": "source"}})
    report = status.backup_status(datetime(2026, 10, 5, 5, tzinfo=timezone.utc))
    assert report["healthy"] is True
    assert report["encrypted_export"] == f"{snapshot.name}.enc"
