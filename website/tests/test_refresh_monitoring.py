"""Worker activity is visible without conflating it with source publication dates."""

from datetime import datetime, timedelta, timezone

from backend import signal_store
from backend.config import settings


def test_freshness_exposes_worker_liveness_and_source_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    empty = signal_store.freshness()
    assert empty["worker_recent"] is False
    assert empty["source_checks"] == []
    assert empty["failed_sources"] == []

    signal_store.record_worker_heartbeat()
    now = datetime.now(timezone.utc).isoformat()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO refresh_runs
            (source_name, started_at, finished_at, status)
            VALUES ('source_a', ?, ?, 'success')""", (now, now))
    fresh = signal_store.freshness()
    assert fresh["worker_recent"] is True
    assert fresh["worker_last_check_at"]
    assert fresh["source_checks"][0]["last_status"] == "success"
    assert fresh["source_checks"][0]["last_success_at"] == now
    assert fresh["failed_sources"] == []

    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    with signal_store.connect() as db:
        db.execute("UPDATE service_heartbeats SET checked_at=?", (old,))
        db.execute("""INSERT INTO refresh_runs
            (source_name, started_at, finished_at, status)
            VALUES ('source_a', ?, ?, 'failed')""", (now, now))
    stale = signal_store.freshness()
    assert stale["worker_recent"] is False
    assert stale["source_checks"][0]["last_status"] == "failed"
    assert stale["source_checks"][0]["last_success_at"] == now
    assert stale["failed_sources"] == ["source_a"]
