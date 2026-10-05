"""Shortage snapshots use the worker's UTC calendar day."""

from datetime import datetime, timezone

from backend import refresh_shortages, signal_store
from backend.config import settings


def test_shortage_snapshot_day_is_utc_at_midnight_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))

    class UtcBoundary(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = datetime(2026, 10, 4, 0, 30, tzinfo=timezone.utc)
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(refresh_shortages, "datetime", UtcBoundary)
    monkeypatch.setattr(refresh_shortages, "fetch_shortages", lambda: ([
        {"generic_name": "Example", "status": "active", "change_date": "10/03/2026"},
    ], "2026-10-03"))

    refresh_shortages.refresh_shortages()
    with signal_store.connect() as db:
        snapshot = db.execute("SELECT snapshot_date, retrieved_at FROM shortage_snapshots").fetchone()
    assert snapshot["snapshot_date"] == "2026-10-04"
    assert snapshot["retrieved_at"].startswith("2026-10-04T00:30:00+00:00")


def test_latest_shortage_view_uses_one_complete_same_day_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    today = datetime.now(timezone.utc).strftime("%m/%d/%Y")
    snapshots = iter([
        ([{"generic_name": "Drug A", "status": "active", "change_date": today}],
         "2026-10-04"),
        ([{"generic_name": "Drug B", "status": "active", "change_date": today}],
         "2026-10-04"),
    ])
    monkeypatch.setattr(refresh_shortages, "fetch_shortages", lambda: next(snapshots))
    refresh_shortages.refresh_shortages()
    refresh_shortages.refresh_shortages()
    latest = signal_store.recent_shortage_changes(days=3)
    assert [row["generic_name"] for row in latest["changes"]] == ["Drug B"]
    assert latest["snapshot_retrieved_at"] is not None
    with signal_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM shortage_snapshots").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM shortage_snapshot_generations").fetchone()[0] == 2
