"""Release audit rejects malformed source timing before it reaches the API."""

import pytest

from backend import signal_store
from backend.config import settings
from scripts.audit_signal_store import audit


def test_signal_store_audit_accepts_known_historical_gap_and_rejects_bad_live_dates(
        tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key)
            VALUES ('sig_test', 'observed', 'test', 'monthly', 'state', 'AR', '')""")
        for row_hash, kind, source_time in (
            ("old", "historical_catalog", ""),
            ("live", "official_api", "2026-10-04T16:00:00+00:00"),
        ):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, source_kind)
                VALUES (?, 'sig_test', '2026-09-30', '2026-09-30', 2,
                        ?, '2026-10-04T16:01:00+00:00', ?)""",
                (row_hash, source_time, kind))
    path = tmp_path / "signals.sqlite3"
    result = audit(path, expected_definitions=1, check_bundled_history=False)
    assert result["records"] == 2
    assert result["unknown_historical_source_timestamps"] == 1

    with signal_store.connect() as db:
        db.execute("UPDATE signal_records SET source_timestamp='' WHERE row_hash='live'")
    with pytest.raises(ValueError, match="missing live source_timestamp"):
        audit(path, expected_definitions=1, check_bundled_history=False)

    with signal_store.connect() as db:
        db.execute("""UPDATE signal_records SET source_timestamp='2026-10-04T16:00:00+00:00',
            observation_date='2026-99-99' WHERE row_hash='live'""")
    with pytest.raises(ValueError, match="invalid observation_date"):
        audit(path, expected_definitions=1, check_bundled_history=False)
