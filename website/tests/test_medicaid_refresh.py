from backend import refresh_medicaid, signal_store
from backend.config import settings


def _row(period, status, final, value):
    row = {"state_abbreviation": "AR", "reporting_period": period,
           "preliminary_or_updated": status, "final_report": final}
    row.update({field: str(value) for field in refresh_medicaid.FIELDS.values()})
    return row


def test_medicaid_prefers_final_and_rejects_equal_rank_conflicts():
    preliminary = _row("202605", "P", "N", 10)
    final = _row("202605", "U", "Y", 11)
    assert refresh_medicaid.select_periods([preliminary, final])[0][1] == final
    try:
        refresh_medicaid.select_periods([final, _row("202605", "U", "Y", 12)])
    except ValueError as exc:
        assert "Conflicting" in str(exc)
    else:
        raise AssertionError("Equal-rank conflict was accepted")


def test_medicaid_refresh_is_idempotent_and_preserves_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        for name in refresh_medicaid.FIELDS:
            definition = {"signal_origin": "model_external_state_feature",
                          "signal_id": name, "cadence": "monthly",
                          "geography_level": "state", "geography_id": "AR",
                          "entity_key": ""}
            db.execute("""INSERT INTO signal_definitions
                (id, signal_origin, signal_id, cadence, geography_level,
                 geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (signal_store.signal_uid(definition), *definition.values()))
    state = {"revised": False}

    def fake_fetch():
        rows = [_row("202605", "P", "N", 10),
                _row("202605", "U", "Y", 11),
                _row("202606", "P", "N", 12)]
        if state["revised"]:
            rows[-1] = _row("202606", "U", "Y", 13)
        return rows, refresh_medicaid._url(0)

    monkeypatch.setattr(refresh_medicaid, "fetch_arkansas_rows", fake_fetch)
    assert refresh_medicaid.refresh_medicaid()["rows_written"] == 16
    assert refresh_medicaid.refresh_medicaid()["rows_written"] == 0
    state["revised"] = True
    assert refresh_medicaid.refresh_medicaid()["rows_written"] == 8
    with signal_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM signal_records").fetchone()[0] == 24
        ids = [row[0] for row in db.execute("SELECT id FROM signal_definitions")]
    latest = signal_store.values(ids, latest_only=True)
    assert len(latest) == 8
    assert {row["value"] for row in latest} == {13.0}
    assert all(row["data_quality"] == "final;publication_time_unknown" for row in latest)
    assert all(row["source_url"].startswith(refresh_medicaid.API_URL) for row in latest)
