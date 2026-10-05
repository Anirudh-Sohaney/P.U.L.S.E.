import pytest

from backend import refresh_cms_geo, signal_store
from backend.config import settings


def _row(year: str, value: str) -> dict:
    row = {"YEAR": year, "BENE_GEO_DESC": "National", "BENE_AGE_LVL": "All"}
    row.update({field: value for field in refresh_cms_geo.FIELDS.values()})
    return row


def test_cms_rows_reject_duplicate_year_and_missing_latest_field():
    with pytest.raises(ValueError, match="Duplicate"):
        refresh_cms_geo.parse_rows([_row("2024", "4"), _row("2024", "4")])
    missing = _row("2025", "4")
    missing["BENES_TOTAL_CNT"] = ""
    with pytest.raises(ValueError, match="missing a catalog field"):
        refresh_cms_geo.parse_rows([_row("2024", "4"), missing])


def test_cms_refresh_keeps_source_year_and_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        for name in refresh_cms_geo.FIELDS:
            definition = {"signal_origin": "model_external_state_feature",
                          "signal_id": name, "cadence": "annual",
                          "geography_level": "national", "geography_id": "US",
                          "entity_key": ""}
            db.execute("""INSERT INTO signal_definitions
                (id, signal_origin, signal_id, cadence, geography_level,
                 geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (signal_store.signal_uid(definition), *definition.values()))
    state = {"value": "4"}

    def fake_fetch():
        return [_row("2024", state["value"])], refresh_cms_geo._url(0)

    monkeypatch.setattr(refresh_cms_geo, "fetch_national_rows", fake_fetch)
    assert refresh_cms_geo.refresh_cms_geo()["rows_written"] == 9
    assert refresh_cms_geo.refresh_cms_geo()["rows_written"] == 0
    state["value"] = "5"
    assert refresh_cms_geo.refresh_cms_geo()["rows_written"] == 9
    with signal_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM signal_records").fetchone()[0] == 18
        ids = [row[0] for row in db.execute("SELECT id FROM signal_definitions")]
    latest = signal_store.values(ids, latest_only=True)
    assert len(latest) == 9
    assert {row["observation_date"] for row in latest} == {"2024-12-31"}
    assert {row["value"] for row in latest} == {5.0}
    assert all(row["source_url"].startswith(refresh_cms_geo.API_URL) for row in latest)
