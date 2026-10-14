"""Public signal API preserves identity, history, and conflicting source values."""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import sys
import pytest

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import settings
from backend import signal_store
from backend.signal_store import import_catalog, import_definitions, import_news_bridge_history

HISTORY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "signals_2023_2025.csv.gz"
NEWS_BRIDGE = Path(__file__).resolve().parents[1] / "catalog" / "news_only_catalog_features.csv.gz"


def test_verified_news_bridge_backfills_only_missing_history(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    first = import_news_bridge_history(NEWS_BRIDGE)
    assert first["periods"] == 97
    assert first["signal_ids"] == 20
    assert first["matched_existing"] == 720
    assert first["inserted"] == 1220
    assert import_news_bridge_history(NEWS_BRIDGE)["inserted"] == 0
    with signal_store.connect() as db:
        assert db.execute("""SELECT COUNT(*) FROM signal_records r
            JOIN signal_definitions d ON d.id=r.signal_uid
            WHERE d.signal_origin='model_news_output'""").fetchone()[0] == 1940
    client = TestClient(create_app())
    uid = next(row["id"] for row in client.get("/api/v1/signals/catalog").json()["signals"]
               if row["signal_id"] == "arkansas_anti_infective_disruption")
    old = client.post("/api/v1/signals/history", json={"ids": [uid], "date": "2018-01-31"}).json()["values"]
    assert len(old) == 1
    assert old[0]["source_kind"] == "historical_news_bridge"
    assert old[0]["source_timestamp"] == ""
    assert old[0]["usable"] is False
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["missing_ids"] == [uid]
    assert latest["unusable_recorded_values"][0]["id"] == uid
    assert latest["unusable_recorded_values"][0]["usable"] is False
    assert latest["unusable_recorded_values"][0]["observation_date"] == "2026-01-31"
    changed = tmp_path / "changed-news.csv.gz"
    changed.write_bytes(NEWS_BRIDGE.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_news_bridge_history(changed)


def test_initialize_preserves_unverified_legacy_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with sqlite3.connect(tmp_path / "signals.sqlite3") as db:
        db.execute("CREATE TABLE signal_observations (id TEXT PRIMARY KEY, value REAL)")
        db.execute("INSERT INTO signal_observations VALUES ('old-only', 99)")
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key)
            VALUES ('new-id', 'observed', 'new-value', 'daily', 'state', 'AR', '')""")
        db.execute("""INSERT INTO signal_records
            (row_hash, signal_uid, signal_date, observation_date, value,
             source_timestamp, ingested_at, source_kind)
            VALUES ('new-row', 'new-id', '2026-10-01', '2026-10-01', 1,
                    '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00', 'official_api')""")
    signal_store.initialize()
    with sqlite3.connect(tmp_path / "signals.sqlite3") as db:
        assert db.execute("SELECT value FROM signal_observations WHERE id='old-only'").fetchone()[0] == 99


def test_definitions_only_seed_is_not_ready_without_historical_value_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    manifest = Path(__file__).resolve().parents[1] / "catalog" / "signal_definitions.json"
    first = import_definitions(manifest)
    assert first == {"definitions_seen": 1312, "definitions_inserted": 1312}
    assert import_definitions(manifest)["definitions_inserted"] == 0
    client = TestClient(create_app())
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["historical_catalog_records"] == 0
    catalog = client.get("/api/v1/signals/catalog").json()["signals"]
    assert len(catalog) == 1312
    latest = client.post("/api/v1/signals/latest", json={
        "ids": [row["id"] for row in catalog]}).json()
    assert latest["values"] == []
    assert len(latest["missing_ids"]) == 1312
    assert latest["unusable_recorded_values"] == []


def test_startup_import_requires_pinned_bundled_history(tmp_path, monkeypatch):
    from scripts import import_signal_catalog as startup

    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr(startup, "DEFAULT_HISTORY", tmp_path / "missing_history.csv.gz")
    monkeypatch.setattr(sys, "argv", ["import_signal_catalog"])
    with pytest.raises(FileNotFoundError):
        startup.main()


def test_bundled_history_matches_tested_source(tmp_path, monkeypatch, capsys):
    from scripts import import_signal_catalog as startup
    from scripts.audit_signal_store import audit

    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["import_signal_catalog"])
    assert startup.DEFAULT_HISTORY.read_bytes() == HISTORY_FIXTURE.read_bytes()
    startup.main()
    result = json.loads(capsys.readouterr().out)
    assert result["history"]["observations_seen"] == 6484
    assert result["history"]["observations_inserted"] == 4097
    assert result["database"]["definitions"] == 1312
    ready = TestClient(create_app()).get("/ready")
    assert ready.status_code == 200
    assert ready.json()["historical_catalog_records"] == 4097
    assert audit(tmp_path / "signals.sqlite3")["records"] == 5317
    with signal_store.connect() as db:
        source_row = db.execute("""SELECT row_hash, source_url FROM signal_records
            WHERE source_kind='historical_catalog' AND source_url!='' LIMIT 1""").fetchone()
        db.execute("UPDATE signal_records SET source_url='' WHERE row_hash=?",
                   (source_row["row_hash"],))
    startup.main()
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["history"]["observations_inserted"] == 0
    with signal_store.connect() as db:
        restored_url = db.execute("SELECT source_url FROM signal_records WHERE row_hash=?",
                                  (source_row["row_hash"],)).fetchone()[0]
        assert restored_url == source_row["source_url"]
    changed = tmp_path / "changed_history.csv.gz"
    changed.write_bytes(HISTORY_FIXTURE.read_bytes() + b"changed")
    monkeypatch.setattr(startup, "DEFAULT_HISTORY", changed)
    with pytest.raises(ValueError, match="checksum mismatch"):
        startup.main()
    with signal_store.connect() as db:
        db.execute("UPDATE signal_records SET value=value+1 WHERE row_hash=?",
                   (source_row["row_hash"],))
    with pytest.raises(ValueError, match="bundled historical row changed"):
        audit(tmp_path / "signals.sqlite3")


def test_catalog_and_observation_queries_do_not_impute_missing_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    imported = import_catalog(HISTORY_FIXTURE)
    assert imported["definitions"] == 1312
    assert imported["rejected_rows"] == 0
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog").json()["signals"]
    assert len(catalog) == 1312
    assert len({row["id"] for row in catalog}) == 1312
    percent_matches = client.get("/api/v1/signals/catalog", params={"search": "%"}).json()["signals"]
    assert 0 < len(percent_matches) < len(catalog)
    assert all("%" in row["signal_id"] or "%" in row["entity_key"]
               for row in percent_matches)
    all_ids = [row["id"] for row in catalog]
    bulk = client.post("/api/v1/signals/latest", json={"ids": all_ids})
    assert bulk.status_code == 200
    values = bulk.json()
    found = {row["id"] for row in values["values"]}
    missing = set(values["missing_ids"])
    assert found.isdisjoint(missing)
    assert found | missing == set(all_ids)
    assert client.post("/api/v1/signals/history", json={
        "ids": all_ids[:101]}).status_code == 422
    uid = next(row["id"] for row in catalog if row["signal_id"] == "dual_eligible_rate")
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["values"][0]["id"] == uid
    assert latest["values"][0]["source_timestamp"]
    absent = client.post("/api/v1/signals/history", json={
        "ids": [uid], "date": "2024-06-01",
    }).json()
    assert absent["values"] == []
    full_span = client.post("/api/v1/signals/history", json={
        "ids": [uid], "start_date": "2013-01-01", "end_date": "2026-12-31",
    })
    assert full_span.status_code == 200
    assert full_span.json()["count"] > 0
    with signal_store.connect() as db:
        dense_uid = db.execute("""SELECT signal_uid FROM signal_records
            GROUP BY signal_uid ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()[0]
    monkeypatch.setattr(signal_store, "MAX_HISTORY_SOURCE_ROWS", 2)
    too_large = client.post("/api/v1/signals/history", json={"ids": [dense_uid]})
    assert too_large.status_code == 413
    assert "use fewer IDs or a shorter date range" in too_large.json()["detail"]


def test_conflicting_source_values_are_marked_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog", params={"search": "adult_medicaid_enrollment"}).json()["signals"]
    uid = next(row["id"] for row in catalog if row["signal_id"] == "adult_medicaid_enrollment")
    history = client.post("/api/v1/signals/history", json={
        "ids": [uid], "date": "2024-01-31",
    }).json()["values"]
    assert len(history) > 1
    assert all(row["ambiguous"] for row in history)
    assert len({row["value"] for row in history}) > 1
    coverage = client.get("/api/v1/signals/coverage").json()
    collisions = [row for row in coverage["signals"]
                  if row["gap_status"] == "identity_dimensions_collapsed"]
    assert len(collisions) == 6


def test_latest_with_conflicting_source_revision_is_not_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    uid = next(row["id"] for row in client.get("/api/v1/signals/catalog",
               params={"search": "adult_medicaid_enrollment"}).json()["signals"]
               if row["signal_id"] == "adult_medicaid_enrollment")
    with signal_store.connect() as db:
        for index, value in enumerate((12.0, 13.0)):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, source_kind)
                VALUES (?, ?, '2026-09-30', '2026-09-30', ?,
                        '2026-10-01T12:00:00+00:00',
                        '2026-10-01T13:00:00+00:00', 'medicaid_state_performance_api')""",
                (f"test-conflict-{index}", uid, value))
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["values"] == []
    assert latest["missing_ids"] == [uid]
    assert latest["ambiguous_ids"] == [uid]
    assert latest["missing_details"][0]["reason"] == "conflicting_latest_revision"
    assert {row["value"] for row in latest["latest_recorded_values"]} == {12.0, 13.0}
    assert all(not row["usable"] and row["ambiguous"] for row in
               latest["latest_recorded_values"])


def test_collapsed_dimension_ids_are_history_only(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog").json()["signals"]
    collapsed = [row for row in catalog
                 if row["signal_id"] in signal_store.COLLAPSED_DIMENSION_SIGNALS]
    assert len(collapsed) == 6
    ids = [row["id"] for row in collapsed]
    latest = client.post("/api/v1/signals/latest", json={"ids": ids}).json()
    assert latest["values"] == []
    assert set(latest["missing_ids"]) == set(ids)
    assert all(row["reason"] == "identity_dimensions_collapsed"
               for row in latest["missing_details"])
    assert {row["id"] for row in latest["unusable_recorded_values"]} == set(ids)
    assert all(row["usable"] is False for row in latest["unusable_recorded_values"])
    for uid in ids:
        history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()
        assert history["values"]
        assert all(row["usable"] is False and
                   row["unusable_reason"] == "identity_dimensions_collapsed"
                   for row in history["values"])


def test_legacy_zero_filled_atc_output_is_not_a_latest_usable_value(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog", params={
        "search": "arkansas_atc_demand_state::A02"}).json()["signals"]
    uid = next(row["id"] for row in catalog if row["signal_id"] ==
               "arkansas_atc_demand_state::A02")
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["values"] == []
    assert latest["missing_ids"] == [uid]
    assert latest["missing_details"] == [{"id": uid,
        "reason": "model_output_requires_current_source_and_validated_rerun",
        "last_recorded_period": "2024-12-31"}]
    assert latest["unusable_recorded_values"][0]["id"] == uid
    assert latest["unusable_recorded_values"][0]["unusable_reason"] == "legacy_zero_filled_feature_vector"
    history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()["values"]
    assert history[0]["usable"] is False
    assert history[0]["unusable_reason"] == "legacy_zero_filled_feature_vector"


def test_historical_drug_model_is_not_latest_without_evaluated_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog", params={
        "search": "cms_part_d_demand_state::"}).json()["signals"]
    uid = catalog[0]["id"]
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["values"] == []
    assert latest["missing_ids"] == [uid]
    assert latest["unusable_recorded_values"][0]["unusable_reason"] == \
        "historical_drug_model_output_not_validated_for_live_use"
    history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()["values"]
    assert history
    assert all(row["usable"] is False for row in history)
    assert all(row["unusable_reason"] ==
               "historical_drug_model_output_not_validated_for_live_use" for row in history)


def test_latest_keeps_usable_revision_when_newer_legacy_revision_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    definition = {"signal_origin": "derived_demand_output",
                  "signal_id": "cms_part_d_demand_state::EXAMPLE", "cadence": "annual",
                  "geography_level": "state", "geography_id": "AR", "entity_key": "EXAMPLE"}
    uid = signal_store.signal_uid(definition)
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, *definition.values()))
        for row_hash, value, source_time, source_kind in (
            ("evaluated", 3, "2026-09-01T00:00:00+00:00",
             "cms_partd_two_year_persistence_v2"),
            ("legacy", 4, "2026-09-02T00:00:00+00:00", "historical_catalog"),
        ):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, forecast_horizon, source_kind)
                VALUES (?, ?, '2024-12-31', '2024-12-31', ?, ?, ?, '2026', ?)""",
                (row_hash, uid, value, source_time, source_time, source_kind))
    client = TestClient(create_app())
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert [(row["value"], row["source_kind"]) for row in latest["values"]] == [
        (3, "cms_partd_two_year_persistence_v2")]
    assert latest["missing_ids"] == []
    history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()
    assert history["values"][0]["usable"] is False


def test_historical_news_artifact_is_not_a_current_latest_value(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    import_catalog(HISTORY_FIXTURE)
    client = TestClient(create_app())
    catalog = client.get("/api/v1/signals/catalog").json()["signals"]
    uid = next(row["id"] for row in catalog if row["signal_origin"] == "model_news_output")
    latest = client.post("/api/v1/signals/latest", json={"ids": [uid]}).json()
    assert latest["values"] == []
    assert latest["missing_ids"] == [uid]
    assert latest["missing_details"][0]["reason"] == "news_model_refresh_not_verified"
    assert latest["unusable_recorded_values"][0]["unusable_reason"] == \
        "historical_news_artifact_no_verified_live_refresh"
    history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()["values"]
    assert history
    assert history[0]["source_kind"] == "historical_catalog"
    assert history[0]["usable"] is False
    assert history[0]["unusable_reason"] == "historical_news_artifact_no_verified_live_refresh"


def test_unvalidated_new_model_rows_cannot_be_published_as_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    cases = (
        ("derived_demand_output", "arkansas_atc_demand_state::A02", "manual_refresh",
         "atc_model_output_no_validated_live_pipeline"),
        ("model_news_output", "news_shortage_pressure", "manual_refresh",
         "news_model_output_no_validated_live_pipeline"),
        ("derived_demand_output", "cms_part_d_demand_state::EXAMPLE",
         "cms_partd_two_year_persistence_v1",
         "drug_model_source_kind_not_evaluated_for_live_use"),
    )
    ids = []
    with signal_store.connect() as db:
        for index, (origin, signal_id, source_kind, _) in enumerate(cases):
            definition = {"signal_origin": origin, "signal_id": signal_id,
                          "cadence": "annual", "geography_level": "state",
                          "geography_id": "AR", "entity_key": str(index)}
            uid = signal_store.signal_uid(definition)
            ids.append(uid)
            db.execute("""INSERT INTO signal_definitions
                (id, signal_origin, signal_id, cadence, geography_level,
                 geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (uid, *definition.values()))
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, forecast_horizon, source_kind)
                VALUES (?, ?, '2026-10-01', '2026-10-01', 2,
                 '2026-10-02T00:00:00+00:00', '2026-10-02T00:00:00+00:00',
                 '2027', ?)""", (str(index), uid, source_kind))
    client = TestClient(create_app())
    latest = client.post("/api/v1/signals/latest", json={"ids": ids}).json()
    assert latest["values"] == []
    assert latest["missing_ids"] == ids
    for uid, (_, _, _, reason) in zip(ids, cases):
        history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()
        assert history["values"][0]["usable"] is False
        assert history["values"][0]["unusable_reason"] == reason


def test_latest_exposes_newer_unvalidated_record_without_publishing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    definition = {"signal_origin": "derived_demand_output",
                  "signal_id": "cms_part_d_demand_state::EXAMPLE",
                  "cadence": "annual", "geography_level": "state",
                  "geography_id": "AR", "entity_key": "EXAMPLE"}
    uid = signal_store.signal_uid(definition)
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, *definition.values()))
        for row_hash, target, value, source_kind in (
            ("validated", "2026", 3, "cms_partd_two_year_persistence_v2"),
            ("unvalidated", "2027", 4, "manual_refresh"),
        ):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, forecast_horizon, source_kind)
                VALUES (?, ?, '2024-12-31', '2024-12-31', ?,
                 '2026-10-02T00:00:00+00:00', '2026-10-02T00:00:00+00:00', ?, ?)""",
                (row_hash, uid, value, target, source_kind))
    response = TestClient(create_app()).post("/api/v1/signals/latest", json={"ids": [uid]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["values"][0]["value"] == 3
    assert payload["values"][0]["forecast_horizon"] == "2026"
    assert payload["latest_recorded_values"][0]["value"] == 4
    assert payload["latest_recorded_values"][0]["forecast_horizon"] == "2027"
    assert payload["latest_recorded_values"][0]["usable"] is False
    assert payload["latest_recorded_values"][0]["unusable_reason"] == \
        "drug_model_source_kind_not_evaluated_for_live_use"


def test_recent_public_signals_show_latest_observation_and_recording_time(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    definition = {"signal_origin": "model_external_state_feature",
                  "signal_id": "national_unemployment_rate", "cadence": "monthly",
                  "geography_level": "national", "geography_id": "US", "entity_key": ""}
    uid = signal_store.signal_uid(definition)
    recorded = datetime.now(timezone.utc).isoformat()
    earlier = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key, unit, source_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'percent', 'BLS')""",
            (uid, *definition.values()))
        for row_hash, day, ingested, source_kind in (
            ("old", "2026-08-31", earlier, "bls_public_api_v1"),
            ("current", "2026-09-30", recorded, "bls_public_api_v1"),
            ("derived", "2026-09-30", recorded, "bls_public_api_v1_derived_v2"),
        ):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, source_kind)
                VALUES (?, ?, ?, ?, 4.2, ?, ?, ?)""",
                (row_hash, uid, day, day, ingested, ingested, source_kind))
    response = TestClient(create_app()).get("/api/v1/signals/recent?days=3&limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["signals"][0]["id"] == uid
    assert payload["signals"][0]["observation_date"] == "2026-09-30"
    assert payload["signals"][0]["ingested_at"] == recorded


def test_recent_public_signals_omit_unresolved_same_revision_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    definition = {"signal_origin": "model_external_state_feature",
                  "signal_id": "conflicting_rate", "cadence": "monthly",
                  "geography_level": "state", "geography_id": "AR", "entity_key": ""}
    uid = signal_store.signal_uid(definition)
    now = datetime.now(timezone.utc).isoformat()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, *definition.values()))
        for row_hash, value in (("conflict_a", 4.2), ("conflict_b", 5.1)):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, source_kind)
                VALUES (?, ?, '2026-09-30', '2026-09-30', ?, ?, ?, 'bls_public_api_v1')""",
                (row_hash, uid, value, now, now))
    client = TestClient(create_app())
    assert client.get("/api/v1/signals/recent?days=3").json()["signals"] == []
    history = client.post("/api/v1/signals/history", json={"ids": [uid]}).json()["values"]
    assert len(history) == 2
    assert all(row["ambiguous"] for row in history)


def test_latest_rejects_date_filters_and_revision_provenance_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    definition = {"signal_origin": "model_external_state_feature",
                  "signal_id": "example_revision", "cadence": "monthly",
                  "geography_level": "state", "geography_id": "AR", "entity_key": ""}
    uid = signal_store.signal_uid(definition)
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, *definition.values()))
        for row_hash, value, source_time, ingested, source_url in (
            ("earlier_revision", 4, "2026-09-01T00:00:00+00:00",
             "2026-09-01T01:00:00+00:00", "https://example.gov/old"),
            ("later_first", 5, "2026-09-03T00:00:00+00:00",
             "2026-09-03T01:00:00+00:00", "https://example.gov/first"),
            ("later_repeat", 5, "2026-09-03T00:00:00+00:00",
             "2026-09-03T02:00:00+00:00", "https://example.gov/repeat"),
        ):
            db.execute("""INSERT INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, source_kind, source_url)
                VALUES (?, ?, '2026-08-31', '2026-08-31', ?, ?, ?, 'official_api', ?)""",
                (row_hash, uid, value, source_time, ingested, source_url))
    client = TestClient(create_app())
    assert client.post("/api/v1/signals/latest", json={
        "ids": [uid], "date": "2026-08-31"}).status_code == 422
    assert client.post("/api/v1/signals/history", json={
        "ids": [uid], "datee": "2026-08-31"}).status_code == 422
    history = client.post("/api/v1/signals/history", json={
        "ids": [uid], "date": "2026-08-31"}).json()["values"]
    assert len(history) == 1
    assert history[0]["value"] == 5
    assert history[0]["source_url"] == "https://example.gov/repeat"
    revisions = client.post("/api/v1/signals/history", json={
        "ids": [uid], "date": "2026-08-31", "include_revisions": True}).json()["values"]
    assert [(row["value"], row["source_timestamp"]) for row in revisions] == [
        (5, "2026-09-03T00:00:00+00:00"),
        (4, "2026-09-01T00:00:00+00:00"),
    ]
    assert client.post("/api/v1/signals/latest", json={
        "ids": [uid]}).json()["values"] == history


def test_signal_queries_bound_identifier_and_search_size(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    client = TestClient(create_app())
    for route in ("latest", "history"):
        for uid in ("", "x" * 65):
            response = client.post(f"/api/v1/signals/{route}", json={"ids": [uid]})
            assert response.status_code == 422
    for route in ("catalog", "demand/drugs", "sources/sdud/latest"):
        response = client.get(f"/api/v1/signals/{route}", params={"search": "x" * 121})
        assert response.status_code == 422


def test_latest_identifies_unknown_ids_without_inventing_values(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    client = TestClient(create_app())
    response = client.post("/api/v1/signals/latest", json={"ids": ["sig_unknown"]})
    assert response.status_code == 200
    assert response.json()["values"] == []
    assert response.json()["missing_details"] == [{"id": "sig_unknown",
        "reason": "unknown_id", "last_recorded_period": None}]
