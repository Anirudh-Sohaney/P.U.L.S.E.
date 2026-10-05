import json

import pandas as pd
import pytest

from backend import refresh_cms_partd_source, refresh_demand_baseline, signal_store
from backend.config import settings


def test_two_year_baseline_uses_real_source_year_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "db"))
    signal_store.initialize()
    definition = {"signal_origin": "derived_demand_output",
                  "signal_id": "cms_part_d_demand_state::Drug 0", "cadence": "annual",
                  "geography_level": "state", "geography_id": "AR",
                  "entity_key": "Drug 0"}
    uid = signal_store.signal_uid(definition)
    with signal_store.connect() as db:
        db.execute("""INSERT INTO signal_definitions
            (id, signal_origin, signal_id, cadence, geography_level,
             geography_id, entity_key) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uid, *definition.values()))
        db.execute("""INSERT INTO signal_records
            (row_hash, signal_uid, signal_date, observation_date, value,
             source_timestamp, ingested_at, forecast_horizon, source_kind)
            VALUES ('legacy', ?, '2024-01-01', '2024-12-31', 1,
                    '', '', '2025', 'historical_catalog')""", (uid,))
    unpublished = signal_store.demand_drugs()
    assert unpublished["drugs"] == []
    assert unpublished["status"] == "evaluated_baseline_not_available"
    with signal_store.connect() as db:
        db.execute("""INSERT INTO demand_forecast_runs
            (run_hash, source_sha256, source_year, target_year, created_at,
             method, source_url, metrics_json, row_count)
            VALUES ('legacy_run', 'old_source', 2024, 2026,
                    '2026-10-01T00:00:00+00:00',
                    'cms_partd_two_year_persistence_v1', '', '{}', 1)""")
        db.execute("""INSERT INTO demand_forecasts
            (run_hash, drug_key, demand_state, observed_claims)
            VALUES ('legacy_run', 'Drug 0', 4, 100)""")
    assert signal_store.demand_drugs()["status"] == "evaluated_baseline_not_available"
    source = tmp_path / "claims.csv.gz"
    rows = [{"year": year, "state": "arkansas", "drug_key": f"Drug {index}",
             "demand_claims": 20 * (index + 1) ** 2 + (year - 2013)}
            for year in range(2013, 2025) for index in range(10)]
    pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")
    evaluation = refresh_demand_baseline.evaluate(pd.DataFrame(rows))
    assert all(fold["validation_feature_year"] <= fold["feature_year"] - 2
               for fold in evaluation["folds"])
    assert all(fold["train_feature_year_max"] + 2 < fold["target_year"]
               for fold in evaluation["folds"])
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(source))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": list(range(2013, 2025)),
                                    "source_urls": [f"https://example.test/{year}"
                                                    for year in range(2013, 2025)]}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    result = refresh_demand_baseline.refresh_demand_baseline()
    assert result["source_year"] == 2024
    assert result["target_year"] == 2026
    assert result["rows_written"] == 10
    assert result["catalog_rows_written"] == 1
    assert refresh_demand_baseline.refresh_demand_baseline()["rows_written"] == 0
    assert {row["forecast_horizon"] for row in signal_store.values([uid])} == {"2025", "2026"}
    assert signal_store.values([uid], latest_only=True)[0]["forecast_horizon"] == "2026"
    ranking = signal_store.demand_drugs()
    assert ranking["total"] == 10
    assert ranking["filtered_total"] == 10
    assert signal_store.demand_drugs(search="Drug 1")["filtered_total"] == 1
    assert ranking["target_year"] == 2026
    assert ranking["method"] == refresh_demand_baseline.SOURCE_KIND
    assert len(ranking["state_thresholds_claims"]) == 4
    assert ranking["state_thresholds_claims"] == sorted(ranking["state_thresholds_claims"])
    assert ranking["drugs"][0]["id"] is None
    assert ranking["drugs"][0]["forecast_horizon"] == "2026"

    with signal_store.connect() as db:
        run = db.execute("SELECT run_hash, metrics_json FROM demand_forecast_runs WHERE method=?",
                         (refresh_demand_baseline.SOURCE_KIND,)).fetchone()
        old_metrics = json.loads(run["metrics_json"])
        old_metrics.pop("publication_thresholds_claims")
        db.execute("UPDATE demand_forecast_runs SET metrics_json=? WHERE run_hash=?",
                   (json.dumps(old_metrics), run["run_hash"]))
    assert signal_store.demand_drugs()["state_thresholds_claims"] is None
    assert refresh_demand_baseline.refresh_demand_baseline()["rows_written"] == 0
    assert len(signal_store.demand_drugs()["state_thresholds_claims"]) == 4

    next_year = pd.DataFrame([row for row in rows if row["year"] == 2024])
    next_year["year"] = 2025
    next_year["demand_claims"] += 1
    monkeypatch.setattr(refresh_cms_partd_source, "discover_releases",
                        lambda: {2025: "https://data.cms.gov/2025.csv"})
    monkeypatch.setattr(refresh_cms_partd_source, "revision_timestamp",
                        lambda _url: "2027-05-01T00:00:00+00:00")
    monkeypatch.setattr(refresh_cms_partd_source, "download_arkansas_year",
                        lambda _url, _year: next_year)
    assert refresh_cms_partd_source.refresh_cms_partd_source()["rows_written"] == 10
    updated = refresh_demand_baseline.refresh_demand_baseline()
    assert updated["target_year"] == 2027
    assert updated["rows_written"] == 10
    assert {row["forecast_horizon"] for row in signal_store.values([uid])} == {
        "2025", "2026", "2027"}

    committed_manifest = refresh_cms_partd_source.current_manifest_path()
    damaged = json.loads(committed_manifest.read_text())
    damaged["years"].remove(2020)
    damaged["source_urls"].pop(7)
    committed_manifest.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="source and manifest years disagree"):
        refresh_demand_baseline.refresh_demand_baseline()


def test_two_year_baseline_rejects_lfs_pointer(tmp_path):
    source = tmp_path / "claims.csv.gz"
    source.write_text("version https://git-lfs.github.com/spec/v1\n")
    try:
        refresh_demand_baseline.load_claims(source)
    except ValueError as exc:
        assert "Git LFS pointer" in str(exc)
    else:
        raise AssertionError("Git LFS pointer was accepted as source data")


def test_drug_search_treats_wildcard_characters_as_literal_text(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO demand_forecast_runs
            (run_hash, source_sha256, source_year, target_year, created_at,
             method, source_url, metrics_json, row_count)
            VALUES ('search_run', 'source', 2024, 2026, '2026-10-01T00:00:00+00:00',
                    'cms_partd_two_year_persistence_v2', '', ?, 3)""",
            (json.dumps({"fold_count": 4, "mean_balanced_accuracy": 0.8}),))
        db.executemany("""INSERT INTO demand_forecasts
            (run_hash, drug_key, demand_state, observed_claims)
            VALUES ('search_run', ?, ?, ?)""", [
                ("Drug_A", 4, 100), ("Drug B", 3, 80), ("Drug%C", 2, 60),
            ])
    underscore = signal_store.demand_drugs(search="_")
    assert underscore["filtered_total"] == 1
    assert [row["drug_name"] for row in underscore["drugs"]] == ["Drug_A"]
    assert [row["drug_name"] for row in signal_store.demand_drugs(search="%")["drugs"]] == ["Drug%C"]
    assert signal_store.demand_drugs(search="drug_a")["filtered_total"] == 1
    second = signal_store.demand_drugs(limit=1, offset=1)
    assert second["total"] == 3
    assert [row["drug_name"] for row in second["drugs"]] == ["Drug B"]
