from backend import refresh_bls, signal_store
from backend.config import settings


def test_bls_derived_formulas_match_catalog_observation_order():
    months = [(day, float(value), "published") for day, value in [
        ("2025-01-31", 1), ("2025-02-28", 2),
        ("2025-04-30", 3), ("2025-05-31", 4),
        ("2025-06-30", 5),
    ]]
    derived = refresh_bls.derive_months("example", months)
    assert derived["example_lag_4"][-1][1] == 1
    assert derived["example_change_4"][-1][1] == 4
    assert derived["example_rolling_mean_4"][-1][1] == 2.5
    assert abs(derived["example_rolling_std_4"][-1][1] - 1.11803398875) < 1e-10


def test_bls_seasonal_baseline_uses_same_month_across_all_prior_years():
    months = [
        ("2020-01-31", 2.0, "published"),
        ("2021-01-31", 4.0, "published"),
        ("2022-01-31", 6.0, "published"),
        ("2023-01-31", 10.0, "published"),
        ("2023-02-28", 100.0, "published"),
        ("2024-01-31", 12.0, "preliminary"),
    ]
    derived = refresh_bls.derive_months("example", months)
    assert derived["example_seasonal_baseline"][-1] == (
        "2024-01-31", 5.5, "preliminary")
    assert derived["example_seasonal_anomaly"][-1] == (
        "2024-01-31", 6.5, "preliminary")
    assert not any(day == "2023-02-28" for day, _, _ in
                   derived["example_seasonal_baseline"])


def test_bls_refresh_records_only_published_months_and_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        for base in refresh_bls.BLS_SERIES:
            for name in refresh_bls.derive_months(base, [("2025-01-31", 4.0, "published")]):
                definition = {"signal_origin": "model_external_state_feature",
                              "signal_id": name, "cadence": "monthly",
                              "geography_level": "national", "geography_id": "US",
                              "entity_key": ""}
                db.execute("""INSERT INTO signal_definitions
                    (id, signal_origin, signal_id, cadence, geography_level,
                     geography_id, entity_key)
                     VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (signal_store.signal_uid(definition), *definition.values()))

    revision = {"value": "4.1"}

    def fake_fetch(_series_id):
        return [
            {"year": "2025", "period": "M13", "value": "999"},
            {"year": "2025", "period": "M01", "value": "4.0"},
            {"year": "2025", "period": "M02", "value": revision["value"],
             "footnotes": [{"code": "P"}]},
        ]

    monkeypatch.setattr(refresh_bls, "fetch_series", fake_fetch)
    assert refresh_bls.refresh_bls()["rows_written"] == 12
    assert refresh_bls.refresh_bls()["rows_written"] == 0
    revision["value"] = "4.2"
    assert refresh_bls.refresh_bls()["rows_written"] == 6

    with signal_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM signal_records").fetchone()[0] == 18
        ids = [row[0] for row in db.execute("SELECT id FROM signal_definitions WHERE signal_id IN ('arkansas_unemployment_rate', 'consumer_price_index_all_items', 'national_unemployment_rate')")]
    latest = signal_store.values(ids, latest_only=True)
    assert {row["value"] for row in latest} == {4.2}
    assert all(row["observation_date"] == "2025-02-28" for row in latest)
    assert all(row["data_quality"] == "preliminary;publication_time_unknown"
               for row in latest)
