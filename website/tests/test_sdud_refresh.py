from backend import refresh_sdud, signal_store
from backend.config import settings


CURRENT_RELEASE = refresh_sdud.SourceRelease(
    2026, "2957a7f9-9a15-453e-9afd-3bbdcbac8fd3", "2026-07-10T21:08:05+00:00")


def _row(ndc, product, count, *, suppressed=False, year="2026", quarter="1", kind="FFSU"):
    return {"state": "AR", "year": year, "quarter": quarter,
            "utilization_type": kind, "ndc": ndc, "product_name": product,
            "number_of_prescriptions": None if suppressed else str(count),
            "suppression_used": str(suppressed).lower()}


def test_sdud_rejects_duplicate_and_does_not_impute_suppressed_counts():
    first = _row("00000000001", "Drug A", 12)
    suppressed = _row("00000000002", "Drug A", None, suppressed=True)
    assert refresh_sdud.normalize([first, suppressed], source_year=2026)[1][-2:] == (None, 1)
    try:
        refresh_sdud.normalize([first, first], source_year=2026)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate SDUD row was accepted")


def test_sdud_catalog_selects_latest_unique_year_and_ignores_future_release():
    catalog = [
        {"title": "State Drug Utilization Data 2025",
         "identifier": "158a1baa-5506-400a-8ec3-97756f0b0536",
         "modified": "2026-07-13T15:28:32+00:00"},
        {"title": "State Drug Utilization Data 2026",
         "identifier": CURRENT_RELEASE.dataset_id,
         "modified": CURRENT_RELEASE.modified_at},
        {"title": "State Drug Utilization Data 2027",
         "identifier": "11111111-1111-4111-8111-111111111111",
         "modified": "2026-09-01T00:00:00+00:00"},
    ]
    assert refresh_sdud.select_latest_release(catalog, today_year=2026) == CURRENT_RELEASE
    assert refresh_sdud.select_latest_release(catalog, today_year=2027).year == 2027
    next_year_row = _row("00000000001", "Drug A", 12, year="2027")
    assert refresh_sdud.normalize([next_year_row], source_year=2027)[0][0] == 2027
    try:
        refresh_sdud.normalize([next_year_row], source_year=2026)
    except ValueError as exc:
        assert "unexpected year" in str(exc)
    else:
        raise AssertionError("Mismatched source year was accepted")
    try:
        refresh_sdud.select_latest_release(catalog + [catalog[1]], today_year=2026)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Duplicate annual release was accepted")


def test_sdud_refresh_preserves_revisions_and_exposes_partial_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    state = {"count": 12}

    def fetch(release):
        assert release == CURRENT_RELEASE
        return [_row("00000000001", "Drug A", state["count"]),
                _row("00000000002", "Drug A", None, suppressed=True),
                _row("00000000003", "Drug B", 5)]

    monkeypatch.setattr(refresh_sdud, "fetch_rows", fetch)
    monkeypatch.setattr(refresh_sdud, "discover_latest_release", lambda: CURRENT_RELEASE)
    assert refresh_sdud.refresh_sdud()["rows_written"] == 3
    assert refresh_sdud.refresh_sdud()["rows_written"] == 0
    state["count"] = 14
    assert refresh_sdud.refresh_sdud()["rows_written"] == 3
    with signal_store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sdud_snapshots").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM sdud_rows").fetchone()[0] == 6
    latest = signal_store.latest_sdud_snapshot(search="Drug A")
    assert latest["period"] == "2026-Q1"
    assert latest["source_modified_at"] == CURRENT_RELEASE.modified_at
    assert latest["source_rows"] == 3
    assert latest["suppressed_rows"] == 1
    assert latest["reported_prescriptions_lower_bound"] == 19
    assert latest["products"][0]["reported_prescriptions_lower_bound"] == 14
    assert latest["products"][0]["suppressed_rows"] == 1
    assert latest["filtered_total"] == 1
    assert signal_store.latest_sdud_snapshot(search="%")["filtered_total"] == 0
    assert signal_store.latest_sdud_snapshot(search="drug a")["filtered_total"] == 1
