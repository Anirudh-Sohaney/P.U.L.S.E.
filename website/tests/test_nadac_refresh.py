from backend import refresh_nadac, signal_store
from backend.config import settings


def _row(ndc: str, price: str, unit: str, classification: str = "B") -> dict:
    return {"ndc": ndc, "nadac_per_unit": price, "pricing_unit": unit,
            "classification_for_rate_setting": classification,
            "as_of_date": "2026-09-30"}


def test_nadac_snapshot_keeps_units_separate_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    rows = [_row("001", "2.00", "EA"), _row("002", "4.00", "EA"),
            _row("003", "1.00", "ML")]
    monkeypatch.setattr(refresh_nadac, "fetch_latest_snapshot",
                        lambda: ("2026-09-30", rows, refresh_nadac._url(as_of_date="2026-09-30")))
    assert refresh_nadac.refresh_nadac()["snapshots_written"] == 1
    assert refresh_nadac.refresh_nadac()["snapshots_written"] == 0
    snapshot = signal_store.latest_nadac_snapshot()
    assert snapshot["total_rows"] == 3
    assert [(group["pricing_unit"], group["row_count"], group["nadac_per_unit_mean"])
            for group in snapshot["groups"]] == [("EA", 2, "3.00"), ("ML", 1, "1.00")]
    rows[0] = _row("001", "3.00", "EA")
    assert refresh_nadac.refresh_nadac()["snapshots_written"] == 1
    assert signal_store.latest_nadac_snapshot()["groups"][0]["nadac_per_unit_mean"] == "3.50"


def test_nadac_rejects_mixed_source_dates():
    row = _row("001", "2.00", "EA")
    row["as_of_date"] = "2026-09-23"
    try:
        refresh_nadac.aggregate_snapshot([row], "2026-09-30")
    except ValueError as exc:
        assert "unexpected snapshot date" in str(exc)
    else:
        raise AssertionError("Mixed NADAC source dates were accepted")
