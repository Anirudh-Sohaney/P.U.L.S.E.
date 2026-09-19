"""Validate the deterministic synthetic pharmacy benchmark artifact."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "arkansas_clinic_daily_pharmacy_sales.csv"
REQUIRED = {
    "date", "state", "facility_type", "drug_name", "therapeutic_class",
    "units_sold", "unit_price_usd", "revenue_usd", "stockout_flag",
    "injected_event_tags",
}


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED - fields
        if missing:
            raise AssertionError(f"missing columns: {sorted(missing)}")
        rows = list(reader)

    expected_dates = (date(2023, 1, 1), date(2025, 12, 31))
    dates = [date.fromisoformat(row["date"]) for row in rows]
    drugs = {row["drug_name"] for row in rows}
    if len(rows) != 32_880:
        raise AssertionError(f"unexpected row count: {len(rows)}")
    if (min(dates), max(dates)) != expected_dates:
        raise AssertionError(f"unexpected date range: {min(dates)} to {max(dates)}")
    if len(drugs) != 30:
        raise AssertionError(f"unexpected drug count: {len(drugs)}")
    if any(int(row["units_sold"]) < 0 for row in rows):
        raise AssertionError("negative units_sold")
    if any(float(row["unit_price_usd"]) < 0 for row in rows):
        raise AssertionError("negative unit_price_usd")
    for row in rows:
        revenue = round(int(row["units_sold"]) * float(row["unit_price_usd"]), 2)
        if abs(revenue - float(row["revenue_usd"])) > 0.011:
            raise AssertionError(f"revenue mismatch on {row['date']} / {row['drug_name']}")
    if not any(row["injected_event_tags"] for row in rows):
        raise AssertionError("no event tags found")
    print(f"validated {len(rows):,} rows, {len(drugs)} drugs, "
          f"{min(dates)} through {max(dates)}")


if __name__ == "__main__":
    main()
