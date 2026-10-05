"""Generate clearly synthetic sales and inventory histories for a browser demo."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


DRUGS = (("Synthetic Drug A", 3, 90),
         ("Synthetic Drug B", 5, 130),
         ("Synthetic Drug C", 2, 70))


def generate(directory: Path, *, days: int = 200, end: date | None = None) -> tuple[Path, Path]:
    if days < 130:
        raise ValueError("Browser training needs at least 130 days")
    directory.mkdir(parents=True, exist_ok=True)
    sales_path = directory / "synthetic_browser_sales.csv"
    inventory_path = directory / "synthetic_browser_inventory.csv"
    # The previous UTC day is never future-dated in a user's local browser.
    last = end or datetime.now(timezone.utc).date() - timedelta(days=1)
    first = last - timedelta(days=days - 1)
    with sales_path.open("w", newline="", encoding="utf-8") as sales_file, \
            inventory_path.open("w", newline="", encoding="utf-8") as inventory_file:
        sales_writer = csv.writer(sales_file)
        inventory_writer = csv.writer(inventory_file)
        sales_writer.writerow(("date", "drug_name", "units_sold"))
        inventory_writer.writerow(("date", "drug_name", "on_hand_units", "on_order_units"))
        for drug_index, (drug, base_sales, starting_stock) in enumerate(DRUGS):
            stock = starting_stock
            for index in range(days):
                day = first + timedelta(days=index)
                sold = base_sales + (index + drug_index) % 4
                stock = max(0, stock - sold)
                if stock < 25:
                    stock += 100
                sales_writer.writerow((day.isoformat(), drug, sold))
                inventory_writer.writerow((day.isoformat(), drug, stock, 0))
    return sales_path, inventory_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Output directory for synthetic demo CSVs")
    parser.add_argument("--days", type=int, default=200)
    args = parser.parse_args()
    for path in generate(args.directory, days=args.days):
        print(path)
