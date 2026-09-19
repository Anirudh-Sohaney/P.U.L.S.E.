#!/usr/bin/env python3
"""Subset CEPII Gravity V202211 to 2012-2022 window -> data/gravity_window.csv.

Keeps all columns; drops rows outside focus window. Also copies Countries
reference table to data/.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
DATA = Path(__file__).resolve().parent.parent / "data"
ZIP = CACHE / "Gravity_csv_V202211.zip"
OUT = DATA / "gravity_window.csv"


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"skip  {OUT.name} (exists)")
    else:
        with zipfile.ZipFile(ZIP) as z:
            with z.open("Gravity_V202211.csv") as src, open(OUT, "w", newline="") as dst:
                rd = csv.reader(io.TextIOWrapper(src))
                wr = csv.writer(dst)
                wr.writerow(next(rd))
                n = 0
                for row in rd:
                    if 2012 <= int(row[0]) <= 2022:
                        wr.writerow(row)
                        n += 1
        print(f"got   {OUT.name} ({OUT.stat().st_size} bytes, {n} rows)")
    # Countries reference table
    ctry = DATA / "gravity_countries.csv"
    if ctry.exists():
        print(f"skip  {ctry.name} (exists)")
    else:
        with zipfile.ZipFile(ZIP) as z:
            with z.open("Countries_V202211.csv") as src, open(ctry, "wb") as dst:
                dst.write(src.read())
        print(f"got   {ctry.name} ({ctry.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())