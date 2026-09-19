#!/usr/bin/env python3
"""Extract BACI HS92 2012-2022 yearly CSVs from cache zip into data/baci/.

BACI rows: t,i,j,k,v,q = year, exporter ISO3num, importer ISO3num, HS6 code,
value (1000 USD), quantity (tonnes). Focus window 2012-2022 only.
"""
import sys
import zipfile
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
DATA = Path(__file__).resolve().parent.parent / "data" / "baci"
ZIP = CACHE / "BACI_HS92_V202601.zip"
YEARS = range(2012, 2023)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        for y in YEARS:
            name = f"BACI_HS92_Y{y}_V202601.csv"
            out = DATA / f"baci_hs92_{y}.csv"
            if out.exists():
                print(f"skip  {name} (exists)")
                continue
            with z.open(name) as src, open(out, "wb") as dst:
                dst.write(src.read())
            print(f"got   {name} -> {out.name} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())