#!/usr/bin/env python3
"""Stream CMS Part D Prescribers - by Provider and Drug full CSVs, keep only
Arkansas rows (Prscrbr_State_Abrvtn == 'AR'). Full files are multi-GB; full
originals are not retained. Raw subset saved under raw/, normalized copy under data/."""
import csv
import gzip
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")

# year -> official downloadURL from data.cms.gov/data.json (retrieved this run)
with open("/tmp/opencode/partd_urls.txt") as f:
    URLS = {}
    for line in f:
        yr, url = line.strip().split("|")
        URLS[yr.split("-")[0]] = url.strip()

KEEP = [
    "Prscrbr_NPI", "Prscrbr_Last_Org_Name", "Prscrbr_First_Name",
    "Prscrbr_City", "Prscrbr_State_Abrvtn", "Prscrbr_Type",
    "Brnd_Name", "Gnrc_Name", "Tot_Clms", "Tot_30day_Fills",
    "Tot_Drug_Cst", "Tot_Benes",
]

def stream_filter(url, year, w_raw, w_norm):
    req = urllib.request.Request(url, headers={"User-Agent": "meditrack-data/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, io.TextIOWrapper(
        resp, encoding="utf-8-sig", errors="replace"
    ) as txt:
        reader = csv.DictReader(txt)
        missing = [c for c in KEEP if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"{year}: missing cols {missing}; fields={reader.fieldnames}")
        n = 0
        for row in reader:
            if row.get("Prscrbr_State_Abrvtn") == "AR":
                out = {c: row.get(c) for c in KEEP}
                out["year"] = year
                w_raw.writerow(out)
                w_norm.writerow(out)
                n += 1
        return n

def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    norm_path = os.path.join(DATA, "arkansas_partd_provider_drug_by_year.csv.gz")
    total = 0
    year_counts = {}
    with gzip.open(norm_path, "wt", newline="", encoding="utf-8") as fz:
        w_norm = csv.DictWriter(fz, fieldnames=KEEP + ["year"])
        w_norm.writeheader()
        for year in sorted(URLS):
            raw_path = os.path.join(RAW, f"arkansas_partd_{year}.csv")
            if os.path.exists(os.path.join(RAW, f".done_{year}")):
                with open(raw_path) as fr, open(raw_path) as f2:
                    w_norm.writerows(csv.DictReader(fr))
                    n = sum(1 for _ in f2) - 1
                total += n
                print(year, "skip", n, flush=True)
                continue
            with open(raw_path, "w", newline="", encoding="utf-8") as fr:
                w_raw = csv.DictWriter(fr, fieldnames=KEEP + ["year"])
                w_raw.writeheader()
                n = stream_filter(URLS[year], year, w_raw, w_norm)
            open(os.path.join(RAW, f".done_{year}"), "w").close()
            year_counts[year] = n
            total += n
            print(year, n, flush=True)
    manifest = {
        "source_name": "Medicare Part D Prescribers - by Provider and Drug (CMS)",
        "source_url": "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(URLS),
        "row_counts_by_year": year_counts,
        "total_arkansas_rows": total,
        "filter": "Prscrbr_State_Abrvtn == 'AR' (streamed full CSV, original multi-GB files not retained)",
        "note": "Dataset has no ZIP field; city/state only.",
    }
    with open(os.path.join(ROOT, "source_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("TOTAL", total)

if __name__ == "__main__":
    main()
