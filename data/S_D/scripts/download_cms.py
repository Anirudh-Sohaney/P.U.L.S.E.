"""Download CMS Medicare Part D Spending by Drug (national) via data.cms.gov API.
Covers 2020-2022 within the inclusive range. National-level demand data."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, BY_SOURCE, ensure_dirs, today, norm_ndc, empty_geo,
                    empty_drug, make_record, read_json, sha1)

SOURCE_ID = "cms_part_d"
SOURCE_NAME = "CMS Medicare Part D Spending by Drug"
SOURCE_URL = "https://data.cms.gov/data-api/v1/dataset/7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b/data"
DATASET = "7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b"
OUT = os.path.join(BY_SOURCE, SOURCE_ID)
CACHEF = os.path.join(CACHE, "cms_part_d.json")
YEARS = [2020, 2021, 2022]


def fetch_all():
    if os.path.exists(CACHEF):
        print("cache exists, skipping download")
        return read_json(CACHEF)
    rows, offset, size = [], 0, 10000
    while True:
        url = f"https://data.cms.gov/data-api/v1/dataset/{DATASET}/data?size={size}&offset={offset}"
        data = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        if not data:
            break
        rows.extend(data)
        offset += len(data)
        print(f"  fetched {offset}")
        if len(data) < size:
            break
    with open(CACHEF, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    return rows


def num(v):
    if v in (None, "", "N/A"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(rows):
    by_year = {y: [] for y in YEARS}
    seen = set()
    for r in rows:
        brand = r.get("Brnd_Name")
        gnrc = r.get("Gnrc_Name")
        mftr = (r.get("Mftr_Name") or "Overall").strip()
        name = brand or gnrc or "Unknown"
        key = (name, mftr)
        if key in seen:
            continue  # dedupe identical source rows
        seen.add(key)
        drug = empty_drug(name)
        drug["canonical_name"] = (gnrc or brand or "").lower() or None
        drug["ingredient"] = (gnrc or "").lower() or None
        drug["brand_name"] = brand
        drug["identity_level"] = "ingredient" if gnrc else "unmapped"
        drug["mapping_status"] = "mapped" if gnrc else "unmapped"
        drug["mapping_confidence"] = "medium" if gnrc else "none"
        geo = {"country": "US", "admin1": None, "admin2": None,
               "postal_code": None, "name": "United States", "code": "US",
               "level": "national"}
        for y in YEARS:
            claims = num(r.get(f"Tot_Clms_{y}"))
            fills = num(r.get(f"Tot_Dsg_Unts_{y}"))
            cost = num(r.get(f"Tot_Spndng_{y}"))
            benes = num(r.get(f"Tot_Benes_{y}"))
            if claims is None and fills is None and cost is None and benes is None:
                continue
            start = f"{y}-01-01"; end = f"{y}-12-31"
            rid = f"cms_part_d:{y}:{sha1(name)[:12]}:{sha1(mftr)[:8]}"
            for metric, val, unit, sem, isq, ism in [
                ("part_d_claims", claims, "claims", "Medicare Part D claims count", True, False),
                ("thirty_day_fills", fills, "30-day fills", "Medicare Part D 30-day equivalent fills", True, False),
                ("part_d_total_drug_cost", cost, "USD", "Medicare Part D total drug cost", False, True),
                ("part_d_beneficiaries", benes, "beneficiaries", "Medicare Part D beneficiaries", True, False),
            ]:
                if val is None:
                    continue
                by_year[y].append(make_record(
                    record_id=f"{rid}:{metric}", period_start=start, period_end=end,
                    granularity="year", geo=geo, drug=drug,
                    obs_type="demand", metric=metric, value=val, unit=unit,
                    semantics=sem, is_quantity=isq, is_monetary=ism,
                    source_id=SOURCE_ID, source_name=SOURCE_NAME, source_url=SOURCE_URL,
                    source_record_id=f"{y}:{name}:{mftr}", native_frequency="year",
                    published_at=None, event_date=None))
    return by_year


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    rows = fetch_all()
    by_year = build(rows)
    for y, recs in by_year.items():
        json.dump(recs, open(os.path.join(OUT, f"{y}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"cms_part_d: {sum(len(v) for v in by_year.values())} records from {len(rows)} rows")


if __name__ == "__main__":
    main()