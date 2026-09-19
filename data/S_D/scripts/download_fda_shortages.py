"""Download FDA drug shortage data from openFDA, write yearly JSON.
Covers 2012-2022. Shortage status/event observations, not quantities."""
import json, os, sys, time, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, BY_SOURCE, ensure_dirs, today, norm_ndc, empty_geo,
                    empty_drug, make_record, read_json)

SOURCE_ID = "fda_shortages"
SOURCE_NAME = "FDA Drug Shortages"
SOURCE_URL = "https://open.fda.gov/apis/drug/drugshortages/"
API = "https://api.fda.gov/drug/shortages.json"
OUT = os.path.join(BY_SOURCE, SOURCE_ID)
CACHEF = os.path.join(CACHE, "fda_shortages.json")
YEAR_MIN, YEAR_MAX = 2012, 2022


def fetch_all():
    if os.path.exists(CACHEF):
        print("cache exists, skipping download")
        return read_json(CACHEF)
    recs, skip, limit = [], 0, 1000
    while True:
        url = f"{API}?{urllib.parse.urlencode({'limit': limit, 'skip': skip})}"
        data = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        page = data.get("results", [])
        total = data["meta"]["results"]["total"]
        recs.extend(page)
        skip += len(page)
        print(f"  fetched {skip}/{total}")
        if not page or skip >= total:
            break
    with open(CACHEF, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    return recs


def parse_date(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m/%d/%Y %H:%M"):
        try:
            from datetime import datetime
            return datetime.strptime(s[:16].strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def build(records):
    by_year = {y: [] for y in range(YEAR_MIN, YEAR_MAX + 1)}
    for r in records:
        pid = parse_date(r.get("initial_posting_date") or r.get("last_update_date"))
        if not pid or not (YEAR_MIN <= int(pid[:4]) <= YEAR_MAX):
            continue
        year = int(pid[:4])
        name = r.get("generic_name") or r.get("openfda", {}).get("generic_name", [""])[0] or "Unknown"
        of = r.get("openfda", {})
        brand = of.get("brand_name", [None])[0]
        ndc = norm_ndc(r.get("package_ndc") or (of.get("product_ndc") or [None])[0])
        drug = empty_drug(name)
        drug["ingredient"] = name.split()[0].lower() if name else None
        drug["canonical_name"] = name.lower()
        drug["brand_name"] = brand
        drug["ndc"] = ndc
        drug["manufacturer"] = of.get("manufacturer_name", [None])[0]
        drug["route"] = (of.get("route") or [None])[0]
        if of.get("rxcui"):
            drug["rxnorm_rxcui"] = of["rxcui"][0]
        drug["identity_level"] = "ingredient"
        drug["mapping_status"] = "mapped"
        drug["mapping_confidence"] = "medium"
        geo = {"country": "US", "admin1": None, "admin2": None,
               "postal_code": None, "name": "National", "code": None,
               "level": "national"}
        status = r.get("status")
        rec = make_record(
            record_id=f"fda_shortages:{r.get('id') or sha(id(r))}",
            period_start=pid, period_end=pid, granularity="event",
            geo=geo, drug=drug,
            obs_type="supply_event", metric="shortage_active",
            value=(1 if (status and "resolved" not in status.lower()) else 0),
            unit="status", semantics=f"drug shortage status observation: {status}",
            is_quantity=False, is_monetary=False,
            source_id=SOURCE_ID, source_name=SOURCE_NAME, source_url=SOURCE_URL,
            source_record_id=r.get("id"),
            native_frequency="event", published_at=parse_date(r.get("last_update_date")),
            event_date=pid,
            quality_notes="active shortage represented as event/status observation; not a missing-unit count",
            is_suppressed=False)
        by_year[year].append(rec)
    return by_year


def sha(x):
    from common import sha1
    return sha1(str(x))


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    records = fetch_all()
    by_year = build(records)
    for y, recs in by_year.items():
        json.dump(recs, open(os.path.join(OUT, f"{y}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"fda_shortages: {sum(len(v) for v in by_year.values())} records")


if __name__ == "__main__":
    main()
