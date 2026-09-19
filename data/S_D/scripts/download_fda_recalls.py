"""Download FDA drug enforcement (recall) data from openFDA, write yearly JSON.
Covers 2004-2022. Supply-disruption events, not sales quantities."""
import json, os, sys, time, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, BY_SOURCE, ensure_dirs, today, norm_ndc, empty_geo,
                    empty_drug, make_record, read_json, sha1)

SOURCE_ID = "fda_recalls"
SOURCE_NAME = "FDA Drug Enforcement / Recall Data"
SOURCE_URL = "https://open.fda.gov/apis/drug/enforcement/"
API = "https://api.fda.gov/drug/enforcement.json"
OUT = os.path.join(BY_SOURCE, SOURCE_ID)
CACHEF = os.path.join(CACHE, "fda_recalls.json")
YEAR_MIN, YEAR_MAX = 2004, 2022


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
                with urllib.request.urlopen(url, timeout=90) as r:
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
    # keep only drug recalls with a description and an initiation date
    keep = [r for r in recs
            if r.get("product_type") == "Drugs"
            and r.get("product_description")
            and r.get("recall_initiation_date")]
    with open(CACHEF, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False)
    return keep


def parse_date(s):
    """openFDA enforcement dates are YYYYMMDD."""
    if not s:
        return None
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        y, m, d = s[:4], s[4:6], s[6:8]
        return f"{y}-{m}-{d}"
    if len(s) == 4 and s.isdigit():
        return f"{s}-01-01"
    return None


def to_number(s):
    """Extract leading numeric from strings like '1 vial' or '250 units'."""
    import re
    if s is None:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", str(s))
    return float(m.group(1)) if m else None


def build(records):
    by_year = {y: [] for y in range(YEAR_MIN, YEAR_MAX + 1)}
    for r in records:
        pid = parse_date(r.get("recall_initiation_date"))
        if not pid or not (YEAR_MIN <= int(pid[:4]) <= YEAR_MAX):
            continue
        year = int(pid[:4])
        name = r.get("product_description") or "Unknown"
        ndc = None
        for f in ("product_ndc", "ndc"):
            if r.get("openfda", {}).get(f):
                ndc = norm_ndc(r["openfda"][f][0])
                break
        geo = empty_geo()
        st = (r.get("state") or "").strip()
        if st:
            geo = {"country": "US", "admin1": None, "admin2": None,
                   "postal_code": r.get("postal_code"), "name": st,
                   "code": st.upper(), "level": "state"}
        elif r.get("country"):
            geo = {"country": "US", "admin1": None, "admin2": None,
                   "postal_code": None, "name": "Nationwide",
                   "code": None, "level": "national"}
        drug = empty_drug(name)
        drug["ndc"] = ndc
        drug["manufacturer"] = r.get("recalling_firm")
        value = to_number(r.get("product_quantity"))
        unit = None
        if value is not None:
            import re
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/ ]+)", str(r["product_quantity"]))
            unit = (m.group(2).strip() if m else "units")
        rec = make_record(
            record_id=f"fda_recalls:{r.get('event_id')}:{(r.get('recall_number') or 'x')}",
            period_start=pid, period_end=pid, granularity="event",
            geo=geo, drug=drug,
            obs_type="supply_event", metric="recall_event",
            value=value, unit=(unit or "event"), semantics="drug-specific recall event; quantity is recalled product count when reported",
            is_quantity=(value is not None), is_monetary=False,
            source_id=SOURCE_ID, source_name=SOURCE_NAME, source_url=SOURCE_URL,
            source_record_id=r.get("recall_number") or r.get("event_id"),
            native_frequency="event", published_at=parse_date(r.get("report_date")),
            event_date=pid,
            quality_notes="supply-disruption event; no quantity means none reported",
            is_suppressed=False)
        by_year[year].append(rec)
    return by_year


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    records = fetch_all()
    by_year = build(records)
    for y, recs in by_year.items():
        json.dump(recs, open(os.path.join(OUT, f"{y}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"fda_recalls: {sum(len(v) for v in by_year.values())} records, {len(records)} source rows")


if __name__ == "__main__":
    main()
