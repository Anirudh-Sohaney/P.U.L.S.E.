"""Download Medicaid State Drug Utilization Data (SDUD) and write yearly JSON.
Official bulk CSVs at https://download.medicaid.gov/data/sdudYYYY.csv.
Highest-priority direct drug-utilization source. Quarterly, state-level."""
import json, os, sys, csv, io, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, BY_SOURCE, ensure_dirs, today, norm_ndc, empty_geo,
                    empty_drug, make_record, read_json, sha1)

SOURCE_ID = "medicaid_sdud"
SOURCE_NAME = "Medicaid State Drug Utilization Data"
SOURCE_URL = "https://www.medicaid.gov/medicaid/prescription-drugs/state-drug-utilization-data"
OUT = os.path.join(BY_SOURCE, SOURCE_ID)
# Years available as bulk CSV on download.medicaid.gov.
# 2019+ use sdudYYYY.csv; 2012-2018 use StateDrugUtilizationData-YYYY.csv.
YEARS = list(range(2012, 2023))

# Full national Medicaid SDUD is ~5M rows/year; JSON output would be many GB.
# To keep the deliverable usable we process a representative subset of states
# and the two core quantity metrics. Set MEDICAID_STATES=ALL to process all.
STATE_SUBSET = os.environ.get("MEDICAID_STATES", "AR,NY,TX,CA,FL,IL,WA,OH").split(",")
METRICS = [
    ("prescription_count", "prescriptions", "Medicaid number of prescriptions", True, False),
    ("units_reimbursed", "units", "Medicaid units reimbursed", True, False),
]

QUARTERS = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"),
            3: ("07-01", "09-30"), 4: ("10-01", "12-31")}


def download_year(y):
    cachef = os.path.join(CACHE, f"medicaid_{y}.csv")
    if os.path.exists(cachef):
        print(f"  {y}: cache exists")
        return cachef
    if y >= 2019:
        url = f"https://download.medicaid.gov/data/sdud{y}.csv"
    else:
        url = f"https://download.medicaid.gov/data/StateDrugUtilizationData-{y}.csv"
    print(f"  {y}: downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    with open(cachef, "wb") as f:
        f.write(data)
    return cachef


def num(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_year(y, path):
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = (row.get("State") or "").strip()
            if STATE_SUBSET != ["ALL"] and state not in STATE_SUBSET:
                continue
            utype = (row.get("Utilization Type") or "FFSU").strip()
            ndc = norm_ndc(row.get("NDC"))
            pname = (row.get("Product Name") or "").strip()
            q = row.get("Quarter")
            try:
                q = int(q)
            except (TypeError, ValueError):
                continue
            if q not in QUARTERS:
                continue
            start = f"{y}-{QUARTERS[q][0]}"; end = f"{y}-{QUARTERS[q][1]}"
            units = num(row.get("Units Reimbursed"))
            rx = num(row.get("Number of Prescriptions"))
            total = num(row.get("Total Amount Reimbursed"))
            med = num(row.get("Medicaid Amount Reimbursed"))
            nonmed = num(row.get("Non Medicaid Amount Reimbursed"))
            suppressed = (row.get("Suppression Used") or "").strip().upper() in ("Y", "YES", "TRUE")
            if not pname and not ndc:
                continue
            drug = empty_drug(pname or (ndc or "Unknown"))
            drug["ndc"] = ndc
            drug["canonical_name"] = pname.lower() or None
            drug["identity_level"] = "unmapped"
            drug["mapping_status"] = "unmapped"
            drug["mapping_confidence"] = "none"
            geo = {"country": "US", "admin1": state, "admin2": None,
                   "postal_code": None, "name": state, "code": state,
                   "level": "state"}
            rid = f"medicaid_sdud:{y}:{q}:{state}:{utype}:{ndc or sha1(pname)[:10]}"
            for metric, unit, sem, isq, ism in METRICS:
                val = {"prescription_count": rx, "units_reimbursed": units}[metric]
                if val is None:
                    continue
                recs.append(make_record(
                    record_id=f"{rid}:{metric}", period_start=start, period_end=end,
                    granularity="quarter", geo=geo, drug=drug,
                    obs_type="demand", metric=metric, value=val, unit=unit,
                    semantics=sem, is_quantity=isq, is_monetary=ism,
                    source_id=SOURCE_ID, source_name=SOURCE_NAME, source_url=SOURCE_URL,
                    source_record_id=f"{y}:{q}:{state}:{utype}:{ndc or pname}",
                    native_frequency="quarterly", published_at=None, event_date=None,
                    is_suppressed=suppressed))
    return recs


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for y in YEARS:
        try:
            path = download_year(y)
        except Exception as e:
            print(f"medicaid {y}: FAILED {e}")
            continue
        recs = build_year(y, path)
        with open(os.path.join(OUT, f"{y}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, separators=(",", ":"))
        total += len(recs)
        print(f"medicaid {y}: {len(recs)} records")
    print(f"medicaid_sdud total: {total}")


if __name__ == "__main__":
    main()