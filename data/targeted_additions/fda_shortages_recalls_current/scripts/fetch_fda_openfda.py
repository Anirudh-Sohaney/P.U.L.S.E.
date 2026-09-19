#!/usr/bin/env python3
"""Fetch openFDA drug shortages (all records) and drug enforcement (2023-01-01..today).
Pagination with limit=1000 and skip; stops when meta.results < limit or error."""
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
os.makedirs(RAW, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

BASE = "https://api.fda.gov/drug/{}.json"
LIMIT = 1000
UA = {"User-Agent": "meditrack-data/1.0"}

def fetch_all(endpoint, query=None):
    skip = 0
    rows = []
    while True:
        params = {"limit": LIMIT, "skip": skip}
        if query:
            params["search"] = query
        url = BASE.format(endpoint) + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        results = data.get("results", [])
        rows.extend(results)
        got = data.get("meta", {}).get("results", {}).get("total", None)
        if len(results) < LIMIT:
            break
        skip += LIMIT
        time.sleep(0.4)
    return rows, got

def main():
    retrieved = datetime.now(timezone.utc).isoformat()
    out = {}

    short_path = os.path.join(RAW, "drug_shortages.json")
    if os.path.exists(short_path):
        short = json.load(open(short_path))
        total = None
        print("shortages: loaded from raw (", len(short), ")")
    else:
        short, total = fetch_all("shortages")
        with open(short_path, "w") as f:
            json.dump(short, f, indent=1)
    out["shortages"] = {"endpoint": BASE.format("shortages"), "total_reported": total,
                        "fetched": len(short)}

    enf_query = "report_date:[20230101 TO 20261231]"
    enf, total = fetch_all("enforcement", enf_query)
    out["enforcement"] = {"endpoint": BASE.format("enforcement"),
                          "search": enf_query, "total_reported": total, "fetched": len(enf)}
    with open(os.path.join(RAW, "drug_enforcement.json"), "w") as f:
        json.dump(enf, f, indent=1)

    def flatten(rec, prefix_keep=True):
        r = dict(rec)
        r.pop("openfda", None)
        return r

    import csv
    sfields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes"] + sorted(
        {k for r in short for k in r.keys() if k != "openfda"})
    with open(os.path.join(DATA, "fda_shortages_current.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
        w.writeheader()
        for r in short:
            row = {"source_id": "FDA_drug_shortages",
                   "source_url": BASE.format("shortages"),
                   "retrieved_at_utc": retrieved,
                   "extraction_notes": "All current shortage records from openFDA drug/shortages"}
            row.update(flatten(r))
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})

    efields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes"] + sorted(
        {k for r in enf for k in r.keys() if k != "openfda"})
    with open(os.path.join(DATA, "fda_enforcement_2023_current.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=efields, extrasaction="ignore")
        w.writeheader()
        for r in enf:
            row = {"source_id": "FDA_drug_enforcement",
                   "source_url": BASE.format("enforcement") + "?" + enf_query,
                   "retrieved_at_utc": retrieved,
                   "extraction_notes": "report_date 2023-01-01..retrieval date"}
            row.update(flatten(r))
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})

    out["retrieved_at_utc"] = retrieved
    with open(os.path.join(ROOT, "source_manifest.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))

if __name__ == "__main__":
    main()