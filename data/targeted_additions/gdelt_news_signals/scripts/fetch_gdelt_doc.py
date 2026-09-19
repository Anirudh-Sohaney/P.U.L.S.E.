import json
import time
from datetime import datetime

import pandas as pd
import requests

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "meditrack-data/1.0 (research)"
QUERIES = {
    "arkansas_pharmacy": '"Arkansas pharmacy"',
    "arkansas_pharmacist": '"Arkansas pharmacist"',
    "arkansas_opioid": '"Arkansas opioid"',
    "arkansas_hospital": '"Arkansas hospital"',
    "arkansas_medicaid": '"Arkansas Medicaid"',
    "arkansas_pharmaceutical": '"Arkansas pharmaceutical"',
    "arkansas_drug_shortage": '"Arkansas drug shortage"',
    "arkansas_prescription_drug": '"Arkansas prescription drug"',
}
START = "20170101000000"
END = datetime.now().strftime("%Y%m%d%H%M%S")


def call(params, retries=8):
    for i in range(retries):
        r = requests.get(BASE, params=params, headers={"User-Agent": USER_AGENT}, timeout=120)
        if r.status_code == 200:
            return r
        msg = r.text[:200]
        time.sleep(30 * (i + 1))
    raise RuntimeError(f"GDELT failed for {params}: {msg}")


def month_counts(query, key):
    r = call({"query": query, "mode": "timelinevol", "format": "json",
              "startdatetime": START, "enddatetime": END, "datacolumn": "c"})
    raw = r.json()
    with open(f"raw/timeline_{key}.json", "w") as f:
        json.dump(raw, f)
    series = (raw.get("timeline") or [{}])[0]
    tl = series.get("data") or []
    rows = []
    for pt in tl:
        rows.append((str(pt["date"])[:6], int(pt["value"])))
    return rows


def article_samples(query, key):
    years = range(2017, 2027)
    out = []
    for y in years:
        r = call({"query": query, "mode": "artlist", "format": "json",
                  "startdatetime": f"{y}0101000000", "enddatetime": f"{y}1231235959",
                  "maxrecords": "250"})
        arts = r.json().get("articles", [])
        with open(f"raw/artlist_{key}_{y}.json", "w") as f:
            json.dump(arts, f)
        for a in arts:
            a["sample_year"] = y
            out.append(a)
        time.sleep(60)
    return out


count_rows = []
sample_rows = []
for key, q in QUERIES.items():
    print(f"== {key} ==", flush=True)
    try:
        mc = month_counts(q, key)
        for date, n in mc:
            count_rows.append({"query": key, "query_text": q, "month": date, "article_count": n})
        print("  timeline rows:", len(mc), flush=True)
        time.sleep(60)
    except Exception as e:
        print("  COUNT FAILED:", e, flush=True)
        count_rows.append({"query": key, "query_text": q, "month": None, "article_count": None,
                           "error": str(e)})
        time.sleep(120)
    try:
        samples = article_samples(q, key)
        for a in samples:
            a["query"] = key
            a["query_text"] = q
        sample_rows.extend(samples)
        print("  sample articles:", len(samples), flush=True)
    except Exception as e:
        print("  SAMPLES FAILED:", e, flush=True)

pd.DataFrame(count_rows).to_csv("data/gdelt_ar_health_monthly_counts.csv.gz", index=False)
pd.DataFrame(sample_rows).to_csv("data/gdelt_ar_health_article_samples.csv.gz", index=False)
print("DONE", len(count_rows), len(sample_rows))