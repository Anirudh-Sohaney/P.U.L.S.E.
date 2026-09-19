import csv
import gzip
import io
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

OUT = {
    "fluview": "data/cdc_fluview_ar_national_weekly.csv.gz",
    "nndss": "data/cdc_nndss_ar_national_weekly.csv.gz",
    "wastewater_ar": "data/cdc_wastewater_ar_site_weekly.csv.gz",
    "wastewater_nat": "data/cdc_wastewater_national_weekly.csv.gz",
}
RETRIEVED = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url, params=None, raw_q=None):
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def paginate(url, params, page_size=50000):
    offset = 0
    all_rows = []
    while True:
        p = dict(params, **{"$limit": page_size, "$offset": offset})
        rows = get(url, p)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
        time.sleep(0.3)
    return all_rows


def save(df, name, notes, source_id, source_url):
    df = df.copy()
    df["source_id"] = source_id
    df["source_url"] = source_url
    df["retrieved_at_utc"] = RETRIEVED
    df["extraction_notes"] = notes
    with gzip.open(name, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    print(f"{name}: {len(df)} rows")


# 7a Delphi FluView (ILI%)
data = get("https://delphi.cmu.edu/epidata/api.php",
           params={"source": "fluview", "regions": "ar,nat", "epiweeks": "201201-202653"})
with open("raw/delphi_fluview_ar_nat.json", "w") as f:
    json.dump(data, f)
df = pd.DataFrame(data["epidata"])
df.to_csv("raw/delphi_fluview_ar_nat.csv", index=False)
save(df, OUT["fluview"],
     "ILI outpatient surveillance, latest published issue per epiweek; ar=Arkansas, nat=US national. epiweeks 201201-latest.",
     "delphi-fluview", "https://delphi.cmu.edu/epidata/api.php?source=fluview")

# 7b NNDSS weekly notifiable disease (Arkansas + US RESIDENTS)
nndss = paginate("https://data.cdc.gov/resource/x9gk-5huc.json",
                 {"$where": "year>='2022' and (states='Arkansas' or states='US RESIDENTS')", "$order": "year,week"})
with open("raw/nndss_weekly_ar_nat.json", "w") as f:
    json.dump(nndss, f)
df = pd.DataFrame(nndss)
df.to_csv("raw/nndss_weekly_ar_nat.csv", index=False)
save(df, OUT["nndss"],
     "Provisional weekly notifiable disease cases (m1-m4 monthly counts) for Arkansas and US RESIDENTS. Dataset covers 2022-present; earlier years not available in this source.",
     "cdc-nndss-weekly", "https://data.cdc.gov/dataset/NNDSS-Weekly-Data/x9gk-5huc")

# 7c CDC wastewater WVAL (site-level), keep AR rows + derive national weekly mean
ww = paginate("https://data.cdc.gov/resource/atcp-73re.json", {"$limit": 50000, "$order": "week_end"})
with open("raw/cdc_wastewater_wval_all.json", "w") as f:
    json.dump(ww, f)
df = pd.DataFrame(ww)
df.to_csv("raw/cdc_wastewater_wval_all.csv", index=False)
ar = df[df["state_territory"] == "Arkansas"].copy()
save(ar, OUT["wastewater_ar"],
     "Site-level weekly WVAL (site_wval 0-10, site_wval_category) for Arkansas wastewater sites, per pathogen target.",
     "cdc-wastewater-wval", "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re")

df["site_wval"] = pd.to_numeric(df["site_wval"], errors="coerce")
nat = (df.dropna(subset=["site_wval"])
         .groupby(["week_end", "pathogen_target"], as_index=False)["site_wval"]
         .mean())
nat["n_sites"] = (df.dropna(subset=["site_wval"])
                    .groupby(["week_end", "pathogen_target"])["site"]
                    .nunique().reset_index(drop=True))
nat = nat.rename(columns={"site_wval": "national_wval_mean"})
save(nat, OUT["wastewater_nat"],
     "Derived national weekly WVAL: mean site_wval across all reporting US wastewater sites per week_end per pathogen target (not an official CDC national series).",
     "cdc-wastewater-wval", "https://data.cdc.gov/dataset/CDC-Wastewater-Viral-Activity-Level-for-SARS-CoV-2-Influenza-A-and-RSV/atcp-73re")
