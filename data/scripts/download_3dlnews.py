#!/usr/bin/env python3
"""Download 3DLNews2 preprocessed per-state files for years 2000-2001 via HTTPS."""
import os, subprocess, concurrent.futures

BASE = "https://g-e840e1.746abe.8540.data.globus.org/1-Google/1-Newspaper/preprocessed_state"
RAW = "/home/ubuntu/meditrack_proj/medi_track/data/raw"
STATES = ["AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]
YEARS = list(range(2013, 2023))

def fetch(state, year):
    fname = f"preprocessed_newspaper_articles_{state}_{year}.jsonl.gz"
    dest = os.path.join(RAW, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return fname, "skip"
    url = f"{BASE}/{state}/{fname}"
    r = subprocess.run(["curl", "-s", "-o", dest, "--max-time", "300", url], capture_output=True)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return fname, f"ok({size})" if r.returncode == 0 and size > 0 else f"FAIL rc={r.returncode}"

os.makedirs(RAW, exist_ok=True)
jobs = [(s, y) for s in STATES for y in YEARS]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, s, y): (s, y) for s, y in jobs}
    for f in concurrent.futures.as_completed(futs):
        fname, status = f.result()
        print(f"{fname}: {status}", flush=True)
print("DONE")
