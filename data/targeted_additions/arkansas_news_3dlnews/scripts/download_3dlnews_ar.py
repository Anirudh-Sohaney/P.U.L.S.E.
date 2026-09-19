#!/usr/bin/env python3
"""Download 3DLNews2 preprocessed Arkansas newspaper article files for 2013-2022."""
import os
import subprocess
import concurrent.futures

BASE = "https://g-e840e1.746abe.8540.data.globus.org/1-Google/1-Newspaper/preprocessed_state/AR"
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
YEARS = list(range(2013, 2023))


def fetch(year):
    fname = f"preprocessed_newspaper_articles_AR_{year}.jsonl.gz"
    dest = os.path.join(RAW, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return fname, "skip"
    r = subprocess.run(
        ["curl", "-s", "-o", dest, "--max-time", "300", f"{BASE}/{fname}"],
        capture_output=True,
    )
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return fname, f"ok({size})" if r.returncode == 0 and size > 0 else f"FAIL rc={r.returncode}"


os.makedirs(RAW, exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for f in concurrent.futures.as_completed([ex.submit(fetch, y) for y in YEARS]):
        fname, status = f.result()
        print(f"{fname}: {status}", flush=True)
print("DONE")
