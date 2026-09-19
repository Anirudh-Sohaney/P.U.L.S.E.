#!/usr/bin/env python3
"""Download needed HTML files from 3DLNews2 HTML collection via HTTPS."""
import os, subprocess, concurrent.futures

BASE = "https://g-953a4d.746abe.8540.data.globus.org/1-Google/1-Newspaper"
HTMLDIR = "/home/ubuntu/meditrack_proj/medi_track/data/raw/html"
needed = [l.strip() for l in open("/home/ubuntu/meditrack_proj/medi_track/data/raw/needed_html.txt") if l.strip()]

def fetch(rel):
    dest = os.path.join(HTMLDIR, rel)  # rel like HTML/AR/2000/md5.html.gz
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return rel, "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{BASE}/{rel}"
    r = subprocess.run(["curl", "-s", "-o", dest, "--max-time", "120", url], capture_output=True)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return rel, f"ok({size})" if r.returncode == 0 and size > 0 else f"FAIL rc={r.returncode}"

os.makedirs(HTMLDIR, exist_ok=True)
ok = fail = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(fetch, rel): rel for rel in needed}
    for f in concurrent.futures.as_completed(futs):
        rel, status = f.result()
        if status.startswith("ok") or status == "skip": ok += 1
        else: fail += 1
        if fail and fail % 50 == 0: print(f"failures so far: {fail}", flush=True)
print(f"DONE ok={ok} fail={fail}")
