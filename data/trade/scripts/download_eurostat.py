#!/usr/bin/env python3
"""Download Eurostat Comext trade data (EU side, CN8) into cache/.

ext_lt_intratrd: intra-EU trade (EU27 reporter x partner, monthly, CN8)
ext_lt_intertrd: extra-EU trade (EU27 reporter x non-EU partner, monthly, CN8)

Full JSON per dataset is small (~1-2MB). Re-runnable: skips unless --force.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

WANT = {
    "comext_intra_eu.json": (
        f"{BASE}/ext_lt_intratrd?format=JSON&lang=EN",
        "Eurostat Comext intra-EU trade, CN8, monthly",
    ),
    "comext_extra_eu.json": (
        f"{BASE}/ext_lt_intertrd?format=JSON&lang=EN",
        "Eurostat Comext extra-EU trade, CN8, monthly",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, why) in WANT.items():
        dest = CACHE / name
        if dest.exists() and not args.force:
            print(f"skip  {name} (exists)")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as out:
                out.write(r.read())
            print(f"got   {name} ({dest.stat().st_size} bytes)")
        manifest[name] = {"url": url, "size": dest.stat().st_size, "why": why}

    (CACHE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print("manifest written to", CACHE / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())