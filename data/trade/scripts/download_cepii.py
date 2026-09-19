#!/usr/bin/env python3
"""Download CEPII BACI + Gravity trade data into cache/.

BACI: reconciled bilateral HS6 trade (value + quantity), primary trade edges.
Gravity: static country-pair edge features (distance, RTA, colonial ties).

Re-runnable: skips files already present unless --force.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "cache"

# name -> (url, why)
WANT = {
    "BACI_HS92_V202601.zip": (
        "https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS92_V202601.zip",
        "BACI HS92 revision, 1995-2022, reconciled bilateral HS6 value+quantity (primary)",
    ),
    "Gravity_csv_V202211.zip": (
        "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip",
        "CEPII Gravity V202211, country-pair static edge features",
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
            with urllib.request.urlopen(req, timeout=3600) as r, open(dest, "wb") as out:
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