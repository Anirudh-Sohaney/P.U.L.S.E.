#!/usr/bin/env python3
"""Download OFAC SDN + Consolidated sanctions files.

Resolves current file URLs from the OFAC Sanctions List Service API, then
downloads the selected formats into cache/. Re-runnable: skips files already
present unless --force.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

API_HOST = "https://sanctionslistservice.ofac.treas.gov"
CACHE = Path(__file__).resolve().parent.parent / "cache"

# fileName -> (list_endpoint, why we keep it)
WANT = {
    "SDN.CSV": ("SdnList", "entity-level SDN, flat CSV"),
    "SDN.XML": ("SdnList", "SDN with startDate/addresses/ids (dated)"),
    "CONS_PRIM.CSV": ("ConsolidatedList", "consolidated non-SDN primary list"),
    "CONSOLIDATED.XML": ("ConsolidatedList", "consolidated full XML"),
}


def get_exports(endpoint: str) -> list:
    req = urllib.request.Request(
        f"{API_HOST}/api/PublicationPreview/{endpoint}",
        data=b"{}",
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    exports = {}
    for endpoint in ("SdnList", "ConsolidatedList"):
        for f in get_exports(endpoint):
            exports[f["fileName"]] = f

    manifest = {}
    for name, (endpoint, why) in WANT.items():
        f = exports.get(name)
        if not f:
            print(f"WARN: {name} not in {endpoint} exports", file=sys.stderr)
            continue
        # Frontend builds download URL as {host}/api/PublicationPreview/exports/{fileName};
        # that endpoint 302-redirects to a presigned S3 URL.
        full = f"{API_HOST}/api/PublicationPreview/exports/{name}"
        dest = CACHE / name
        if dest.exists() and not args.force:
            print(f"skip  {name} (exists)")
        else:
            req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as out:
                out.write(r.read())
            print(f"got   {name} ({dest.stat().st_size} bytes)")
        manifest[name] = {
            "url": full,
            "lastUpdated": f["lastUpdated"],
            "size": dest.stat().st_size,
            "why": why,
        }

    (CACHE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print("manifest written to", CACHE / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())