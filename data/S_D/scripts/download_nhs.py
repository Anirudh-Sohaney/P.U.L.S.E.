"""Attempt NHS England prescribing data download.
openprescribing.net is behind Cloudflare; NHSBSA opendata is the official bulk
source. This script attempts the NHSBSA opendata API and records status.
Structured extraction is documented in coverage.json."""
import os, sys, urllib.request, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, ensure_dirs

SOURCE_ID = "nhs_prescribing"
SOURCE_NAME = "NHS England Prescribing Data / OpenPrescribing"
SOURCE_URL = "https://opendata.nhsbsa.net/"
OUT = os.path.join(os.path.dirname(CACHE), "data", "by_source", SOURCE_ID)


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    # Probe the NHSBSA opendata API root
    for url in ["https://opendata.nhsbsa.net/api/3/action/package_list",
                "https://opendata.nhsbsa.net/"]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            print(f"nhs {url}: HTTP {r.status}, {len(body)} bytes")
        except Exception as e:
            print(f"nhs {url}: FAILED {e}")
    print("NOTE: NHS prescribing extraction not performed in this run; "
          "see coverage.json. No NHS records written to by_source.")


if __name__ == "__main__":
    main()