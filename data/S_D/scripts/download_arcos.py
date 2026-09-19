"""Download DEA ARCOS retail drug summary reports (official PDFs) into cache.
The official ARCOS retail drug summary reports are published as PDFs. This
script downloads them and records status. Structured table parsing of the PDFs
is limited; coverage is documented in coverage.json."""
import os, sys, urllib.request, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, ensure_dirs

SOURCE_ID = "dea_arcos"
SOURCE_NAME = "DEA ARCOS Controlled-Substance Distribution"
SOURCE_URL = "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html"
BASE = "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/"
YEARS = list(range(2006, 2023))
OUT = os.path.join(os.path.dirname(CACHE), "data", "by_source", SOURCE_ID)


def main():
    ensure_dirs()
    os.makedirs(OUT, exist_ok=True)
    ok, fail = [], []
    for y in YEARS:
        url = f"{BASE}report_yr_{y}.pdf"
        dest = os.path.join(CACHE, f"arcos_{y}.pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            ok.append(y)
            continue
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 1000 or not data[:5].startswith(b"%PDF"):
                raise ValueError("not a PDF")
            with open(dest, "wb") as f:
                f.write(data)
            ok.append(y)
            print(f"arcos {y}: downloaded {len(data)} bytes")
        except Exception as e:
            fail.append(y)
            print(f"arcos {y}: FAILED {e}")
    print(f"arcos: ok={ok} fail={fail}")
    print("NOTE: ARCOS official data is PDF; structured extraction not performed. "
          "See coverage.json. No ARCOS records written to by_source.")


if __name__ == "__main__":
    main()