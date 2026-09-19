#!/usr/bin/env python3
"""Extract Arkansas records from NPPES monthly full-replacement V2 zip.
npidata file is ~11.6 GB uncompressed -> streamed, never fully held in memory.
Raw AR subsets saved under raw/, normalized gz under data/."""
import csv
import gzip
import io
import json
import os
import zipfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
ZIPF = os.path.join(RAW, "NPPES_Data_Dissemination_August_2026_V2.zip")
NPIDATA = "npidata_pfile_20050523-20260809.csv"
PLREF = "pl_pfile_20050523-20260809.csv"
ZIP_META = "npidata_pfile_20050523-20260809_fileheader.csv"

KEEP = [
    "NPI", "Entity Type Code",
    "Provider Organization Name (Legal Business Name)",
    "Provider Last Name (Legal Name)", "Provider First Name", "Provider Middle Name",
    "Provider Credential Text",
    "Provider First Line Business Mailing Address",
    "Provider Business Mailing Address City Name",
    "Provider Business Mailing Address State Name",
    "Provider Business Mailing Address Postal Code",
    "Provider First Line Business Practice Location Address",
    "Provider Second Line Business Practice Location Address",
    "Provider Business Practice Location Address City Name",
    "Provider Business Practice Location Address State Name",
    "Provider Business Practice Location Address Postal Code",
    "Provider Enumeration Date", "Last Update Date",
    "NPI Deactivation Reason Code", "NPI Deactivation Date", "NPI Reactivation Date",
]
KEEP += [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)]
KEEP += [f"Healthcare Provider Primary Taxonomy Switch_{i}" for i in range(1, 16)]
KEEP += [f"Provider License Number State Code_{i}" for i in range(1, 16)]

RETRIEVED = datetime.now(timezone.utc).isoformat()

def state_fields(headers):
    mail = "Provider Business Mailing Address State Name"
    pl = "Provider Business Practice Location Address State Name"
    return mail, pl

def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    meta = {}
    counts = {"arkansas_npi_records": 0, "arkansas_practice_location_ref_rows": 0}
    with zipfile.ZipFile(ZIPF) as zf:
        with zf.open(NPIDATA) as fh, open(os.path.join(RAW, "arkansas_nppes_providers.csv"), "w", newline="") as fo, gzip.open(os.path.join(DATA, "arkansas_nppes_provider_locations.csv.gz"), "wt", newline="") as fz:
            txt = io.TextIOWrapper(fh, encoding="utf-8")
            reader = csv.DictReader(txt)
            w = csv.DictWriter(fo, fieldnames=KEEP, extrasaction="ignore")
            w.writeheader()
            kz = KEEP[:]
            kz[0:0] = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes"]
            wz = csv.DictWriter(fz, fieldnames=kz, extrasaction="ignore")
            wz.writeheader()
            mail, plstate = state_fields(reader.fieldnames)
            for row in reader:
                if row.get(mail) == "AR" or row.get(plstate) == "AR":
                    w.writerow(row)
                    out = {
                        "source_id": "NPPES_full_replacement",
                        "source_url": "https://download.cms.gov/nppes/NPI_Files.html",
                        "retrieved_at_utc": RETRIEVED,
                        "extraction_notes": "Filter: mailing OR practice location state == AR",
                    }
                    out.update(row)
                    wz.writerow(out)
                    counts["arkansas_npi_records"] += 1
        with zf.open(PLREF) as fh, gzip.open(os.path.join(DATA, "arkansas_nppes_practice_location_reference.csv.gz"), "wt", newline="") as fz:
            txt = io.TextIOWrapper(fh, encoding="utf-8")
            reader = csv.DictReader(txt)
            hdr = reader.fieldnames
            wz = csv.DictWriter(fz, fieldnames=["source_id", "source_url", "retrieved_at_utc", "extraction_notes"] + list(hdr))
            wz.writeheader()
            statefield = [c for c in hdr if "State Name" in c][0]
            for row in reader:
                if row.get(statefield) == "AR":
                    out = {"source_id": "NPPES_full_replacement",
                           "source_url": "https://download.cms.gov/nppes/NPI_Files.html",
                           "retrieved_at_utc": RETRIEVED,
                           "extraction_notes": "Secondary practice location ref rows, state == AR"}
                    out.update(row)
                    wz.writerow(out)
                    counts["arkansas_practice_location_ref_rows"] += 1
        with zf.open(ZIP_META) as fh:
            meta["fileheader"] = fh.read().decode("utf-8", "replace")
    meta["retrieved_at_utc"] = RETRIEVED
    meta["source_url"] = "https://download.cms.gov/nppes/NPI_Files.html"
    meta["zip_file"] = ZIPF
    meta["counts"] = counts
    with open(os.path.join(ROOT, "source_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(counts))

if __name__ == "__main__":
    main()