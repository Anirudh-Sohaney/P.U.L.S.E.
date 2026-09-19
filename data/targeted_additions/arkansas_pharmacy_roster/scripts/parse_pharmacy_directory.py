#!/usr/bin/env python3
"""Parse Arkansas State Board of Pharmacy 'Pharmacy Directory' PDFs into
facility rows. Text per page pre-extracted into raw/txt/{year}.txt."""
import os
import re
import json
import gzip
import csv
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
YRS = ["2008", "2009", "2010", "2011", "2012", "2013", "2016",
       "2018", "2019", "2020", "2021", "2022"]

CITY_RE = re.compile(r"^([A-Za-z .'\-()]+?)\s+(AR)\s*,\s*(\d{5})$")
FAC_RE = re.compile(r"^([A-Z]{2,3})\d{4,6}$")
EMP_RE = re.compile(r"^([A-Z]{2,3})\d{4,6}\**$")
FAC_PREFIX = {"AR", "HP", "SM", "IP"}
EMP_PREFIX = {"PD", "PT", "PN", "PI"}
PHONE_RE = re.compile(r"\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}")
ADDR_RE = re.compile(r"(Street|St\b|Ave\.?|Avenue|Suite|Ste|Hwy|Highway|Drive|Dr\.?|Box|Ln\.?|Lane|Rd\.?|Road|Blvd|Parkway|Pkwy|Ct\.?|Court|Way|Loop|Circle|Cir\.?|Route|Rte|Bldg|Building|Unit|Trail|Trl|Plaza|Mailbox|Campus)", re.I)

def is_addr(l):
    if re.match(r"^[\d#]", l):
        return True
    if re.match(r"^(P\.?\s?O\.?\s*|PO\s)?(Box|Drawer)\b", l, re.I):
        return True
    if l.endswith(",") and ADDR_RE.search(l):
        return True
    return False

HEAD_SKIP = ("Arkansas State Board", "Page:", "Date:", "** Denotes", "Arkansas ")

def detect_section(text):
    m = re.search(r"Arkansas\s+([A-Za-z]+)\s+Pharmacies", text)
    if m:
        w = m.group(1).lower()
        if w in ("retail", "hospital", "institutional", "specialty"):
            return w
    if "PHARMACIE" in text:
        pi = text.find("PHARMACIE")
        for kw in ("RETAIL", "HOSPITAL", "INSTITUTIONAL", "SPECIALTY"):
            k = text.find(kw)
            if 0 <= k < pi:
                return kw.lower()
    return None

def parse_year(yr):
    rows = []
    pages = open(os.path.join(RAW, "txt", yr + ".txt")).read().split("@@PAGE")
    section = None
    for page_i, page in enumerate(pages[1:], start=1):
        lines = [l.strip() for l in page.splitlines()]
        text = "\n".join(lines)
        s = detect_section(text)
        if s:
            section = s
        city = None
        i = 0
        n = len(lines)
        cur = None
        unmatched = None
        while i < n:
            line = lines[i]
            if not line or line.startswith(HEAD_SKIP):
                i += 1
                continue
            m = CITY_RE.match(line)
            if m:
                city = {"city": m.group(1).strip(), "state": "AR", "zip": m.group(3)}
                i += 1
                continue
            fm = FAC_RE.match(line)
            if fm and fm.group(1) in FAC_PREFIX:
                if cur:
                    rows.append(cur)
                # walk back: address lines, phone lines, then name lines
                j = i - 1
                addr = []
                phone = ""
                while j >= 0:
                    lj = lines[j]
                    if not lj or lj.startswith(HEAD_SKIP) or CITY_RE.match(lj):
                        break
                    if FAC_RE.match(lj) or EMP_RE.match(lj) or "Pharmacies and Employees" in lj:
                        break
                    pm = PHONE_RE.search(lj)
                    if pm and pm.start() == 0 and len(lj) <= len(pm.group(0)) + 3:
                        phone = pm.group(0)
                        j -= 1
                        continue
                    if is_addr(lj):
                        if pm and not phone:
                            phone = pm.group(0)
                        addr.insert(0, lj)
                        j -= 1
                    else:
                        break
                k = j
                name_lines = []
                while k >= 0:
                    lk = lines[k]
                    if not lk or lk.startswith(HEAD_SKIP) or CITY_RE.match(lk):
                        break
                    if FAC_RE.match(lk) or EMP_RE.match(lk) or "Pharmacies and Employees" in lk:
                        break
                    pm = PHONE_RE.search(lk)
                    if pm and pm.start() == 0 and len(lk) <= len(pm.group(0)) + 3:
                        if not phone:
                            phone = pm.group(0)
                        k -= 1
                        continue
                    name_lines.insert(0, lk)
                    k -= 1
                name = " ".join(name_lines)
                joined_addr = ", ".join(addr)
                pm = PHONE_RE.search(joined_addr)
                if pm:
                    if not phone:
                        phone = pm.group(0)
                    joined_addr = joined_addr.replace(pm.group(0), "").rstrip(" ,")
                # phone might be its own captured line in name-buffer already consumed; fine
                cur = {
                    "year": yr, "section": section or "",
                    "license_number": line, "facility_name": name.rstrip(","),
                    "address": joined_addr.rstrip(","),
                    "city": city["city"] if city else "",
                    "state": city["state"] if city else "AR",
                    "zip": city["zip"] if city else "",
                    "phone": phone, "pharmacist_in_charge": "",
                    "source_pdf": f"arkansas_pharmacy_directory_{yr}.pdf",
                    "source_page": page_i,
                }
                i += 1
                continue
            if cur is None:
                i += 1
                continue
            if PHONE_RE.fullmatch(line) or "phone" in line or "fax" in line or \
               (PHONE_RE.match(line) and "fax" not in line):
                if "fax" not in line and not cur["phone"]:
                    pm = PHONE_RE.search(line)
                    if pm:
                        cur["phone"] = pm.group(0)
                i += 1
                continue
            em = EMP_RE.match(line)
            if em and em.group(1) in EMP_PREFIX:
                if unmatched and line.endswith("*") and not cur["pharmacist_in_charge"]:
                    cur["pharmacist_in_charge"] = unmatched
                unmatched = None
            elif (" " in line or line.isalpha()) and not PHONE_RE.search(line):
                unmatched = line
            i += 1
        if cur:
            rows.append(cur)
    return rows

def main():
    allrows = []
    per_year = {}
    for yr in YRS:
        r = parse_year(yr)
        per_year[yr] = len(r)
        allrows.extend(r)
        print(yr, len(r))
    fields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes"] + \
        ["year", "section", "license_number", "facility_name", "address", "city", "state",
         "zip", "phone", "pharmacist_in_charge", "source_pdf", "source_page"]
    retrieved = datetime.now(timezone.utc).isoformat()
    with gzip.open(os.path.join(DATA, "arkansas_pharmacy_facilities.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in allrows:
            out = {"source_id": "ASBP_Pharmacy_Directory",
                   "source_url": "https://cdm16039.contentdm.oclc.org/digital/collection/p266101coll7",
                   "retrieved_at_utc": retrieved,
                   "extraction_notes": "Parsed from official ASBP Pharmacy Directory PDFs (pymupdf text extraction + heuristics)"}
            out.update(r)
            w.writerow(out)
    json.dump({"per_year_rows": per_year, "total": len(allrows),
               "years_available": YRS,
               "note": "2007 PDF is a scanned personal pharmacist directory (no OCR), excluded."},
              open(os.path.join(ROOT, "source_manifest.json"), "w"), indent=2)
    print("TOTAL", len(allrows))

if __name__ == "__main__":
    main()