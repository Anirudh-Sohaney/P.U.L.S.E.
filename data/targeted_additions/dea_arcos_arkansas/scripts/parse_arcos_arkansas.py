#!/usr/bin/env python3
"""Parse DEA ARCOS Retail Drug Summary Report 01 (by ZIP code within state)
PDFs, keep only Arkansas state blocks. Handles two PDF layouts:
  classic (2006-2016): 'DRUG CODE:1100DRUG NAME:AMPHETAMINE' + STATE: blocks
  new (2018-2025): 'DRUG: 1100 - AMPHETAMINE' + 'STATE: AR - ARKANSAS' blocks
Data columns are quarterly grams (Q1-Q4) and total grams per registrant ZIP."""
import os
import re
import json
import gzip
import csv
import pymupdf
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

NUM = re.compile(r"^\$?[\d,]+(?:\.\d{2})?$")
ZIP = re.compile(r"^\d{3}$")

def page_text_new(doc, i):
    blocks = doc[i].get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
    return "\n".join(b[4] for b in blocks)

def parse(pdf_path, year):
    doc = pymupdf.open(pdf_path)
    classic = False
    for i in range(min(6, doc.page_count)):
        if "DRUG CODE:" in doc[i].get_text():
            classic = True
            break
    rows = []
    drug_code = drug_name = None
    in_ar = False
    buf = []
    cur_section = None  # 'report01' or other (we only want report 01)

    def flush():
        nonlocal buf
        if in_ar and drug_code and len(buf) == 6 and ZIP.match(str(buf[0])) and all(NUM.match(str(x)) for x in buf[1:]):
            rows.append({"year": year, "drug_code": drug_code, "drug_name": drug_name,
                         "zip3": buf[0], "quarter1_grams": buf[1], "quarter2_grams": buf[2],
                         "quarter3_grams": buf[3], "quarter4_grams": buf[4], "total_grams": buf[5]})
        buf = []

    for i in range(doc.page_count):
        text = page_text_new(doc, i) if not classic else doc[i].get_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        k = 0
        while k < len(lines):
            ln = lines[k]
            if cur_section == "report01":
                m = re.search(r"DRUG\s*CODE\s*:\s*(\d+)\s*[A-Z]?\s*DRUG\s*NAME\s*:\s*([A-Z0-9/\\(\\) .,&\-]+)", ln)
                if not m:
                    m = re.search(r"DRUG\s*:\s*(\d+)\s*-\s*([A-Z0-9/\\(\\) .,&\-]+)", ln)
                if m:
                    drug_code, drug_name = m.group(1), m.group(2).strip()
                    in_ar = False
                    flush()
                    k += 1
                    continue
                st = None
                m = re.match(r"^STATE\s*:\s*([A-Z]{2})\s*-\s*([A-Z ]+)$", ln)
                if m:
                    st = m.group(2).strip().upper()
                else:
                    m = re.match(r"^STATE\s*:\s*([A-Z ]+)$", ln)
                    if m:
                        st = m.group(1).strip().upper()
                    elif ln.strip() == "STATE:" or ln.strip().startswith("STATE: "):
                        nxt = k + 1
                        while nxt < len(lines) and not lines[nxt].strip():
                            nxt += 1
                        if nxt < len(lines):
                            st = lines[nxt].strip().upper()
                            k = nxt
                if st is not None:
                    in_ar = st in ("ARKANSAS", "AR")
                    flush()
                    k += 1
                    continue
                if re.search(r"- Total\s*$", ln) and not ln.startswith("ARKANSAS") and "AR - ARKANSAS" not in ln:
                    flush()
                    if "AK - ALASKA" in ln or "AR - ARKANSAS" in ln:
                        pass
                    k += 1
                    continue
                if "REGISTRANT ZIP" in ln or "ZIP CODE" in ln:
                    flush()
                    k += 1
                    continue
                if in_ar and drug_code:
                    if ZIP.match(ln):
                        flush()
                        buf = [ln]
                    elif NUM.match(ln) and buf:
                        buf.append(ln)
                    elif "ARKANSAS - Total" in ln or "AR - ARKANSAS - Total" in ln or "ARKANSAS" in ln:
                        flush()
                    if len(buf) == 6:
                        flush()
            else:
                if "REPORT 01" in ln or "REPORT 1" in ln or ("RETAIL DRUG" in ln and "ZIP CODE" in ln):
                    cur_section = "report01"
            k += 1
    doc.close()
    return rows

def main():
    files = {}
    for y in range(2006, 2016):
        files[str(y)] = os.path.join(RAW, f"rpt1_{y}.pdf")
    for y in range(2016, 2026):
        files[str(y)] = os.path.join(RAW, f"report_yr_{y}.pdf")
    allrows = []
    per_year = {}
    for y, path in files.items():
        if not os.path.exists(path):
            continue
        r = parse(path, y)
        per_year[y] = len(r)
        allrows.extend(r)
        print(y, len(r), flush=True)
    fields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes", "year",
              "drug_code", "drug_name", "zip3", "quarter1_grams", "quarter2_grams",
              "quarter3_grams", "quarter4_grams", "total_grams"]
    with gzip.open(os.path.join(DATA, "arcos_arkansas_retail_summary.csv.gz"), "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        ret = datetime.now(timezone.utc).isoformat()
        for r in allrows:
            out = {"source_id": "DEA_ARCOS_retail_drug_summary_report1",
                   "source_url": "https://www.deadiversion.usdoj.gov/arcos/retail_drug_summary/arcos-drug-summary-reports.html",
                   "retrieved_at_utc": ret,
                   "extraction_notes": "Report 01 (retail drug distribution by ZIP within state, grams), Arkansas blocks only; quarterly"}
            out.update(r)
            w.writerow(out)
    json.dump({"per_year_rows": per_year, "total": len(allrows),
               "note": "2011 Report 1 PDF returns an error page (not published on DEA site)."},
              open(os.path.join(ROOT, "source_manifest.json"), "w"), indent=2)
    print("TOTAL", len(allrows))

if __name__ == "__main__":
    main()