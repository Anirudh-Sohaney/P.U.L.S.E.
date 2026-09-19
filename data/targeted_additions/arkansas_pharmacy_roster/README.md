# Arkansas State Board of Pharmacy — Pharmacy Directory (facility roster)

## Source
- Official name: Arkansas State Board of Pharmacy (ASBP), Arkansas Department of Health — "Pharmacy Directory" / "Arkansas Pharmacies and Employees"
- Official page: https://healthy.arkansas.gov/boards-commissions/boards/pharmacy-arkansas-state-board/
- License verification tool (public, no login): https://arbopharmv7prod.glsuite.us/glsuiteweb/clients/arbopharm/public/verification/search.aspx — **blocked by an Azure WAF JS challenge at retrieval time; not used.**
- Official directory PDFs (published by ASBP, hosted in the State of Arkansas digital collection on OCLC CONTENTdm): https://cdm16039.contentdm.oclc.org/digital/collection/p266101coll7 — compound object id 43114, years **2007–2022**.

## Retrieval
- Retrieved at (UTC): 2026-08-12
- Directory years downloaded: 2008, 2009, 2010, 2011, 2012, 2013, 2016, 2018, 2019, 2020, 2021, 2022 (13 PDFs). 2014, 2015, 2017 not published on contentdm; 2023+ directory not yet posted publicly.
- Download URL pattern: `https://cdm16039.contentdm.oclc.org/digital/api/collection/p266101coll7/id/{pointer}/download` (pointers listed in `source_manifest.json`).
- 2007 PDF (id 43105) is a scanned personal *pharmacist* directory ("Arkansas Licensed Pharmacists", 323 image pages) with no text layer; OCR was not performed, so it is **excluded**.
- Parsing: `scripts/parse_pharmacy_directory.py` — pymupdf text extraction per page, then a line-based state machine keyed on facility license lines (`AR` retail, `HP` hospital, `IP` institutional, `SM` specialty) with employee-license lines (`PD`/`PT`/`PN`/`PI`) for pharmacist-in-charge detection (`**` markers).

## Files
- `raw/arkansas_pharmacy_directory_{year}.pdf` — official PDFs.
- `raw/txt/{year}.txt` — extracted page text.
- `data/arkansas_pharmacy_facilities.csv.gz` — **11,716 facility rows** (2008–2022).

## Fields kept
year, section (retail/hospital/institutional/specialty), license_number, facility_name, address, city, state, zip, phone, pharmacist_in_charge, source_pdf, source_page, plus `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Rows by year
2008: 954 · 2009: 947 · 2010: 925 · 2011: 956 · 2012: 973 · 2013: 959 · 2016: 1,008 · 2018: 1,015 · 2019: 1,002 · 2020: 993 · 2021: 990 · 2022: 994 — **Total: 11,716** (includes retail, hospital, institutional, and specialty facilities).

## License / access notes
- Public government records. The board's verification search is the current authoritative roster; the directory PDFs are the official published facility/employee lists.
- No credentials used; nothing behind login was accessed.

## Known limitations
- Live verification search was inaccessible (Azure WAF JS challenge); directory PDFs used instead (published through 2022).
- No 2014, 2015, 2017, or 2023+ directory was posted publicly at retrieval time.
- 2007 directory is a scanned personal-pharmacist roster; excluded (no OCR).
- Heuristic PDF parsing: ~2.4% of rows lack a facility name (multi-line names split mid-block); ~1.3% lack address; phone captured when printed. Pharmacist-in-charge only available where the PDF marked `**`.
- In some years (2009–2011) phone number was printed on the same line as the street address; the parser separates them when possible, otherwise the address carries the phone digits (minor truncation possible, e.g. "P.O. Bo(479)632-2011").
- ZIP/city derived from the page's city header line; facility rows assume the city header preceding them.