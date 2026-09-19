#!/usr/bin/env python3
"""Normalize OFAC SDN XML + Consolidated CSV into flat entity-level CSVs.

Outputs:
  data/ofac_sdn.csv          one row per SDN entity (lists pipe-joined)
  data/ofac_consolidated.csv consolidated non-SDN list (source-native flat)
"""
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA = ROOT / "data"

NS = "{https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML}"


def text(el, tag):
    e = el.find(f"{NS}{tag}")
    return e.text.strip() if e is not None and e.text else ""


def texts(el, tag):
    return [e.text.strip() for e in el.findall(f"{NS}{tag}") if e.text]


def subtexts(el, list_tag, item_tag, field):
    out = []
    for lst in el.findall(f"{NS}{list_tag}"):
        for item in lst.findall(f"{NS}{item_tag}"):
            v = text(item, field)
            if v:
                out.append(v)
    return out


def parse_sdn(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for e in root.findall(f"{NS}sdnEntry"):
        name = " ".join(
            x for x in (text(e, "firstName"), text(e, "lastName")) if x
        )
        rows.append(
            {
                "uid": text(e, "uid"),
                "name": name,
                "sdn_type": text(e, "sdnType"),
                "programs": "|".join(
                    p.text.strip()
                    for p in e.findall(f"{NS}programList/{NS}program")
                    if p.text
                ),
                "akas": "|".join(subtexts(e, "akaList", "aka", "lastName")),
                "addresses": "|".join(
                    ", ".join(
                        x
                        for x in (
                            text(a, "address1"),
                            text(a, "city"),
                            text(a, "stateOrProvince"),
                            text(a, "postalCode"),
                            text(a, "country"),
                        )
                        if x
                    )
                    for a in e.findall(f"{NS}addressList/{NS}address")
                ),
                "ids": "|".join(
                    f"{text(i,'idType')}:{text(i,'idNumber')}"
                    for i in e.findall(f"{NS}idList/{NS}id")
                ),
                "nationalities": "|".join(
                    subtexts(e, "nationalityList", "nationality", "country")
                ),
                "citizenships": "|".join(
                    subtexts(e, "citizenshipList", "citizenship", "country")
                ),
                "dob": "|".join(
                    subtexts(e, "dateOfBirthList", "dateOfBirthItem", "dateOfBirth")
                ),
                "pob": "|".join(
                    subtexts(e, "placeOfBirthList", "placeOfBirthItem", "placeOfBirth")
                ),
                "remarks": text(e, "remarks"),
            }
        )
    return rows


def main() -> int:
    DATA.mkdir(exist_ok=True)

    rows = parse_sdn(CACHE / "SDN.XML")
    root = ET.parse(CACHE / "SDN.XML").getroot()
    declared = int(
        root.find(f"{NS}publshInformation/{NS}Record_Count").text
    )
    assert len(rows) == declared, f"count mismatch: {len(rows)} != {declared}"
    with open(DATA / "ofac_sdn.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"ofac_sdn.csv: {len(rows)} entities")

    # Consolidated list: source-native flat CSV, keep as-is
    import shutil

    shutil.copy(CACHE / "CONS_PRIM.CSV", DATA / "ofac_consolidated.csv")
    n = sum(1 for _ in open(DATA / "ofac_consolidated.csv", encoding="utf-8")) - 1
    print(f"ofac_consolidated.csv: {n} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())