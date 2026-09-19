"""Normalize drug identity across by_source records using RxNorm/RxNav where
possible. Writes drug_dictionary.json and unmapped_records.json.
Records are updated in place in their source files."""
import json, os, sys, time, re, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CACHE, BY_SOURCE, DATA, ensure_dirs, today, read_json, write_json, sha1)

RXNORM = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
SOURCES = ["medicaid_sdud", "nhs_prescribing", "cms_part_d", "dea_arcos",
           "fda_recalls", "fda_shortages"]


def lookup_rxcui(name):
    """Resolve a drug name to an RxNorm RXCUI (single best match)."""
    if not name or len(name) < 3:
        return None
    url = f"{RXNORM}?{urllib.parse.urlencode({'name': name, 'search': 1})}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        for g in data.get("idGroup", {}).get("rxnormId", [])[:1]:
            return g
    except Exception:
        pass
    return None


def parse_ingredient(pname):
    """Best-effort active-ingredient guess from a product name string."""
    if not pname:
        return None
    s = re.sub(r"\b(capsule|tablet|injection|ointment|cream|suspension|"
               r"syrup|solution|susp|vial|mg|ml|mcg|gm|gr|tab|cap|unit|" 
               r"pack|bottle|bx|kit)\b\.?", "", pname, flags=re.I)
    s = re.sub(r"[0-9][0-9.]*", "", s)
    s = re.sub(r"[^A-Za-z ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80] or None


def process_source(src):
    d = os.path.join(BY_SOURCE, src)
    if not os.path.isdir(d):
        return 0
    total = 0
    cache = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        recs = read_json(path, [])
        for r in recs:
            drug = r["drug"]
            if drug.get("mapping_status") == "mapped":
                continue
            name = drug.get("ingredient") or drug.get("canonical_name") or drug.get("original_name")
            if name not in cache:
                cache[name] = lookup_rxcui(name) if name and len(name) < 60 else None
            rxcui = cache[name]
            if rxcui:
                drug["rxnorm_rxcui"] = rxcui
                drug["ingredient"] = name.lower()
                drug["canonical_name"] = name.lower()
                drug["mapping_status"] = "mapped"
                drug["mapping_confidence"] = "low"  # name-based, not verified
                drug["identity_level"] = "ingredient"
                total += 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, separators=(",", ":"))
    return total


def build_dictionary_and_unmapped():
    dict_map = {}
    unmapped = []
    for src in SOURCES:
        d = os.path.join(BY_SOURCE, src)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            for r in read_json(os.path.join(d, fn), []):
                drug = r["drug"]
                key = drug.get("original_name") or drug.get("canonical_name") or "?"
                if drug.get("mapping_status") == "mapped":
                    dict_map.setdefault(key, {
                        "original_name": drug["original_name"],
                        "canonical_name": drug.get("canonical_name"),
                        "ingredient": drug.get("ingredient"),
                        "brand_name": drug.get("brand_name"),
                        "rxnorm_rxcui": drug.get("rxnorm_rxcui"),
                        "atc_code": drug.get("atc_code"),
                        "ndc": drug.get("ndc"),
                    })
                else:
                    unmapped.append({
                        "source_id": src,
                        "original_name": drug["original_name"],
                        "mapping_status": drug.get("mapping_status"),
                        "mapping_confidence": drug.get("mapping_confidence"),
                        "mapping_notes": "No reliable free mapping found; record preserved.",
                    })
    write_json(os.path.join(DATA, "drug_dictionary.json"), list(dict_map.values()))
    write_json(os.path.join(DATA, "unmapped_records.json"), unmapped)
    print(f"drug_dictionary: {len(dict_map)} entries; unmapped: {len(unmapped)}")


def main():
    ensure_dirs()
    mapped = 0
    for src in SOURCES:
        n = process_source(src)
        print(f"{src}: mapped {n}")
        mapped += n
    build_dictionary_and_unmapped()
    print(f"total newly mapped via RxNorm: {mapped}")


if __name__ == "__main__":
    main()