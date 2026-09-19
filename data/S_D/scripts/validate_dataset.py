"""Validate the S_D dataset. Writes validation_errors.json and prints a summary."""
import json, os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BY_SOURCE, COMBINED, DATA, ensure_dirs, read_json, write_json

SOURCES = ["medicaid_sdud", "nhs_prescribing", "cms_part_d", "dea_arcos",
           "fda_recalls", "fda_shortages"]
YEAR_MIN, YEAR_MAX = 2000, 2022
DATE_MIN = datetime.date(2000, 1, 1)
DATE_MAX = datetime.date(2022, 12, 31)


def parse_date(s):
    try:
        return datetime.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def validate_file(path, errors, per_year=None):
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        errors.append({"file": path, "error": f"invalid JSON: {e}"})
        return 0
    if not isinstance(data, list):
        errors.append({"file": path, "error": "not a JSON array"})
        return 0
    ids = set()
    for i, r in enumerate(data):
        if not isinstance(r, dict):
            errors.append({"file": path, "index": i, "error": "record not object"})
            continue
        rid = r.get("record_id")
        if not rid:
            errors.append({"file": path, "index": i, "error": "missing record_id"})
        elif rid in ids:
            errors.append({"file": path, "index": i, "error": f"duplicate record_id {rid}"})
        ids.add(rid)
        d1, d2 = parse_date(r.get("period_start")), parse_date(r.get("period_end"))
        if not d1 or not d2:
            errors.append({"file": path, "index": i, "error": "bad period dates"})
        else:
            if not (DATE_MIN <= d1 <= DATE_MAX) or not (DATE_MIN <= d2 <= DATE_MAX):
                errors.append({"file": path, "index": i, "error": "date out of range"})
            if d1 > d2:
                errors.append({"file": path, "index": i, "error": "period_start after period_end"})
            if per_year is not None and d1.year != per_year:
                errors.append({"file": path, "index": i, "error": f"year {d1.year} != file year {per_year}"})
        if not r.get("source", {}).get("source_id"):
            errors.append({"file": path, "index": i, "error": "missing source_id"})
        if not isinstance(r.get("drug"), dict):
            errors.append({"file": path, "index": i, "error": "missing drug object"})
        obs = r.get("observation")
        if not isinstance(obs, dict):
            errors.append({"file": path, "index": i, "error": "missing observation object"})
        else:
            if not obs.get("metric"):
                errors.append({"file": path, "index": i, "error": "missing metric"})
            if not obs.get("unit"):
                errors.append({"file": path, "index": i, "error": "missing unit"})
            if obs.get("value") is not None and not isinstance(obs["value"], (int, float)):
                errors.append({"file": path, "index": i, "error": "value not numeric"})
        if r.get("quality", {}).get("is_imputed"):
            errors.append({"file": path, "index": i, "error": "is_imputed true"})
    return len(data)


def main():
    ensure_dirs()
    errors = []
    per_source = {}
    per_year = {}
    metrics = {}
    units = {}
    canon_names = set()
    orig_names = set()
    geos = set()
    mapped = unmapped = 0
    src_dates = {}
    for src in SOURCES:
        d = os.path.join(BY_SOURCE, src)
        if not os.path.isdir(d):
            continue
        n = 0
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(d, fn)
            n += validate_file(path, errors)
            for r in read_json(path, []):
                per_year[r["period_start"][:4]] = per_year.get(r["period_start"][:4], 0) + 1
                m = r["observation"]["metric"]
                metrics[m] = metrics.get(m, 0) + 1
                units[r["observation"]["unit"]] = units.get(r["observation"]["unit"], 0) + 1
                if r["drug"].get("mapping_status") == "mapped":
                    mapped += 1
                else:
                    unmapped += 1
                if r["drug"].get("canonical_name"):
                    canon_names.add(r["drug"]["canonical_name"])
                orig_names.add(r["drug"]["original_name"])
                g = r["geography"]
                geos.add((g.get("level"), g.get("name")))
                sd = src_dates.setdefault(src, [None, None])
                if sd[0] is None or r["period_start"] < sd[0]:
                    sd[0] = r["period_start"]
                if sd[1] is None or r["period_end"] > sd[1]:
                    sd[1] = r["period_end"]
        per_source[src] = n
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        path = os.path.join(COMBINED, f"{y}.json")
        if not os.path.exists(path):
            errors.append({"file": path, "error": "missing combined file"})
        else:
            validate_file(path, errors, per_year=y)
    write_json(os.path.join(DATA, "validation_errors.json"), errors)
    print("=== S_D validation summary ===")
    print("Records per source:")
    for s, n in per_source.items():
        print(f"  {s}: {n}")
    print("Records per year:")
    for y in sorted(per_year):
        print(f"  {y}: {per_year[y]}")
    print(f"Unique canonical drugs: {len(canon_names)}")
    print(f"Unique original drug names: {len(orig_names)}")
    print(f"Unique geographies: {len(geos)}")
    print(f"Mapped: {mapped}, Unmapped: {unmapped}")
    print("Records by metric:")
    for m, n in metrics.items():
        print(f"  {m}: {n}")
    print("Records by unit:")
    for u, n in units.items():
        print(f"  {u}: {n}")
    print("Earliest/latest date per source:")
    for s, (a, b) in src_dates.items():
        print(f"  {s}: {a} .. {b}")
    print(f"Validation errors: {len(errors)}")
    for e in errors[:50]:
        print("  ", e)


if __name__ == "__main__":
    main()