"""Build yearly combined files (2000-2022) from by_source normalized files.
Year-major: one year in memory at a time (avoids OOM on large datasets).
Combined files are plain concatenations of source records (no cross-source
aggregation). Years with no observations get an empty array."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BY_SOURCE, COMBINED, ensure_dirs, read_json

SOURCES = ["medicaid_sdud", "nhs_prescribing", "cms_part_d", "dea_arcos",
           "fda_recalls", "fda_shortages"]
YEAR_MIN, YEAR_MAX = 2000, 2022


def main():
    ensure_dirs()
    total = 0
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        recs = []
        for src in SOURCES:
            d = os.path.join(BY_SOURCE, src)
            p = os.path.join(d, f"{y}.json")
            if os.path.exists(p):
                recs.extend(read_json(p, []))
        with open(os.path.join(COMBINED, f"{y}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, separators=(",", ":"))
        total += len(recs)
        print(f"{y}: {len(recs)}", flush=True)
    print(f"combined: {total} records across {YEAR_MAX - YEAR_MIN + 1} years")


if __name__ == "__main__":
    main()