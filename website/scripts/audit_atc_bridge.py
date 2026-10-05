"""Measure whether the historical ATC crosswalk covers the latest SDUD source.

This is an audit of real source rows, not an ATC forecast or a replacement for
the monthly HHS pharmacy-provider target.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from backend import signal_store


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = ROOT / "data/targeted_additions/rxnorm_ndc_atc/data/rxnorm_ndc_atc_mapping.csv.gz"
DEFAULT_MANIFEST = ROOT / "data/targeted_additions/rxnorm_ndc_atc/manifest.json"
QUALIFIED_ATC = frozenset((
    "C10", "C08", "N06", "B01", "G01", "A10", "D06", "A02", "A06",
    "V03", "C02", "M03", "D01", "L01", "C03", "N07", "D10", "V04",
))


def load_mapping(mapping_path: Path, manifest_path: Path) -> tuple[dict[str, set[str]], str]:
    raw = mapping_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest != manifest["sha256"]["mapping_csv"]:
        raise ValueError("ATC mapping bytes do not match the source manifest")
    mapping: dict[str, set[str]] = {}
    with gzip.open(mapping_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not {"ndc11", "class_id"}.issubset(reader.fieldnames or []):
            raise ValueError("ATC mapping lacks NDC and class columns")
        for row in reader:
            ndc, class_id = row["ndc11"].strip(), row["class_id"].strip()
            if len(ndc) == 11 and ndc.isdigit() and len(class_id) >= 3:
                mapping.setdefault(ndc, set()).add(class_id[:3])
    return mapping, digest


def audit(mapping_path: Path = DEFAULT_MAPPING,
          manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    mapping, mapping_digest = load_mapping(mapping_path, manifest_path)
    signal_store.initialize()
    with signal_store.connect() as db:
        snapshot = db.execute("""SELECT content_hash, source_year, latest_quarter,
            source_url, retrieved_at, last_seen_at FROM sdud_snapshots
            ORDER BY last_seen_at DESC LIMIT 1""").fetchone()
        if snapshot is None:
            raise ValueError("No verified SDUD snapshot is stored")
        rows = db.execute("""SELECT ndc, prescriptions, suppressed FROM sdud_rows
            WHERE content_hash=? AND year=? AND quarter=?""",
            (snapshot["content_hash"], snapshot["source_year"],
             snapshot["latest_quarter"])).fetchall()
    if not rows:
        raise ValueError("Latest SDUD period has no rows")
    mapped = [row for row in rows if row["ndc"] in mapping]
    all_ndcs = {row["ndc"] for row in rows}
    mapped_ndcs = {row["ndc"] for row in mapped}
    reported = sum(row["prescriptions"] or 0 for row in rows)
    mapped_reported = sum(row["prescriptions"] or 0 for row in mapped)
    represented = sorted(QUALIFIED_ATC.intersection(
        class_id for ndc in mapped_ndcs for class_id in mapping[ndc]))
    return {
        "sdud_period": f"{snapshot['source_year']}-Q{snapshot['latest_quarter']}",
        "sdud_snapshot_sha256": snapshot["content_hash"],
        "sdud_source_url": snapshot["source_url"],
        "sdud_first_retrieved_at": snapshot["retrieved_at"],
        "sdud_last_checked_at": snapshot["last_seen_at"],
        "historical_atc_mapping_sha256": mapping_digest,
        "sdud_rows": len(rows),
        "sdud_suppressed_rows": sum(row["suppressed"] for row in rows),
        "sdud_unique_ndcs": len(all_ndcs),
        "mapped_rows": len(mapped),
        "mapped_unique_ndcs": len(mapped_ndcs),
        "mapped_ndc_fraction": len(mapped_ndcs) / len(all_ndcs),
        "reported_prescriptions_lower_bound": reported,
        "mapped_reported_prescriptions_lower_bound": mapped_reported,
        "mapped_reported_fraction": mapped_reported / reported if reported else None,
        "qualified_atc_groups_represented": represented,
        "qualified_atc_groups_missing": sorted(QUALIFIED_ATC.difference(represented)),
        "usable_as_monthly_hhs_model_input": False,
        "reason": "Quarterly statewide SDUD prescriptions and suppression differ from "
                  "monthly HHS pharmacy-provider claim lines; unmapped and suppressed "
                  "counts remain unknown.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(audit(args.mapping, args.manifest), indent=2, sort_keys=True))
