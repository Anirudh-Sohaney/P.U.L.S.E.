"""Export only public signal identities and metadata from the local catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backend.signal_store import IDENTITY_FIELDS, signal_uid


METADATA_FIELDS = ("unit", "source_name", "source_url", "state_definition",
                   "model_output_type")
DEFAULT_SOURCE = (Path(__file__).resolve().parents[2] / "test" / "test_Signals" /
                  "signals_2023_2025.csv.gz")
DEFAULT_OUTPUT = (Path(__file__).resolve().parents[1] / "catalog" /
                  "signal_definitions.json")


def export(source: Path, output: Path) -> dict:
    frame = pd.read_csv(source, low_memory=False).fillna("")
    fields = (*IDENTITY_FIELDS, *METADATA_FIELDS)
    if not set(fields).issubset(frame):
        raise ValueError("Historical signal catalog lacks identity metadata")
    if frame.duplicated(IDENTITY_FIELDS).any():
        variants = frame.groupby(list(IDENTITY_FIELDS), dropna=False)[list(METADATA_FIELDS)].nunique()
        if variants.gt(1).any().any():
            raise ValueError("One signal identity has conflicting metadata")
    definitions = []
    for row in frame.drop_duplicates(list(IDENTITY_FIELDS))[list(fields)].to_dict("records"):
        item = {field: str(row[field]).strip() for field in fields}
        item["id"] = signal_uid(item)
        definitions.append(item)
    definitions.sort(key=lambda item: item["id"])
    if len(definitions) != 1312 or len({item["id"] for item in definitions}) != 1312:
        raise ValueError("Expected exactly 1,312 unique public signal identities")
    payload = {"schema": "pulse_signal_definitions_v1",
               "definition_count": len(definitions),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "note": "Definitions only. No historical values or timestamps are included.",
               "definitions": definitions}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return {"definition_count": len(definitions), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
