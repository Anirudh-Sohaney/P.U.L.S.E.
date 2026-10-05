"""Import the supplied historical signal catalog without inventing missing dates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.signal_store import (import_catalog, import_definitions,
                                  import_news_bridge_history, freshness)


DEFAULT_DEFINITIONS = (Path(__file__).resolve().parents[1] / "catalog" /
                       "signal_definitions.json")
DEFAULT_HISTORY = (Path(__file__).resolve().parents[1] / "catalog" /
                   "historical_signal_catalog_2023_2025.csv.gz")
DEFAULT_HISTORY_SHA256 = "64d6c0f5f9a2efe36a76e8a2c89795f5a13c8270246eff4989f387ad0ac8cbed"
NEWS_BRIDGE_HISTORY = (Path(__file__).resolve().parents[1] / "catalog" /
                       "news_only_catalog_features.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--source", type=Path,
                        help="Use a separately verified historical value file instead of the bundled catalog")
    args = parser.parse_args()
    seeded = import_definitions(args.definitions)
    history_source = args.source or DEFAULT_HISTORY
    if not history_source.is_file():
        raise FileNotFoundError(history_source)
    if args.source is None and hashlib.sha256(history_source.read_bytes()).hexdigest() != DEFAULT_HISTORY_SHA256:
        raise ValueError("Bundled historical catalog checksum mismatch")
    historical = import_catalog(history_source)
    news_bridge = import_news_bridge_history(NEWS_BRIDGE_HISTORY)
    print(json.dumps({"definitions": seeded, "history": historical,
                      "news_bridge_history": news_bridge,
                      "database": freshness()}, indent=2))


if __name__ == "__main__":
    main()
