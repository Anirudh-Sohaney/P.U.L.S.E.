#!/usr/bin/env python3
"""Capture public catalog metadata for sources without anonymous observations."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from common import CACHE_DIR, METADATA_DIR, fetch_bytes, utc_now

CATALOGS = {
    "ihme_gbd": "https://ghdx.healthdata.org/ihme_data",
    "ghdx": "https://ghdx.healthdata.org/gbd-2021",
    "promed": "https://www.promedmail.org/",
    "hmd": "https://www.mortality.org/",
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "link", "script"}:
            return
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append(value)


def capture(source_id: str, url: str, refresh: bool) -> dict[str, Any]:
    body, raw_file, cache_hit = fetch_bytes(source_id + "_catalog", url, "html", refresh=refresh)
    text = body.decode("utf-8", "replace")
    parser = LinkParser()
    parser.feed(text)
    links = sorted({urllib.parse.urljoin(url, link) for link in parser.links})
    interesting = [link for link in links if any(token in link.lower() for token in (
        "download", "file", "csv", "zip", "xlsx", "api", "archive", "login", "record", "search",
    ))]
    return {
        "status": "captured",
        "source_id": source_id,
        "source_url": url,
        "raw_file": raw_file,
        "cache_hit": cache_hit,
        "bytes": len(body),
        "interesting_links": interesting,
        "metadata_captured": True,
        "observations_available": source_id == "hmd",
        "access_requirement": (
            "provider archive permission" if source_id == "promed" else
            "registered or authenticated download" if source_id in {"ihme_gbd", "ghdx", "hmd"} else
            "none"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    results = {source_id: capture(source_id, url, args.refresh) for source_id, url in CATALOGS.items()}
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {"captured_at": utc_now(), "catalogs": results}
    (METADATA_DIR / "catalog_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
