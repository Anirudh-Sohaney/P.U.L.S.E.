#!/usr/bin/env python3
"""Download public crosswalk inputs.

The script intentionally does not guess authenticated URLs. It discovers Census
links from the official page, downloads the public Dartmouth annual ZIP/HSA/HRR
files listed by the official page, retrieves RxNav's public NDC catalog, and
retrieves a bounded GLEIF API page. Use --refresh to replace cache files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html.parser
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data"
USER_AGENT = "meditrack-crosswalk-extractor/1.0 (reproducible research)"
CENSUS_PAGE = "https://www.census.gov/naics/concordances/concordances.html"
DARTMOUTH_PAGE = "https://data.dartmouthatlas.org/supplemental/"
RXNAV_VERSION = "https://rxnav.nlm.nih.gov/REST/version.json"
RXNAV_NDCS = "https://rxnav.nlm.nih.gov/REST/allNDCstatus.json"
GLEIF_API = "https://api.gleif.org/api/v1/lei-records"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def request(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def save(name: str, url: str, refresh: bool, records: dict[str, object]) -> Path | None:
    path = CACHE / name
    if path.exists() and not refresh:
        records[name] = {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "retrieved_at": "existing"}
        print(f"skip  {name}")
        return path
    try:
        raw = request(url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        records[name] = {"url": url, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"fail  {name}: {exc}", file=sys.stderr)
        return None
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    records[name] = {"url": url, "path": str(path.relative_to(ROOT)), "bytes": len(raw), "retrieved_at": now()}
    print(f"got   {name} ({len(raw)} bytes)")
    return path


def discover_links(page_url: str, suffixes: tuple[str, ...], text_pattern: str | None = None) -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(request(page_url).decode("utf-8", "ignore"))
    result: list[tuple[str, str]] = []
    for href, text in parser.links:
        absolute = urllib.parse.urljoin(page_url, href)
        parsed = urllib.parse.urlparse(absolute)
        # Census occasionally renders internal filesystem paths such as
        # /eos/www/... as hrefs. They are not downloadable web URLs; retaining
        # them would create repeated 404s and misleading cache entries.
        if parsed.scheme not in {"http", "https"} or parsed.netloc != urllib.parse.urlparse(page_url).netloc:
            continue
        if any(ord(ch) < 32 or ch.isspace() for ch in absolute):
            continue
        if parsed.path.lower().startswith("/eos/www/"):
            continue
        if not parsed.path.lower().endswith(suffixes):
            continue
        if text_pattern and not re.search(text_pattern, text, flags=re.I) and not re.search(text_pattern, absolute, flags=re.I):
            continue
        result.append((absolute, text))
    return result


def write_pending(name: str, text: str) -> None:
    (DATA / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--gleif-all", action="store_true", help="download every GLEIF API page; large")
    ap.add_argument("--no-network", action="store_true", help="only write pending access notes")
    args = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {}

    write_pending("WHO_ATC_PENDING.txt", "WHO ATC/DDD: complete machine-readable annual indexes and historical vintages require an authorized WHOCC portal download. Drop authorized files under cache/who_atc/ and document the release in source_manifest.json.")
    write_pending("WITS_PENDING.txt", "WITS product concordances: the official portal exposes an interactive download workflow. No stable unauthenticated bulk endpoint was assumed. Download an authorized table from https://wits.worldbank.org/product_concordance.html into cache/wits/ and document its release before normalization.")
    write_pending("HUD_USPS_PENDING.txt", "HUD USPS Crosswalk: HUD states that current downloads/API access require portal login and an access token. Obtain the required 2012 Q1–2022 Q4 files from https://www.huduser.gov/portal/datasets/usps_crosswalk.html, place them under cache/hud_usps/, and record quarter/geography type before normalization.")
    if args.no_network:
        print("network disabled; pending notes written")
        return 0

    # RxNav public codelist and version. allNDCstatus is intentionally kept raw;
    # it is ~20 MB and is normalized by normalize.py.
    save("rxnav_version.json", RXNAV_VERSION, args.refresh, records)
    save("rxnav_allNDCstatus.json", RXNAV_NDCS, args.refresh, records)

    # Discover and download official Census spreadsheets without hard-coding a
    # potentially changing CMS/CDN URL. Keep only concordance XLS/XLSX links.
    try:
        census_links = discover_links(CENSUS_PAGE, (".xls", ".xlsx"), "NAICS|ISIC|SIC")
        seen: set[str] = set()
        for url, label in census_links:
            if url in seen:
                continue
            seen.add(url)
            suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if suffix not in {".xls", ".xlsx"}:
                continue
            # Link text is often only the generic label "XLSX"; use the URL
            # basename in that case so distinct concordances do not collide.
            candidate = Path(urllib.parse.urlparse(url).path).name
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
            safe = candidate if safe_label.lower() in {"xls", "xlsx"} else safe_label
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe).strip("_") or candidate
            if not safe.lower().endswith(suffix):
                safe += suffix
            save(f"census_{safe}", url, args.refresh, records)
    except Exception as exc:
        records["census_discovery"] = {"url": CENSUS_PAGE, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"fail  Census discovery: {exc}", file=sys.stderr)

    # Dartmouth's official page exposes direct annual ZIPHSAHRR files. Only
    # download links actually present on that page; no guessed 2020-2022 files.
    try:
        dart_links = discover_links(DARTMOUTH_PAGE, (".zip", ".csv", ".xls", ".xlsx"), "ZipHsaHrr|ZIP.*HSA|HSA.*HRR")
        seen = set()
        for url, label in dart_links:
            if url in seen:
                continue
            seen.add(url)
            filename = Path(urllib.parse.urlparse(url).path).name
            if not filename:
                continue
            save(f"dartmouth_{filename}", url, args.refresh, records)
    except Exception as exc:
        records["dartmouth_discovery"] = {"url": DARTMOUTH_PAGE, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"fail  Dartmouth discovery: {exc}", file=sys.stderr)

    # GLEIF JSON: one page by default, all pages only by explicit opt-in.
    try:
        page = 1
        while True:
            url = GLEIF_API + "?" + urllib.parse.urlencode({"page[size]": 100, "page[number]": page})
            name = "gleif_lei_page_%04d.json" % page
            if save(name, url, args.refresh, records) is None:
                break
            if not args.gleif_all:
                break
            payload = json.loads((CACHE / name).read_text(encoding="utf-8"))
            last = int(((payload.get("meta") or {}).get("pagination") or {}).get("lastPage") or page)
            if page >= last:
                break
            page += 1
    except Exception as exc:
        records["gleif"] = {"url": GLEIF_API, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"fail  GLEIF: {exc}", file=sys.stderr)

    (CACHE / "manifest.json").write_text(json.dumps({"retrieved_at": now(), "files": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest written to {CACHE / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
