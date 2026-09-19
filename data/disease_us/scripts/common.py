"""Shared standard-library helpers for the disease extraction package."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data" / "by_source"
METADATA_DIR = ROOT / "metadata"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80]


def cache_path(source_id: str, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{safe_name(source_id)}_{digest}.json"


def fetch_json(source_id: str, url: str, refresh: bool = False) -> tuple[Any, str, bool]:
    """Fetch JSON, retaining the exact response in the local cache.

    Returns (decoded payload, workspace-relative raw path, cache_hit).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = cache_path(source_id, url)
    cache_hit = target.exists() and not refresh
    if cache_hit:
        body = target.read_bytes()
    else:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "disease-us-extractor/1.0 (reproducible research)",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
        target.write_bytes(body)
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in response cache {target}: {exc}") from exc
    return payload, str(target.relative_to(ROOT)), cache_hit


def socrata_pages(
    source_id: str,
    base_url: str,
    where: str,
    refresh: bool = False,
    page_size: int = 50_000,
) -> Iterator[tuple[list[dict[str, Any]], str, bool]]:
    """Yield Socrata pages until the endpoint returns fewer than page_size rows."""
    offset = 0
    while True:
        params = {
            "$limit": str(page_size),
            "$offset": str(offset),
            "$order": ":id",
        }
        if where:
            params["$where"] = where
        query = "&".join(f"{key}={urllib.parse.quote(value, safe="'():,")}" for key, value in params.items())
        url = f"{base_url}?{query}"
        payload, raw_path, cache_hit = fetch_json(source_id, url, refresh=refresh)
        if not isinstance(payload, list):
            raise RuntimeError(f"Expected a Socrata list response from {url}")
        yield payload, raw_path, cache_hit
        if len(payload) < page_size:
            break
        offset += page_size


def as_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text in {"", "-", ".", "null", "None", "NA", "N/A"}:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
