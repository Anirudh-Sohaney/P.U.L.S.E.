"""Shared helpers for the economics_us extraction package.

Only Python's standard library is used so extraction remains runnable in the
existing project environment. Raw HTTP responses are cached before parsing.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data" / "by_source"
METADATA = ROOT / "metadata"
START_YEAR = 2012
END_YEAR = 2022
USER_AGENT = "economics-us-extractor/1.0 (reproducible research)"


class FetchError(RuntimeError):
    """An HTTP or decoding failure with a concise, serializable message."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def ensure_dirs() -> None:
    for path in (CACHE, DATA, METADATA):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:160]


def _cache_path(label: str, suffix: str, request_fingerprint: str = "") -> Path:
    digest = hashlib.sha256(request_fingerprint.encode("utf-8")).hexdigest()[:16] if request_fingerprint else ""
    suffix_label = f"_{digest}" if digest else ""
    return CACHE / f"{safe_name(label)}{suffix_label}{suffix}"


def clear_source_partitions(source_id: str) -> None:
    """Remove prior normalized partitions before replacing a successful run."""
    out_dir = DATA / source_id
    if out_dir.exists():
        for path in out_dir.glob("*.jsonl"):
            path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def fiscal_partition_label(record: dict[str, Any]) -> str:
    if record.get("granularity") == "fiscal_year":
        end = str(record.get("period_end", ""))
        if len(end) >= 4 and end[:4].isdigit():
            return f"FY{end[:4]}"
    return str(record.get("period_start", ""))[:4]


def _request_fingerprint(url: str, method: str, body: bytes | None) -> str:
    return "|".join((method.upper(), url, (body or b"").decode("utf-8", errors="replace")))


def _redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.query:
        return url
    safe_query = urllib.parse.urlencode([
        (key, "<redacted>" if key.lower() in {"key", "apikey", "api_key", "userid", "registrationkey"} else value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    ])
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, safe_query, parsed.fragment))


def fetch(
    url: str,
    *,
    label: str,
    method: str = "GET",
    body: bytes | None = None,
    refresh: bool = False,
    timeout: int = 90,
    retries: int = 3,
    content_suffix: str = ".bin",
    headers: dict[str, str] | None = None,
) -> tuple[bytes, Path]:
    """Fetch and cache a response; retry only transient transport/5xx errors."""
    ensure_dirs()
    path = _cache_path(label, content_suffix, _request_fingerprint(url, method, body))
    if path.exists() and not refresh:
        return path.read_bytes(), path
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            path.write_bytes(payload)
            return payload, path
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                raise FetchError(f"HTTP {exc.code} for {url}", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(2 ** attempt)
    raise FetchError(f"request failed for {url}: {last_error}") from last_error


def get_text(url: str, **kwargs: Any) -> tuple[str, Path]:
    payload, path = fetch(url, **kwargs)
    return payload.decode("utf-8-sig", errors="replace"), path


def get_json(url: str, **kwargs: Any) -> tuple[Any, Path]:
    text, path = get_text(url, **kwargs)
    try:
        return json.loads(text), path
    except json.JSONDecodeError as exc:
        raise FetchError(f"non-JSON response cached from {url}") from exc


def post_json(url: str, payload: dict[str, Any], **kwargs: Any) -> tuple[Any, Path]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    kwargs.setdefault("method", "POST")
    kwargs["body"] = body
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("Content-Type", "application/json")
    kwargs.setdefault("content_suffix", ".json")
    return get_json(url, **kwargs)


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in {"NA", "N/A", "NULL", "NONE", "(D)", "D", "N", "S"}:
        return None
    # BLS/Census/QCEW sometimes use a leading sign or parentheses.
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_date_period(year: int, period: str | None = None) -> tuple[str, str, str]:
    """Return ISO start, end, and canonical granularity for a BLS period."""
    if not period or period == "M13":
        return f"{year}-01-01", f"{year}-12-31", "year"
    if period.startswith("M") and period[1:].isdigit():
        month = int(period[1:])
        if 1 <= month <= 12:
            start = dt.date(year, month, 1)
            end = dt.date(year + (month == 12), (month % 12) + 1, 1) - dt.timedelta(days=1)
            return start.isoformat(), end.isoformat(), "month"
    return f"{year}-01-01", f"{year}-12-31", "year"


def canonical_record(
    *,
    source_id: str,
    source_name: str,
    source_url: str,
    source_record_id: str | None,
    period_start: str,
    period_end: str,
    granularity: str,
    geography: dict[str, Any],
    metric: str,
    value: float,
    unit: str,
    semantics: str,
    vintage_date: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    identity = "|".join(str(x) for x in (source_id, source_record_id, period_start, metric, geography.get("code"), vintage_date))
    record_id = f"{source_id}:{hashlib.sha1(identity.encode()).hexdigest()[:20]}"
    return {
        "record_id": record_id,
        "period_start": period_start,
        "period_end": period_end,
        "granularity": granularity,
        "geography": geography,
        "observation": {"metric": metric, "value": value, "unit": unit, "semantics": semantics},
        "source": {
            "source_id": source_id,
            "source_name": source_name,
            "source_url": source_url,
            "source_record_id": source_record_id,
            "retrieved_at": utc_now(),
            "vintage_date": vintage_date,
        },
        "quality": {"is_imputed": False, "is_suppressed": False, "notes": notes},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_partitioned(source_id: str, records: Iterable[dict[str, Any]], *, prefix: str = "", clear: bool = True) -> dict[str, int]:
    """Replace JSONL partitions, using calendar year or explicit fiscal year."""
    if clear:
        clear_source_partitions(source_id)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        label = prefix or fiscal_partition_label(record)
        grouped.setdefault(label, []).append(record)
    out_dir = DATA / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for label, rows in sorted(grouped.items()):
        path = out_dir / f"{safe_name(label)}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[label] = len(rows)
    return counts


def csv_rows(text: str) -> Iterable[dict[str, str]]:
    return csv.DictReader(text.splitlines())


def state_geo(code: str, name: str | None = None) -> dict[str, Any]:
    return {"country": "US", "name": name or code, "code": code, "level": "state"}


def national_geo() -> dict[str, Any]:
    return {"country": "US", "name": "United States", "code": "US", "level": "national"}
