#!/usr/bin/env python3
"""Shared helpers for the tariff_policy extraction scripts."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data"
METADATA = ROOT / "metadata"
START = dt.date(2012, 1, 1)
END = dt.date(2022, 12, 31)
USER_AGENT = "meditrack-tariff-policy-extractor/1.0 (reproducible research)"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "response"


def cache_path(label: str, suffix: str) -> Path:
    return CACHE / f"{safe_name(label)}{suffix}"


def fetch_bytes(url: str, *, label: str, refresh: bool = False, suffix: str = ".bin") -> tuple[bytes, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(label, suffix)
    if path.exists() and not refresh:
        return path.read_bytes(), path
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/csv, */*"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"GET failed for {url}: {exc}") from exc
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw, path


def fetch_json(url: str, *, label: str, refresh: bool = False) -> tuple[Any, Path]:
    raw, path = fetch_bytes(url, label=label, refresh=refresh, suffix=".json")
    try:
        return json.loads(raw.decode("utf-8-sig")), path
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"response from {url} is not valid JSON: {exc}") from exc


def parse_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    if re.fullmatch(r"\d{4}", text):
        return dt.date(int(text), 1, 1)
    return None


def year_date(value: Any, *, end: bool = False) -> dt.date | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return dt.date(year, 12, 31) if end else dt.date(year, 1, 1)
    return parse_date(text)


def clamp_interval(start: dt.date | None, end: dt.date | None) -> tuple[dt.date, dt.date] | None:
    start = start or END
    end = end or START
    if end < START or start > END:
        return None
    return max(start, START), min(end, END)


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"na", "n/a", "null", "-", "..", "…"}:
        return None
    text = text.replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def code(value: Any, width: int | None = None) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return text.zfill(width) if width and text.isdigit() else text


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_record(
    *, record_type: str, period_start: dt.date, period_end: dt.date,
    granularity: str, reporter: dict[str, Any], partner: dict[str, Any],
    product: dict[str, Any], category: str, subtype: str | None,
    rate: float | None, unit: str | None, medical_flag: bool | None,
    source_id: str, source_name: str, source_url: str, source_record_id: str | None,
    payload: Any, release: str | None = None, legal_text: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    identity = "|".join([
        source_id, str(source_record_id or ""), record_type,
        period_start.isoformat(), period_end.isoformat(),
    ])
    record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return {
        "record_id": record_id,
        "record_type": record_type,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "granularity": granularity,
        "reporter": reporter,
        "partner": partner,
        "product": product,
        "measure": {
            "category": category, "subtype": subtype, "rate": rate, "unit": unit,
            "medical_flag": medical_flag, "legal_text": legal_text,
        },
        "source": {
            "source_id": source_id, "source_name": source_name, "source_url": source_url,
            "source_record_id": source_record_id, "retrieved_at": utc_now(), "release": release,
        },
        "quality": {"is_imputed": False, "is_suppressed": False, "notes": notes},
        "source_payload_json": json_text(payload),
    }


FIELDS = [
    "record_id", "record_type", "period_start", "period_end", "granularity",
    "reporter_code", "reporter_name", "reporter_level", "partner_code", "partner_name",
    "partner_level", "product_code", "product_revision", "product_level",
    "product_description", "measure_category", "measure_subtype", "measure_rate",
    "measure_unit", "medical_flag", "legal_text", "source_id", "source_name",
    "source_url", "source_record_id", "retrieved_at", "release", "is_imputed",
    "is_suppressed", "quality_notes", "source_payload_json",
]


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"], "record_type": record["record_type"],
        "period_start": record["period_start"], "period_end": record["period_end"],
        "granularity": record["granularity"],
        "reporter_code": record["reporter"]["code"], "reporter_name": record["reporter"]["name"],
        "reporter_level": record["reporter"]["level"],
        "partner_code": record["partner"]["code"], "partner_name": record["partner"]["name"],
        "partner_level": record["partner"]["level"],
        "product_code": record["product"]["code"], "product_revision": record["product"]["revision"],
        "product_level": record["product"]["level"], "product_description": record["product"].get("description"),
        "measure_category": record["measure"]["category"], "measure_subtype": record["measure"]["subtype"],
        "measure_rate": record["measure"]["rate"], "measure_unit": record["measure"]["unit"],
        "medical_flag": record["measure"]["medical_flag"], "legal_text": record["measure"].get("legal_text"),
        "source_id": record["source"]["source_id"], "source_name": record["source"]["source_name"],
        "source_url": record["source"]["source_url"], "source_record_id": record["source"].get("source_record_id"),
        "retrieved_at": record["source"]["retrieved_at"], "release": record["source"].get("release"),
        "is_imputed": record["quality"]["is_imputed"], "is_suppressed": record["quality"]["is_suppressed"],
        "quality_notes": record["quality"].get("notes"), "source_payload_json": record["source_payload_json"],
    }


def write_records(name: str, records: Iterable[dict[str, Any]]) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    rows = [flatten(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
