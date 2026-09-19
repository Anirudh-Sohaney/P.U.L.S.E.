"""Shared standard-library helpers for the global disease extraction package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import io
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


def cache_path(source_id: str, url: str, extension: str = "bin") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{safe_name(source_id)}_{digest}.{extension.lstrip('.') or 'bin'}"


def fetch_bytes(source_id: str, url: str, extension: str, refresh: bool = False) -> tuple[bytes, str, bool]:
    """Fetch bytes and retain the exact response in the local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = cache_path(source_id, url, extension)
    cache_hit = target.exists() and not refresh
    if cache_hit:
        body = target.read_bytes()
    else:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/csv,application/json;q=0.9,*/*;q=0.1",
                "User-Agent": "disease-global-extractor/1.0 (reproducible research)",
            },
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = response.read()
                target.write_bytes(body)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", "replace")
                last_error = RuntimeError(f"HTTP {exc.code} for {url}: {detail}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = RuntimeError(f"Unable to fetch {url}: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError(f"Unable to fetch {url}")
    return body, str(target.relative_to(ROOT)), cache_hit


def fetch_json(source_id: str, url: str, refresh: bool = False) -> tuple[Any, str, bool]:
    body, raw_file, cache_hit = fetch_bytes(source_id, url, "json", refresh=refresh)
    try:
        return json.loads(body.decode("utf-8-sig")), raw_file, cache_hit
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc


def fetch_json_post(source_id: str, url: str, payload: dict[str, Any], refresh: bool = False) -> tuple[Any, str, bool]:
    """POST JSON with a content-addressed cache keyed by URL and payload."""
    cache_key = url + "?" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    target = cache_path(source_id, cache_key, "json")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_hit = target.exists() and not refresh
    if cache_hit:
        body = target.read_bytes()
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "disease-global-extractor/1.0 (reproducible research)",
            },
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    body = response.read()
                target.write_bytes(body)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", "replace")
                last_error = RuntimeError(f"HTTP {exc.code} for {url}: {detail}")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = RuntimeError(f"Unable to fetch {url}: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError(f"Unable to fetch {url}")
    try:
        return json.loads(body.decode("utf-8-sig")), str(target.relative_to(ROOT)), cache_hit
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc


def xlsx_matrix(body: bytes, sheet_name: str) -> list[list[str]]:
    """Return worksheet rows as strings, including title/header rows."""
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    archive = zipfile.ZipFile(io.BytesIO(body))
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_map = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{package_rel_namespace}}}Relationship")
    }
    sheet_path = None
    for sheet in workbook.findall(f"{{{namespace}}}sheets/{{{namespace}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            target = relationship_map[sheet.attrib[f"{{{rel_namespace}}}id"]]
            sheet_path = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
            break
    if sheet_path is None:
        raise RuntimeError(f"Worksheet {sheet_name!r} not found")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_root.findall(f"{{{namespace}}}si"):
            shared.append("".join(text.text or "" for text in item.iter(f"{{{namespace}}}t")))
    root = ET.fromstring(archive.read(sheet_path))
    xml_rows = root.findall(f".//{{{namespace}}}sheetData/{{{namespace}}}row")

    def cell_value(cell: ET.Element) -> str:
        value = cell.findtext(f"{{{namespace}}}v", default="")
        if cell.attrib.get("t") == "s" and value != "":
            return shared[int(value)]
        if cell.attrib.get("t") == "inlineStr":
            return "".join(text.text or "" for text in cell.iter(f"{{{namespace}}}t"))
        return value

    def column_number(reference: str) -> int:
        letters = "".join(character for character in reference if character.isalpha())
        number = 0
        for character in letters:
            number = number * 26 + ord(character.upper()) - ord("A") + 1
        return max(0, number - 1)

    matrix: list[list[str]] = []
    for row in xml_rows:
        cells = row.findall(f"{{{namespace}}}c")
        width = max((column_number(cell.attrib.get("r", "")) for cell in cells), default=-1) + 1
        values = [""] * width
        for cell in cells:
            index = column_number(cell.attrib.get("r", ""))
            if index >= len(values):
                values.extend([""] * (index + 1 - len(values)))
            values[index] = cell_value(cell)
        matrix.append(values)
    return matrix


def xlsx_rows(body: bytes, sheet_name: str, expected_headers: set[str] | None = None) -> Iterator[dict[str, str]]:
    """Yield rows from a simple XLSX worksheet using only the standard library."""
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    archive = zipfile.ZipFile(io.BytesIO(body))
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_map = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{package_rel_namespace}}}Relationship")
    }
    sheet_path = None
    for sheet in workbook.findall(f"{{{namespace}}}sheets/{{{namespace}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            target = relationship_map[sheet.attrib[f"{{{rel_namespace}}}id"]]
            sheet_path = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
            break
    if sheet_path is None:
        raise RuntimeError(f"Worksheet {sheet_name!r} not found")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_root.findall(f"{{{namespace}}}si"):
            shared.append("".join(text.text or "" for text in item.iter(f"{{{namespace}}}t")))
    root = ET.fromstring(archive.read(sheet_path))
    rows = root.findall(f".//{{{namespace}}}sheetData/{{{namespace}}}row")
    if not rows:
        return

    def cell_value(cell: ET.Element) -> str:
        value = cell.findtext(f"{{{namespace}}}v", default="")
        if cell.attrib.get("t") == "s" and value != "":
            return shared[int(value)]
        if cell.attrib.get("t") == "inlineStr":
            return "".join(text.text or "" for text in cell.iter(f"{{{namespace}}}t"))
        return value

    def column_name(reference: str) -> str:
        return "".join(character for character in reference if character.isalpha())

    header_cells = rows[0].findall(f"{{{namespace}}}c")
    headers = [cell_value(cell).strip() for cell in header_cells]
    if expected_headers and not expected_headers.issubset(headers):
        missing = sorted(expected_headers.difference(headers))
        raise RuntimeError(f"Worksheet {sheet_name!r} is missing expected headers: {missing}")
    for row in rows[1:]:
        values = {column_name(cell.attrib.get("r", "")): cell_value(cell) for cell in row.findall(f"{{{namespace}}}c")}
        yield {header: values.get(column, "") for header, column in zip(headers, (column_name(cell.attrib.get("r", "")) for cell in header_cells)) if header}


def as_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    text = str(value).strip()
    if text.lower() in {"", "-", ".", "null", "none", "na", "n/a", "nan", "inf", "+inf", "-inf"}:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def write_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    """Write JSONL atomically so a failed stream cannot replace a good output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temporary_name, path)
        temporary_name = None
        return count
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
