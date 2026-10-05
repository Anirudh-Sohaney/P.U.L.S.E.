"""Official HHS release checks fail visibly before any unvalidated publication."""

import json

import pytest

from backend.config import settings
from backend import refresh_hhs_ndc_release as watcher
from backend.signal_store import freshness


def _page(artifacts):
    dataset = {"slug": "medicaid-provider-spending-ndc", "versions": [
        {"version": "2026-07-24", "artifacts": []},
        {"version": "2027-01-01", "artifacts": artifacts},
    ]}
    chunk = '6:["$","$e",null,{"dataset":' + json.dumps(dataset) + '}]'
    return 'self.__next_f.push([1,' + json.dumps(chunk) + '])'


def test_hhs_release_parser_requires_official_zip_and_checksum():
    slug = "medicaid-provider-spending-ndc"
    artifact = {"name": f"{slug}.csv.zip", "format": "zip",
                "url": f"https://stopendataprod.blob.core.windows.net/datasets/"
                       f"{slug}/2027-01-01/dataset/{slug}.csv.zip",
                "checksum": "sha256:" + "a" * 64}
    assert watcher.parse_release_page(_page([artifact]), slug=slug) == {
        "version": "2027-01-01", "zip_url": artifact["url"],
        "zip_sha256": "a" * 64}
    with pytest.raises(ValueError, match="unexpected origin or path"):
        watcher.parse_release_page(_page([{**artifact,
            "url": artifact["url"].replace("blob.core.windows.net", "example.com")}] ),
            slug=slug)
    with pytest.raises(ValueError, match="SHA-256"):
        watcher.parse_release_page(_page([{**artifact, "checksum": "unknown"}]), slug=slug)


def test_hhs_new_release_is_reported_without_becoming_a_model_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    current = json.loads(watcher.MANIFEST.read_text(encoding="utf-8"))
    monkeypatch.setattr(watcher, "discover_release", lambda: {
        "version": "2027-01-01", "zip_sha256": "b" * 64,
        "zip_url": "https://stopendataprod.blob.core.windows.net/new.zip"})
    with pytest.raises(RuntimeError, match="download, validate, and evaluate"):
        watcher.refresh_hhs_ndc_release()
    assert watcher.SOURCE_NAME in freshness()["failed_sources"]
    monkeypatch.setattr(watcher, "discover_release", lambda: {
        "version": current["processed_version"],
        "zip_sha256": current["processed_zip_sha256"], "zip_url": ""})
    assert watcher.refresh_hhs_ndc_release()["status"] == "success"
    assert watcher.SOURCE_NAME not in freshness()["failed_sources"]
