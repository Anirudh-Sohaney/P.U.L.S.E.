import io
import json
import hashlib

import pandas as pd
import pytest

from backend import refresh_cms_partd_source as source
from backend.config import settings


CMS_URL = "https://data.cms.gov/sites/default/files/2027-05/partd-2025.csv"


@pytest.fixture(autouse=True)
def fixed_revision_timestamp(monkeypatch):
    monkeypatch.setattr(source, "revision_timestamp", lambda _url: "2027-05-01T00:00:00+00:00")


def test_discover_release_uses_official_catalog_and_checks_unique_csv(monkeypatch):
    payload = {"datasetSeries": [{"title": source.DATASET_TITLE}], "dataset": [
        {"landingPage": {"accessURL": source.LANDING_URL},
         "temporal": [{"endDate": "2025-12-31"}],
         "distribution": [{"mediaType": "text/csv", "downloadURL": CMS_URL}]},
    ]}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(source, "urlopen", lambda *_args, **_kwargs:
                        Response(json.dumps(payload).encode()))
    assert source.discover_latest_release() == (2025, CMS_URL)
    payload["dataset"][0]["distribution"][0]["downloadURL"] = "http://example.org/unsafe.csv"
    with pytest.raises(ValueError, match="official CSV"):
        source.discover_latest_release()


def test_download_aggregates_arkansas_generic_claims(monkeypatch):
    csv_bytes = ("Prscrbr_Geo_Lvl,Prscrbr_Geo_Cd,Gnrc_Name,Tot_Clms\n"
                 "National,,Drug A,100\nState,05,Drug A,10\n"
                 "State,05,Drug A,20\nState,05,Drug B,7\n"
                 "State,06,Drug A,999\n").encode()
    monkeypatch.setattr(source, "urlopen", lambda *_args, **_kwargs: io.BytesIO(csv_bytes))
    frame = source.download_arkansas_year(CMS_URL, 2025)
    assert frame.to_dict("records") == [
        {"year": 2025, "state": "arkansas", "drug_key": "Drug A", "demand_claims": 30},
        {"year": 2025, "state": "arkansas", "drug_key": "Drug B", "demand_claims": 7},
    ]


def test_new_release_is_staged_with_manifest_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024],
                                    "source_urls": ["https://data.cms.gov/old.csv"]}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(source, "discover_releases", lambda: {2025: CMS_URL})
    calls = []

    def download(url, year):
        calls.append((url, year))
        return pd.DataFrame([{"year": 2025, "state": "arkansas", "drug_key": "Drug A",
                              "demand_claims": 20}])

    monkeypatch.setattr(source, "download_arkansas_year", download)
    first = source.refresh_cms_partd_source()
    assert first["rows_written"] == 1
    assert calls == [(CMS_URL, 2025)]
    assert pd.read_csv(source.current_source_path())["year"].tolist() == [2024, 2025]
    committed = json.loads(source.current_manifest_path().read_text())
    assert committed["source_urls"][-1] == CMS_URL
    assert source.current_source_path().name == committed["source_file"]
    second = source.refresh_cms_partd_source()
    assert second["rows_written"] == 0
    assert calls == [(CMS_URL, 2025)]

    revised_url = "https://data.cms.gov/sites/default/files/2027-06/partd-2025.csv"
    monkeypatch.setattr(source, "discover_releases", lambda: {2025: revised_url})
    monkeypatch.setattr(source, "download_arkansas_year", lambda _url, _year:
                        pd.DataFrame([{"year": 2025, "state": "arkansas",
                                       "drug_key": "Drug A", "demand_claims": 21}]))
    assert source.refresh_cms_partd_source()["rows_written"] == 1
    revised = pd.read_csv(source.current_source_path())
    assert revised["year"].tolist() == [2024, 2025]
    assert revised.loc[revised["year"] == 2025, "demand_claims"].iloc[0] == 21


def test_same_url_revision_replaces_current_year_and_preserves_prior_years(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024], "source_urls": [CMS_URL],
                                    "latest_source_last_modified": "2027-04-01T00:00:00+00:00"}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(source, "discover_releases", lambda: {2024: CMS_URL})
    calls = []

    def download(url, year):
        calls.append((url, year))
        return pd.DataFrame([{"year": year, "state": "arkansas", "drug_key": "Drug A",
                              "demand_claims": 11}])

    monkeypatch.setattr(source, "download_arkansas_year", download)
    first = source.refresh_cms_partd_source()
    assert first["rows_written"] == 1
    assert calls == [(CMS_URL, 2024)]
    assert pd.read_csv(source.current_source_path())["demand_claims"].tolist() == [11]
    assert json.loads(source.current_manifest_path().read_text())["latest_source_last_modified"] == "2027-05-01T00:00:00+00:00"
    assert source.refresh_cms_partd_source()["rows_written"] == 0
    assert calls == [(CMS_URL, 2024)]


def test_initial_revision_metadata_verifies_content_and_keeps_matching_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024], "source_urls": [CMS_URL]}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(source, "discover_releases", lambda: {2024: CMS_URL})
    calls = []

    def matching_download(url, year):
        calls.append((url, year))
        return pd.DataFrame([{"year": 2024, "state": "arkansas",
                              "drug_key": "Drug A", "demand_claims": 10}])

    monkeypatch.setattr(source, "download_arkansas_year", matching_download)
    assert source.refresh_cms_partd_source()["rows_written"] == 0
    assert calls == [(CMS_URL, 2024)]
    assert source.current_source_path().read_bytes() == bundled.read_bytes()
    committed = json.loads(source.current_manifest_path().read_text())
    assert committed["latest_source_last_modified"] == "2027-05-01T00:00:00+00:00"
    assert committed["latest_source_verified_url"] == CMS_URL
    assert source.refresh_cms_partd_source()["rows_written"] == 0
    assert calls == [(CMS_URL, 2024)]


def test_first_content_verification_detects_same_url_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024], "source_urls": [CMS_URL],
                                    "latest_source_last_modified":
                                    "2027-05-01T00:00:00+00:00"}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(source, "discover_releases", lambda: {2024: CMS_URL})
    monkeypatch.setattr(source, "download_arkansas_year", lambda _url, _year:
                        pd.DataFrame([{"year": 2024, "state": "arkansas",
                                       "drug_key": "Drug A", "demand_claims": 11}]))
    assert source.refresh_cms_partd_source()["rows_written"] == 1
    assert pd.read_csv(source.current_source_path())["demand_claims"].tolist() == [11]


def test_missed_releases_are_backfilled_without_inventing_unpublished_years(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024],
                                    "source_urls": ["https://data.cms.gov/old.csv"]}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    release_urls = {2025: CMS_URL, 2027: "https://data.cms.gov/partd-2027.csv"}
    monkeypatch.setattr(source, "discover_releases", lambda: release_urls)
    calls = []

    def download(url, year):
        calls.append((url, year))
        return pd.DataFrame([{"year": year, "state": "arkansas", "drug_key": "Drug A",
                              "demand_claims": year}])

    monkeypatch.setattr(source, "download_arkansas_year", download)
    result = source.refresh_cms_partd_source()
    assert result["rows_written"] == 2
    assert calls == [(release_urls[2025], 2025), (release_urls[2027], 2027)]
    staged = pd.read_csv(source.current_source_path())
    assert staged["year"].tolist() == [2024, 2025, 2027]
    assert json.loads(source.current_manifest_path().read_text())["years"] == [2024, 2025, 2027]
    assert source.refresh_cms_partd_source()["rows_written"] == 0
    assert len(calls) == 2


def test_interrupted_manifest_commit_keeps_previous_source_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    pd.DataFrame([{"year": 2024, "state": "arkansas", "drug_key": "Drug A",
                   "demand_claims": 10}]).to_csv(bundled, index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"years": [2024],
                                    "source_urls": ["https://data.cms.gov/old.csv"]}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    release_url = "https://data.cms.gov/partd-2025.csv"
    monkeypatch.setattr(source, "discover_releases", lambda: {2025: release_url})
    monkeypatch.setattr(source, "download_arkansas_year", lambda _url, year:
                        pd.DataFrame([{"year": year, "state": "arkansas",
                                       "drug_key": "Drug A", "demand_claims": 20}]))
    source.refresh_cms_partd_source()
    first_source = source.current_source_path()
    first_manifest = source.current_manifest_path().read_bytes()

    revised_url = "https://data.cms.gov/partd-2025-revised.csv"
    monkeypatch.setattr(source, "discover_releases", lambda: {2025: revised_url})
    monkeypatch.setattr(source, "download_arkansas_year", lambda _url, year:
                        pd.DataFrame([{"year": year, "state": "arkansas",
                                       "drug_key": "Drug A", "demand_claims": 21}]))
    original_replace = source.os.replace

    def interrupt_before_manifest(staged, destination):
        if destination == source.runtime_manifest_path():
            raise OSError("simulated interruption before manifest commit")
        return original_replace(staged, destination)

    monkeypatch.setattr(source.os, "replace", interrupt_before_manifest)
    with pytest.raises(OSError, match="simulated interruption"):
        source.refresh_cms_partd_source()
    assert source.current_source_path() == first_source
    assert source.current_manifest_path().read_bytes() == first_manifest
    assert pd.read_csv(source.current_source_path())["demand_claims"].tolist() == [10, 20]


def test_source_snapshot_rejects_changed_bytes_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "runtime"))
    bundled = tmp_path / "bundled.csv.gz"
    bundled.write_bytes(b"verified source bytes")
    digest = hashlib.sha256(bundled.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"normalized_sha256": digest}))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_PATH", str(bundled))
    monkeypatch.setattr(settings, "CMS_PARTD_SOURCE_MANIFEST_PATH", str(manifest))
    assert source.source_snapshot()[0] == bundled
    bundled.write_bytes(b"changed source bytes")
    with pytest.raises(ValueError, match="checksum disagrees"):
        source.source_snapshot()

    runtime = source.runtime_manifest_path()
    runtime.parent.mkdir(parents=True)
    committed = runtime.parent / f"cms_partd_claims.{digest}.csv.gz"
    committed.write_bytes(b"changed source bytes")
    runtime.write_text(json.dumps({"source_file": committed.name,
                                   "normalized_sha256": digest}))
    with pytest.raises(ValueError, match="checksum disagrees"):
        source.source_snapshot()
