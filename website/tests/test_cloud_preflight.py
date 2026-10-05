"""Cloud startup must reject LFS pointers and unreadable deployment inputs."""

import gzip
import hashlib
import json
import os

import pytest

from scripts.cloud_preflight import check


def _inputs(tmp_path):
    offsite = tmp_path / "offsite"
    offsite.mkdir(mode=0o700)
    signing = tmp_path / "signing"
    signing.write_text("strong-signing-key-for-an-isolated-test-123456")
    signing.chmod(0o600)
    backup = tmp_path / "backup"
    backup.write_text("ab" * 32)
    backup.chmod(0o600)
    source = tmp_path / "source.csv.gz"
    source.write_bytes(gzip.compress(b"year,state,drug_key,demand_claims\n2024,arkansas,Drug A,1\n"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"normalized_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
    return {"domain": "pulse.example.org", "offsite": offsite,
            "signing_key": signing, "backup_key": backup,
            "source": source, "manifest": manifest,
            "container_uid": os.getuid(), "container_gid": os.getgid()}


def test_cloud_preflight_accepts_hydrated_source_and_private_inputs(tmp_path):
    result = check(**_inputs(tmp_path))
    assert result["ready_for_compose"] is True
    assert result["cms_source_sha256"]


def test_cloud_preflight_rejects_unhydrated_lfs_pointer(tmp_path):
    values = _inputs(tmp_path)
    values["source"].write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="hydrate Git LFS"):
        check(**values)


def test_cloud_preflight_rejects_public_backup_key(tmp_path):
    values = _inputs(tmp_path)
    values["backup_key"].chmod(0o644)
    with pytest.raises(ValueError, match="accessible to others"):
        check(**values)


def test_cloud_preflight_rejects_data_directory_as_offsite(tmp_path, monkeypatch):
    values = _inputs(tmp_path)
    monkeypatch.setattr("scripts.cloud_preflight.WEBSITE", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    values["offsite"] = data
    with pytest.raises(ValueError, match="outside website/data"):
        check(**values)
