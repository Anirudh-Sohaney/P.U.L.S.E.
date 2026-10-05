"""Encrypted snapshots round-trip and reject altered ciphertext or wrong keys."""

import gzip
import os
import sqlite3
import stat

import pytest

from scripts.backup_databases import create_backup, verify_backup
from scripts.encrypted_backup import export_encrypted, restore_encrypted
from backend.backup_worker import ensure_encrypted_export


def _snapshot(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    for name in ("accounts.sqlite3", "signals.sqlite3"):
        with sqlite3.connect(source / name) as db:
            db.execute("CREATE TABLE sample (value TEXT)")
            db.execute("INSERT INTO sample VALUES (?)", ("private account row",))
    cms = tmp_path / "cms.csv.gz"
    cms.write_bytes(gzip.compress(b"year,state,drug_key,demand_claims\n2024,AR,A,2\n"))
    return create_backup(source, tmp_path / "backups", cms_source=cms,
                         cms_manifest={"years": [2024]})


def test_encrypted_backup_round_trip_and_private_permissions(tmp_path):
    snapshot = _snapshot(tmp_path)
    key = os.urandom(32)
    bundle = export_encrypted(snapshot, tmp_path / "export" / "snapshot.enc", key)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o600
    assert b"private account row" not in bundle.read_bytes()
    restored = tmp_path / "restored"
    manifest = restore_encrypted(bundle, restored, key)
    assert manifest == verify_backup(snapshot)
    assert verify_backup(restored) == manifest
    assert stat.S_IMODE(restored.stat().st_mode) == 0o700
    assert all(stat.S_IMODE((restored / name).stat().st_mode) == 0o600
               for name in manifest["files"])
    with sqlite3.connect(restored / "accounts.sqlite3") as db:
        assert db.execute("SELECT value FROM sample").fetchone() == (
            "private account row",)


def test_encrypted_backup_rejects_wrong_key_and_tampering(tmp_path):
    key = os.urandom(32)
    bundle = export_encrypted(_snapshot(tmp_path), tmp_path / "snapshot.enc", key)
    wrong_destination = tmp_path / "wrong-key"
    with pytest.raises(ValueError, match="authentication failed"):
        restore_encrypted(bundle, wrong_destination, os.urandom(32))
    assert not wrong_destination.exists()
    damaged = tmp_path / "damaged.enc"
    ciphertext = bytearray(bundle.read_bytes())
    ciphertext[len(ciphertext) // 2] ^= 1
    damaged.write_bytes(ciphertext)
    damaged_destination = tmp_path / "damaged"
    with pytest.raises(ValueError, match="authentication failed"):
        restore_encrypted(damaged, damaged_destination, key)
    assert not damaged_destination.exists()


def test_scheduled_export_is_verified_and_retries_after_failure(tmp_path):
    snapshot = _snapshot(tmp_path)
    key_file = tmp_path / "backup-key.hex"
    key_file.write_text(os.urandom(32).hex())
    outside = tmp_path / "offsite"
    bundle = ensure_encrypted_export(snapshot, tmp_path / "data", outside, key_file)
    assert ensure_encrypted_export(snapshot, tmp_path / "data", outside, key_file) == bundle
    damaged = bytearray(bundle.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    bundle.write_bytes(damaged)
    with pytest.raises(ValueError, match="authentication failed"):
        ensure_encrypted_export(snapshot, tmp_path / "data", outside, key_file)
    with pytest.raises(ValueError, match="outside the data directory"):
        ensure_encrypted_export(snapshot, tmp_path / "data",
                                tmp_path / "data" / "exports", key_file)
