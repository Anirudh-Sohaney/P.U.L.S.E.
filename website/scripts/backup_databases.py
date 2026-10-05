"""Consistent, owner-only SQLite snapshots for the single-host PULSE service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings
from backend.refresh_cms_partd_source import source_snapshot


DATABASES = ("accounts.sqlite3", "signals.sqlite3")
SNAPSHOT_NAME = re.compile(r"^\d{8}T\d{6}\.\d{6}Z$")
CMS_SOURCE_NAME = re.compile(r"^cms_partd_claims\.[0-9a-f]{64}\.csv\.gz$")


class BackupAlreadyRunning(RuntimeError):
    """Another process is creating or pruning snapshots in this directory."""


@contextmanager
def exclusive_backup(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    descriptor = os.open(root / ".backup.lock",
                         os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupAlreadyRunning("Another PULSE backup holds the snapshot-directory lock") from exc
        yield
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_backup(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not set(DATABASES).issubset(files):
        raise ValueError("Backup manifest does not list both databases")
    extras = set(files) - set(DATABASES)
    if extras and (len(extras) != 2 or "cms_partd_source_manifest.json" not in extras
                   or not any(CMS_SOURCE_NAME.fullmatch(name) for name in extras)):
        raise ValueError("Backup manifest has an incomplete CMS source pair")
    for name in files:
        path = directory / name
        details = files[name]
        actual_digest = _digest(path)
        if path.stat().st_size != details["bytes"] or actual_digest != details["sha256"]:
            raise ValueError(f"Backup checksum mismatch: {name}")
        if CMS_SOURCE_NAME.fullmatch(name) and name.split(".")[1] != actual_digest:
            raise ValueError("Backup CMS source filename does not match its content")
    if extras:
        cms_manifest = json.loads((directory / "cms_partd_source_manifest.json").read_text(
            encoding="utf-8"))
        cms_name = cms_manifest.get("source_file")
        if cms_name not in extras:
            raise ValueError("Backup CMS manifest does not point to its source file")
        if (cms_manifest.get("normalized_sha256") is not None
                and cms_manifest["normalized_sha256"] != files[cms_name]["sha256"]):
            raise ValueError("Backup CMS manifest checksum does not match its source")
    for name in DATABASES:
        path = directory / name
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"Backup SQLite integrity check failed: {name}")
            if name == "signals.sqlite3":
                has_runs = db.execute("""SELECT 1 FROM sqlite_master
                    WHERE type='table' AND name='demand_forecast_runs'""").fetchone()
                if has_runs:
                    run = db.execute("""SELECT source_sha256, source_year
                        FROM demand_forecast_runs
                        WHERE method='cms_partd_two_year_persistence_v2'
                        ORDER BY target_year DESC, created_at DESC LIMIT 1""").fetchone()
                    if run and not extras:
                        raise ValueError("Backup ranking has no committed CMS source pair")
                    if run and (run[0] != files[cms_name]["sha256"]
                                or run[1] != max(cms_manifest.get("years") or [None])):
                        raise ValueError("Backup ranking and CMS source generation disagree")
    return manifest


def _create_backup_unlocked(source_dir: Path, backup_root: Path, *, keep: int,
                            now: datetime | None, cms_source: Path | None,
                            cms_manifest: dict | None) -> Path:
    if keep < 1:
        raise ValueError("keep must be at least one snapshot")
    if (cms_source is None) != (cms_manifest is None):
        raise ValueError("CMS source and manifest must be supplied together")
    backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup_root.chmod(0o700)
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    destination = backup_root / stamp
    temporary = backup_root / f".{stamp}.{os.getpid()}.tmp"
    if destination.exists():
        raise FileExistsError(destination)
    temporary.mkdir(mode=0o700)
    try:
        files = {}
        for name in DATABASES:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = temporary / name
            with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as source_db:
                with closing(sqlite3.connect(target)) as backup_db:
                    source_db.backup(backup_db)
            target.chmod(0o600)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            files[name] = {"bytes": target.stat().st_size, "sha256": _digest(target)}
        if cms_source is not None and cms_manifest is not None:
            cms_hash = _digest(cms_source)
            cms_name = f"cms_partd_claims.{cms_hash}.csv.gz"
            target = temporary / cms_name
            with cms_source.open("rb") as original, target.open("xb") as copy:
                shutil.copyfileobj(original, copy)
                copy.flush()
                os.fsync(copy.fileno())
            target.chmod(0o600)
            files[cms_name] = {"bytes": target.stat().st_size, "sha256": _digest(target)}
            if files[cms_name]["sha256"] != cms_hash:
                raise ValueError("CMS source changed while it was being backed up")
            committed_manifest = {**cms_manifest, "source_file": cms_name,
                                  "normalized_output": cms_name}
            cms_manifest_path = temporary / "cms_partd_source_manifest.json"
            descriptor = os.open(cms_manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                                 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(committed_manifest, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            files[cms_manifest_path.name] = {"bytes": cms_manifest_path.stat().st_size,
                                             "sha256": _digest(cms_manifest_path)}
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": files}
        manifest_path = temporary / "manifest.json"
        descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_backup(temporary)
        _sync_directory(temporary)
        os.replace(temporary, destination)
        _sync_directory(backup_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    snapshots = sorted(path for path in backup_root.iterdir()
                       if path.is_dir() and SNAPSHOT_NAME.fullmatch(path.name))
    for old in snapshots[:-keep]:
        if old.resolve().parent != backup_root.resolve():
            raise ValueError(f"Backup retention target escaped destination: {old}")
        shutil.rmtree(old)
    _sync_directory(backup_root)
    return destination


def create_backup(source_dir: Path, backup_root: Path, *, keep: int = 14,
                  now: datetime | None = None,
                  cms_source: Path | None = None, cms_manifest: dict | None = None) -> Path:
    """Create and prune snapshots under one cross-process directory lock."""
    with exclusive_backup(backup_root):
        return _create_backup_unlocked(source_dir, backup_root, keep=keep,
                                       now=now, cms_source=cms_source,
                                       cms_manifest=cms_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(settings.DATA_PATH))
    parser.add_argument("--destination", type=Path,
                        default=Path(settings.DATA_PATH) / "backups")
    parser.add_argument("--keep", type=int, default=14)
    parser.add_argument("--verify", type=Path, help="Verify an existing snapshot")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_backup(args.verify), indent=2))
    else:
        cms_source, cms_manifest = source_snapshot()
        path = create_backup(args.source, args.destination, keep=args.keep,
                             cms_source=cms_source, cms_manifest=cms_manifest)
        print(path)


if __name__ == "__main__":
    try:
        main()
    except BackupAlreadyRunning as exc:
        raise SystemExit(str(exc)) from None
