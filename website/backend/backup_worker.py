"""Daily, verified snapshots for the Compose deployment (04:00 UTC)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import settings
from .refresh_cms_partd_source import source_snapshot
from scripts.backup_databases import (
    SNAPSHOT_NAME, create_backup, verify_backup,
)
from scripts.encrypted_backup import export_encrypted, load_key, verify_encrypted


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
BACKUP_HOUR_UTC = 4
RETRY_POLL_SECONDS = 60 * 60
MAX_SLEEP_SLICE_SECONDS = 60


def most_recent_schedule(now: datetime) -> datetime:
    scheduled = now.astimezone(timezone.utc).replace(
        hour=BACKUP_HOUR_UTC, minute=0, second=0, microsecond=0)
    return scheduled if scheduled <= now else scheduled - timedelta(days=1)


def latest_verified_backup(root: Path, now: datetime) -> Path | None:
    cutoff = most_recent_schedule(now).strftime("%Y%m%dT%H%M%S.%fZ")
    if not root.exists():
        return None
    snapshots = sorted((path for path in root.iterdir()
                        if path.is_dir() and SNAPSHOT_NAME.fullmatch(path.name)
                        and path.name >= cutoff), reverse=True)
    for snapshot in snapshots:
        try:
            manifest = verify_backup(snapshot)
            if "cms_partd_source_manifest.json" not in manifest["files"]:
                raise ValueError("Backup does not contain the CMS source pair")
            return snapshot
        except Exception:
            logger.exception("Backup verification failed: %s", snapshot)
    return None


def backup_due(root: Path, now: datetime) -> bool:
    return latest_verified_backup(root, now) is None


def ensure_encrypted_export(snapshot: Path, source_dir: Path,
                            export_dir: Path, key_file: Path) -> Path:
    """Export once per verified snapshot; retry after an interrupted export."""
    source_root = source_dir.resolve()
    if export_dir.resolve().is_relative_to(source_root):
        raise ValueError("Encrypted export directory must be outside the data directory")
    if key_file.resolve().is_relative_to(source_root):
        raise ValueError("Backup encryption key must be outside the data directory")
    key = load_key(key_file)
    export_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = export_dir / f"{snapshot.name}.enc"
    source_manifest = verify_backup(snapshot)
    if destination.exists():
        if verify_encrypted(destination, key) != source_manifest:
            raise ValueError("Existing encrypted backup does not match snapshot")
        return destination
    return export_encrypted(snapshot, destination, key)


def sleep_wall_clock(seconds: float) -> None:
    """Resume after host suspend without waiting on a stale monotonic timer."""
    target = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    while True:
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, MAX_SLEEP_SLICE_SECONDS))


def main() -> None:
    source_dir = Path(settings.DATA_PATH)
    backup_root = source_dir / "backups"
    export_dir_setting = os.getenv("PULSE_BACKUP_EXPORT_DIR", "")
    key_file_setting = os.getenv("PULSE_BACKUP_KEY_FILE", "")
    if bool(export_dir_setting) != bool(key_file_setting):
        raise RuntimeError("Set both PULSE_BACKUP_EXPORT_DIR and PULSE_BACKUP_KEY_FILE")
    while True:
        now = datetime.now(timezone.utc)
        try:
            path = latest_verified_backup(backup_root, now)
            if path is None:
                cms_source, cms_manifest = source_snapshot()
                path = create_backup(source_dir, backup_root, keep=14,
                                     cms_source=cms_source, cms_manifest=cms_manifest)
                logger.info("Verified backup committed: %s", path)
            if export_dir_setting:
                exported = ensure_encrypted_export(
                    path, source_dir, Path(export_dir_setting), Path(key_file_setting))
                logger.info("Encrypted backup verified: %s", exported)
        except Exception:
            logger.exception("Backup failed; will retry")
        next_schedule = most_recent_schedule(now) + timedelta(days=1)
        delay = min((next_schedule - datetime.now(timezone.utc)).total_seconds(),
                    RETRY_POLL_SECONDS)
        sleep_wall_clock(max(1, delay))


if __name__ == "__main__":
    main()
