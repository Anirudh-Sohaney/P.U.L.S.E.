"""Health checks for the long-running refresh and backup containers.

Run ``python -m scripts.operational_status worker``, ``backup``, or ``sources``
from website/.
The command emits JSON and exits nonzero when its component cannot perform its
daily duty. Compose uses the same checks for container health.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.backup_worker import latest_verified_backup
from backend.config import settings
from backend.signal_store import freshness
from backend.worker import REFRESH_TASKS, most_recent_schedule
from scripts.backup_databases import verify_backup
from scripts.encrypted_backup import load_key, verify_encrypted


def worker_status() -> dict:
    state = freshness()
    checked_at = state["worker_last_check_at"]
    return {
        "component": "worker",
        "healthy": bool(state["worker_recent"]),
        "last_heartbeat_at": checked_at,
        "reason": None if state["worker_recent"] else "refresh_worker_heartbeat_stale_or_missing",
        "failed_sources": state["failed_sources"],
    }


def sources_status(now: datetime | None = None) -> dict:
    state = freshness()
    checks = state["source_checks"]
    failed = state["failed_sources"]
    expected = {name for _, names in REFRESH_TASKS for name in names}
    observed = {row["source_name"] for row in checks}
    missing = sorted(expected - observed)
    cutoff = most_recent_schedule(now or datetime.now(timezone.utc))
    stale = []
    for row in checks:
        if row["source_name"] not in expected:
            continue
        try:
            attempted = datetime.fromisoformat(row["last_attempt_at"])
            if attempted.tzinfo is None or attempted < cutoff:
                stale.append(row["source_name"])
        except (TypeError, ValueError):
            stale.append(row["source_name"])
    return {
        "component": "sources",
        "healthy": not missing and not stale and not failed,
        "checked_source_count": len(checks),
        "failed_sources": failed,
        "missing_sources": missing,
        "stale_sources": sorted(stale),
        "reason": ("source_checks_missing" if missing else
                   "source_checks_stale" if stale else
                   "upstream_source_refresh_failed" if failed else None),
    }


def backup_status(now: datetime | None = None) -> dict:
    current = now or datetime.now(timezone.utc)
    snapshot = latest_verified_backup(Path(settings.DATA_PATH) / "backups", current)
    result = {
        "component": "backup",
        "healthy": snapshot is not None,
        "snapshot": snapshot.name if snapshot else None,
        "reason": None if snapshot else "verified_daily_snapshot_missing",
        "encrypted_export": None,
    }
    export_dir = os.getenv("PULSE_BACKUP_EXPORT_DIR", "")
    key_file = os.getenv("PULSE_BACKUP_KEY_FILE", "")
    if bool(export_dir) != bool(key_file):
        result.update(healthy=False, reason="incomplete_encrypted_backup_configuration")
    elif snapshot is not None and export_dir:
        destination = Path(export_dir) / f"{snapshot.name}.enc"
        if not destination.is_file():
            result.update(healthy=False, reason="encrypted_export_missing")
        else:
            try:
                if verify_encrypted(destination, load_key(Path(key_file))) != verify_backup(snapshot):
                    raise ValueError("Encrypted export does not match the daily snapshot")
            except (OSError, ValueError) as exc:
                result.update(healthy=False, reason=f"encrypted_export_invalid:{type(exc).__name__}")
            else:
                result["encrypted_export"] = destination.name
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=("worker", "backup", "sources"))
    args = parser.parse_args()
    try:
        report = {"worker": worker_status, "backup": backup_status,
                  "sources": sources_status}[args.component]()
    except Exception as exc:
        report = {"component": args.component, "healthy": False,
                  "reason": f"health_check_failed:{type(exc).__name__}"}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
