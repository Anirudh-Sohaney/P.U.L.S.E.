"""Separate single-instance daily source refresh worker (16:00 UTC)."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from pathlib import Path

from .config import settings
from .refresh_public import (refresh_gdelt_news, refresh_fda_drugs_news,
                             refresh_fda_medwatch_news)
from .refresh_shortages import refresh_shortages
from .refresh_bls import refresh_bls
from .refresh_medicaid import refresh_medicaid
from .refresh_cms_geo import refresh_cms_geo
from .refresh_nadac import refresh_nadac
from .refresh_sdud import refresh_sdud
from .refresh_cms_partd_source import refresh_cms_partd_source
from .refresh_hhs_ndc_release import refresh_hhs_ndc_release
from .refresh_demand_baseline import refresh_demand_baseline
from .signal_store import connect, initialize, record_worker_heartbeat


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
SCHEDULE_HOUR_UTC = 16
RETRY_POLL_SECONDS = 60 * 60
RETRY_DELAY = timedelta(hours=6)
MAX_SLEEP_SLICE_SECONDS = 60
REFRESH_TASKS: tuple[tuple[Callable[[], dict], tuple[str, ...]], ...] = (
    (refresh_gdelt_news, ("gdelt_recent_news",)),
    (refresh_fda_drugs_news, ("fda_drugs_rss",)),
    (refresh_fda_medwatch_news, ("fda_medwatch_rss",)),
    (refresh_shortages, ("openfda_shortages",)),
    (refresh_bls, ("bls_public_api_v1",)),
    (refresh_medicaid, ("medicaid_state_performance_api",)),
    (refresh_cms_geo, ("cms_geographic_variation_api",)),
    (refresh_nadac, ("medicaid_nadac_2026_api",)),
    (refresh_sdud, ("medicaid_sdud_arkansas",)),
    (refresh_cms_partd_source, ("cms_partd_catalog_source",)),
    (refresh_hhs_ndc_release, ("hhs_ndc_release_catalog",)),
    (refresh_demand_baseline, ("cms_partd_two_year_persistence_v2",)),
)


class WorkerAlreadyRunning(RuntimeError):
    """A second worker attempted to use the same data directory."""


@contextmanager
def exclusive_worker():
    """Hold one OS lock across the full refresh process on a shared data mount."""
    directory = Path(settings.DATA_PATH)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".refresh_worker.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                         0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkerAlreadyRunning(
                "Another PULSE refresh worker holds the data-directory lock") from exc
        yield
    finally:
        os.close(descriptor)


def seconds_until_next_run(now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    target = current.replace(hour=SCHEDULE_HOUR_UTC, minute=0, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return (target - current).total_seconds()


def most_recent_schedule(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    target = current.replace(hour=SCHEDULE_HOUR_UTC, minute=0, second=0, microsecond=0)
    return target if target <= current else target - timedelta(days=1)


def due_tasks(now: datetime | None = None) -> list[Callable[[], dict]]:
    """Catch up missed jobs and retry failures within each source's safe budget."""
    initialize()
    current = now or datetime.now(timezone.utc)
    cutoff = most_recent_schedule(current).isoformat()
    with connect() as db:
        successful_at = {row["source_name"]: datetime.fromisoformat(row["last_success"])
                         for row in db.execute("""SELECT source_name,
                             MAX(finished_at) AS last_success FROM refresh_runs
                             WHERE status='success' AND finished_at>=?
                             GROUP BY source_name""", (cutoff,))}
        failed_at = {row["source_name"]: datetime.fromisoformat(row["last_failure"])
                     for row in db.execute("""SELECT source_name, MAX(finished_at) AS last_failure
                         FROM refresh_runs WHERE status='failed' AND finished_at>=?
                         GROUP BY source_name""", (cutoff,))}
    due = []
    for refresh, names in REFRESH_TASKS:
        last_success = max((successful_at[name] for name in names if name in successful_at),
                           default=None)
        all_sources_succeeded = all(name in successful_at for name in names)
        source_newer_than_baseline = (refresh is refresh_demand_baseline
            and last_success is not None
            and successful_at.get("cms_partd_catalog_source", last_success) > last_success)
        if all_sources_succeeded and not source_newer_than_baseline:
            continue
        latest_failure = max((failed_at[name] for name in names if name in failed_at),
                             default=None)
        if latest_failure is not None:
            # The unregistered BLS API permits only 25 queries per day. A
            # partial refresh can already have spent nine, so wait for the
            # next daily schedule before another attempt.
            if refresh is refresh_bls or current - latest_failure < RETRY_DELAY:
                continue
        due.append(refresh)
    return due


def run_tasks(refreshes: list[Callable[[], dict]]) -> list[dict]:
    results = []
    for refresh in refreshes:
        try:
            results.append(refresh())
        except Exception as exc:
            logger.exception("Source refresh failed: %s", refresh.__name__)
            results.append({"source": refresh.__name__, "status": "failed", "error": str(exc)})
    return results


def sleep_wall_clock(seconds: float, *, clock: Callable[[], datetime] | None = None,
                     pause: Callable[[float], None] | None = None) -> None:
    """Wake promptly after laptop suspend, which can pause monotonic sleeps."""
    read_clock = clock or (lambda: datetime.now(timezone.utc))
    sleep = pause or time.sleep
    target = read_clock() + timedelta(seconds=seconds)
    while True:
        remaining = (target - read_clock()).total_seconds()
        if remaining <= 0:
            return
        sleep(min(remaining, MAX_SLEEP_SLICE_SECONDS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one refresh immediately")
    args = parser.parse_args()
    with exclusive_worker():
        if args.once:
            print(json.dumps(run_tasks([refresh for refresh, _ in REFRESH_TASKS]), indent=2))
            return
        while True:
            try:
                record_worker_heartbeat()
                due = due_tasks()
                if due:
                    logger.info("Refreshing %d due sources", len(due))
                    logger.info("Refresh result: %s", run_tasks(due))
                record_worker_heartbeat()
            except Exception:
                logger.exception("Could not determine due source refreshes")
            delay = min(seconds_until_next_run(), RETRY_POLL_SECONDS)
            logger.info("Next source check in %.0f seconds", delay)
            sleep_wall_clock(delay)


if __name__ == "__main__":
    try:
        main()
    except WorkerAlreadyRunning as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None
