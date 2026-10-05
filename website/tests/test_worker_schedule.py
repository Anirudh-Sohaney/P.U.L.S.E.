from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from backend import signal_store, worker
from backend.config import settings


def test_refresh_worker_lock_rejects_a_second_process(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    environment = os.environ.copy()
    environment["DATA_PATH"] = str(tmp_path)
    with worker.exclusive_worker():
        assert stat.S_IMODE((tmp_path / ".refresh_worker.lock").stat().st_mode) == 0o600
        result = subprocess.run([sys.executable, "-m", "backend.worker", "--once"],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode != 0
        assert "Another PULSE refresh worker" in result.stderr
        with pytest.raises(RuntimeError, match="Another PULSE refresh worker"):
            with worker.exclusive_worker():
                pass
    with worker.exclusive_worker():
        pass


def utc(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc)


def test_schedule_boundary_and_next_run():
    assert worker.most_recent_schedule(utc(2, 10)) == utc(1, 16)
    assert worker.most_recent_schedule(utc(2, 16)) == utc(2, 16)
    assert worker.seconds_until_next_run(utc(2, 10)) == 6 * 60 * 60
    assert worker.seconds_until_next_run(utc(2, 16)) == 24 * 60 * 60


def test_wall_clock_sleep_wakes_after_simulated_laptop_suspend():
    current = [utc(2, 10)]
    pauses = []

    def pause(seconds):
        pauses.append(seconds)
        current[0] += timedelta(hours=5)

    worker.sleep_wall_clock(3600, clock=lambda: current[0], pause=pause)
    assert pauses == [60]


def test_catch_up_retries_missing_news_source_after_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        for source, status in (
            ("fda_drugs_rss", "success"),
            ("fda_medwatch_rss", "success"),
            ("bls_public_api_v1", "success"),
            ("cms_geographic_variation_api", "failed"),
        ):
            db.execute("""INSERT INTO refresh_runs
                (source_name, started_at, finished_at, status)
                VALUES (?, ?, ?, ?)""",
                (source, utc(1, 16).isoformat(), utc(1, 16, 5).isoformat(), status))

    due_before = {task.__name__ for task in worker.due_tasks(utc(2, 10))}
    assert "refresh_gdelt_news" in due_before
    assert "refresh_fda_medwatch_news" not in due_before
    assert "refresh_bls" not in due_before
    assert "refresh_cms_geo" in due_before
    assert len(due_before) == len(worker.REFRESH_TASKS) - 3

    due_after = {task.__name__ for task in worker.due_tasks(utc(2, 17))}
    assert len(due_after) == len(worker.REFRESH_TASKS)

    with signal_store.connect() as db:
        for source in ("bls_public_api_v1", "openfda_shortages", "gdelt_recent_news"):
            db.execute("""INSERT INTO refresh_runs
                (source_name, started_at, finished_at, status)
                VALUES (?, ?, ?, 'failed')""",
                (source, utc(2, 16).isoformat(), utc(2, 16, 5).isoformat()))
        db.execute("""INSERT INTO refresh_runs
            (source_name, started_at, finished_at, status)
            VALUES ('fda_drugs_rss', ?, ?, 'success')""",
            (utc(2, 16).isoformat(), utc(2, 16, 5).isoformat()))
        db.execute("""INSERT INTO refresh_runs
            (source_name, started_at, finished_at, status)
            VALUES ('fda_medwatch_rss', ?, ?, 'success')""",
            (utc(2, 16).isoformat(), utc(2, 16, 5).isoformat()))

    due_during_cooldown = {task.__name__ for task in worker.due_tasks(utc(2, 17))}
    assert "refresh_bls" not in due_during_cooldown
    assert "refresh_shortages" not in due_during_cooldown
    assert "refresh_gdelt_news" not in due_during_cooldown
    assert "refresh_fda_drugs_news" not in due_during_cooldown
    assert "refresh_fda_medwatch_news" not in due_during_cooldown
    with signal_store.connect() as db:
        db.execute("DELETE FROM refresh_runs WHERE source_name='fda_medwatch_rss'")
    independent_due = {task.__name__ for task in worker.due_tasks(utc(2, 17))}
    assert "refresh_fda_medwatch_news" in independent_due
    assert "refresh_gdelt_news" not in independent_due
    due_after_cooldown = {task.__name__ for task in worker.due_tasks(utc(2, 23))}
    assert "refresh_shortages" in due_after_cooldown
    assert "refresh_gdelt_news" in due_after_cooldown
    assert "refresh_fda_drugs_news" not in due_after_cooldown
    assert "refresh_fda_medwatch_news" in due_after_cooldown
    assert "refresh_bls" not in due_after_cooldown
    due_next_day = {task.__name__ for task in worker.due_tasks(utc(3, 16))}
    assert "refresh_bls" in due_next_day


def test_baseline_reruns_when_cms_source_succeeds_after_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    with signal_store.connect() as db:
        for source, day, hour in (
            ("cms_partd_two_year_persistence_v2", 2, 16),
            ("cms_partd_catalog_source", 2, 17),
        ):
            db.execute("""INSERT INTO refresh_runs
                (source_name, started_at, finished_at, status)
                VALUES (?, ?, ?, 'success')""",
                (source, utc(day, hour).isoformat(), utc(day, hour, 5).isoformat()))
    due = {task.__name__ for task in worker.due_tasks(utc(2, 18))}
    assert "refresh_cms_partd_source" not in due
    assert "refresh_demand_baseline" in due

    with signal_store.connect() as db:
        db.execute("""INSERT INTO refresh_runs
            (source_name, started_at, finished_at, status)
            VALUES ('cms_partd_two_year_persistence_v2', ?, ?, 'success')""",
            (utc(2, 18).isoformat(), utc(2, 18, 5).isoformat()))
    due = {task.__name__ for task in worker.due_tasks(utc(2, 19))}
    assert "refresh_demand_baseline" not in due
