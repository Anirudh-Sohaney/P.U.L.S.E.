"""Release-safe annual candidates from source-native FluView observations.

This bridge is deliberately separate from the saved model's feature builder:
the older national features use issue year, and the Arkansas weighted-ILI
feature has no published state series. A candidate here is not permission to
run that saved checkpoint; it is an input for a corrected retraining path.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SOURCE_KIND = "delphi_fluview_ilinet_v5"
FEATURE_SERIES = {
    "ar_ili_mean": ("fluview_ili", "AR"),
    "nat_ili_mean": ("fluview_ili", "US"),
    "nat_wili_mean": ("fluview_wili", "US"),
}


def _utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("FluView timestamps must have a timezone")
    return parsed.astimezone(timezone.utc)


def _saturdays(year: int) -> list[date]:
    first = date(year, 1, 1)
    current = first + timedelta(days=(5 - first.weekday()) % 7)
    days = []
    while current.year == year:
        days.append(current)
        current += timedelta(days=7)
    return days


def annual_candidate_from_rows(rows: list[dict[str, Any]], *, year: int,
                               as_of: str | datetime) -> dict[str, Any]:
    """Choose the latest known revision for every observed Saturday in a year.

    Both source publication and local recording must precede ``as_of``. Full
    weekly coverage is required; an incomplete year never receives a mean.
    """
    if year < 1990 or year > 2100:
        raise ValueError("year is outside the supported calendar range")
    cutoff = _utc(as_of)
    expected = _saturdays(year)
    expected_set = set(expected)
    selected: dict[tuple[str, date], tuple[datetime, float]] = {}
    for row in rows:
        identity = (str(row["signal_id"]), str(row["geography_id"]))
        if identity not in FEATURE_SERIES.values():
            continue
        observed = date.fromisoformat(str(row["observation_date"]))
        if observed.year != year:
            continue
        if observed not in expected_set:
            raise ValueError("FluView observation is not a Saturday")
        published = _utc(row["source_timestamp"])
        recorded = _utc(row["ingested_at"])
        if published < datetime.combine(observed, datetime.min.time(), timezone.utc):
            raise ValueError("FluView release precedes its observation week")
        value = float(row["value"])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("FluView percentage is invalid")
        if published > cutoff or recorded > cutoff:
            continue
        key = (f"{identity[0]}:{identity[1]}", observed)
        previous = selected.get(key)
        if previous and previous[0] == published and previous[1] != value:
            raise ValueError("Conflicting FluView values for one source release")
        if previous is None or published > previous[0]:
            selected[key] = (published, value)

    coverage = {}
    candidate = {}
    complete = True
    for feature, (signal_id, geography) in FEATURE_SERIES.items():
        name = f"{signal_id}:{geography}"
        missing = [week.isoformat() for week in expected if (name, week) not in selected]
        values = [selected[(name, week)][1] for week in expected if (name, week) in selected]
        releases = [selected[(name, week)][0] for week in expected if (name, week) in selected]
        coverage[feature] = {
            "weeks_observed": len(values), "weeks_expected": len(expected),
            "missing_weeks": missing,
            "latest_source_release": max(releases).isoformat() if releases else None,
        }
        if missing:
            complete = False
        else:
            candidate[feature] = sum(values) / len(values)
    return {
        "year": year, "as_of": cutoff.isoformat(),
        "status": "complete_source_year" if complete else "incomplete_source_year",
        "candidate_features": candidate if complete else {},
        "coverage": coverage,
        "model_compatible": False,
        "model_compatibility_reason": (
            "saved annual FluView time alignment differs and Arkansas weighted ILI is unpublished; "
            "rebuild features and retrain before model use"
        ),
    }


def annual_candidate_from_signal_store(db_path: Path, *, year: int,
                                       as_of: str | datetime) -> dict[str, Any]:
    """Read only the three captured source-native FluView series from SQLite."""
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("""
            SELECT d.signal_id, d.geography_id, o.observation_date,
                   o.source_timestamp, o.ingested_at, o.value
            FROM signal_records o JOIN signal_definitions d ON d.id=o.signal_uid
            WHERE o.source_kind=? AND o.observation_date>=?
              AND o.observation_date<?
              AND ((d.signal_id='fluview_ili' AND d.geography_id IN ('AR','US'))
                OR (d.signal_id='fluview_wili' AND d.geography_id='US'))
        """, (SOURCE_KIND, f"{year}-01-01", f"{year + 1}-01-01"))]
    return annual_candidate_from_rows(rows, year=year, as_of=as_of)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    args = parser.parse_args()
    print(json.dumps(annual_candidate_from_signal_store(
        args.database, year=args.year, as_of=args.as_of), indent=2))


if __name__ == "__main__":
    main()
