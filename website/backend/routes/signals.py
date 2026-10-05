"""Public, versioned access to observed and model-derived signal values."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .. import signal_store

router = APIRouter()
SignalId = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class LatestQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SignalId] = Field(min_length=1, max_length=1500)


class SignalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SignalId] = Field(min_length=1, max_length=100)
    observation_date: date | None = Field(default=None, alias="date")
    start_date: date | None = None
    end_date: date | None = None
    include_revisions: bool = False


def _validate_dates(query: SignalQuery) -> tuple[date | None, date | None]:
    if query.observation_date and (query.start_date or query.end_date):
        raise HTTPException(422, detail="Use date or start_date/end_date, not both")
    start = query.observation_date or query.start_date
    end = query.observation_date or query.end_date
    if start and end and start > end:
        raise HTTPException(422, detail="start_date must not exceed end_date")
    return start, end


@router.get("/catalog")
def catalog(search: str = Query("", max_length=120), limit: int = Query(1500, ge=1, le=1500),
            offset: int = Query(0, ge=0)):
    """List stable signal IDs, identity, units, source, and last observation."""
    rows = signal_store.list_signals(search=search.strip(), limit=limit, offset=offset)
    return {"signals": rows, "count": len(rows), "offset": offset,
            "note": "State values are ordinal categories, not medication units or purchase quantities."}


@router.post("/latest")
def latest(query: LatestQuery):
    """Return usable values and the absolute latest recorded row for each ID."""
    rows = signal_store.values(query.ids, latest_only=True)
    recorded = signal_store.values(query.ids, latest_only=True, include_unusable=True)
    found = {row["id"] for row in rows}
    missing_ids = [uid for uid in query.ids if uid not in found]
    archived = [row for row in recorded if row["id"] in missing_ids and not row["usable"]]
    coverage = ({row["id"]: row for row in signal_store.gap_report()["signals"]}
                if missing_ids else {})
    missing_details = [{"id": uid,
                        "reason": coverage[uid]["gap_status"] if uid in coverage else "unknown_id",
                        "last_recorded_period": (coverage[uid]["latest_observation_date"]
                                                 if uid in coverage else None)}
                       for uid in missing_ids]
    return {"values": rows, "latest_recorded_values": recorded,
            "unusable_recorded_values": archived,
            "missing_ids": missing_ids,
            "missing_details": missing_details,
            "ambiguous_ids": sorted({row["id"] for row in [*rows, *recorded]
                                     if row["ambiguous"]})}


@router.post("/history")
def history(query: SignalQuery):
    """Return exact recorded observations on one date or in a date range."""
    start, end = _validate_dates(query)
    try:
        rows = signal_store.values(query.ids, start=start, end=end,
                                   include_revisions=query.include_revisions)
    except signal_store.HistoryTooLarge as exc:
        raise HTTPException(413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return {"values": rows, "count": len(rows), "start_date": start,
            "end_date": end, "note": "Dates without a recorded observation are absent, not zero. Multiple conflicting values for one date are marked ambiguous."}


@router.get("/freshness")
def freshness():
    """Expose data coverage so callers can detect stale or missing feeds."""
    return signal_store.freshness()


@router.get("/coverage")
def coverage():
    return signal_store.gap_report()


@router.get("/demand/drugs")
def demand_drugs(search: str = Query("", max_length=120), limit: int = Query(100, ge=1, le=250),
                 offset: int = Query(0, ge=0)):
    return signal_store.demand_drugs(search=search.strip(), limit=limit, offset=offset)


@router.get("/news/recent")
def recent_news(days: int = Query(3, ge=1, le=7)):
    rows = signal_store.recent_news(days=days)
    source_checks = signal_store.news_source_checks()
    return {"articles": rows, "count": len(rows), "window_days": days,
            "source_checks": source_checks,
            "note": "Keyword relevance is context only; news is not model attribution or proof of demand."}


@router.get("/recent")
def recent_public_signals(days: int = Query(3, ge=1, le=7),
                          limit: int = Query(12, ge=1, le=20)):
    rows = signal_store.recent_public_signals(days=days, limit=limit)
    return {"signals": rows, "count": len(rows), "window_days": days,
            "note": "These are newly recorded official observations with unresolved same-revision conflicts omitted. Their observation periods may be older than the recording date; they are context, not drug-demand forecasts."}


@router.get("/sources/shortages/recent")
def recent_shortages(days: int = Query(3, ge=1, le=7)):
    return signal_store.recent_shortage_changes(days=days)


@router.get("/sources/nadac/latest")
def latest_nadac():
    return signal_store.latest_nadac_snapshot()


@router.get("/sources/sdud/latest")
def latest_sdud(search: str = Query("", max_length=120), limit: int = Query(100, ge=1, le=250),
                offset: int = Query(0, ge=0)):
    """Observed Arkansas Medicaid prescriptions; suppressed cells remain unknown."""
    return signal_store.latest_sdud_snapshot(search=search.strip(), limit=limit, offset=offset)
