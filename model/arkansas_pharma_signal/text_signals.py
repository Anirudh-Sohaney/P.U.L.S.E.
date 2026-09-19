"""Deterministic text/event signal features.

Builds annual aggregate features from local news keyword counts, structured
drug-shortage events, FDA shortage records, and FDA enforcement/recall
records. No LLM calls: hashed/rule-based aggregation over real metadata.

Outputs:
- annual_features: [year, variable_id, value] long frame (news groups, event
  counts, shortage/recall severity, enforcement volume).
- drug_features: [year, drug(normalized), variable_id, value] long frame for
  per-drug shortage/recall exposure used by the panel.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pandas as pd

from . import io
from .entities import normalize_name


def build_text_features(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (annual_feature_frame, drug_feature_frame)."""
    rows: List[Dict] = []
    drug_rows: List[Dict] = []

    _news_rows(cfg, rows)
    _event_rows(cfg, rows, drug_rows)
    _shortage_rows(cfg, rows, drug_rows)
    _enforcement_rows(cfg, rows, drug_rows)

    annual = pd.DataFrame(rows)
    drugs = pd.DataFrame(drug_rows)
    if annual.empty:
        annual = pd.DataFrame(columns=["year", "variable_id", "value"])
    if drugs.empty:
        drugs = pd.DataFrame(columns=["year", "drug", "variable_id", "value"])
    return annual, drugs


def _news_rows(cfg, rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.news_dir) / "arkansas_3dlnews_monthly_keyword_counts.csv.gz"
    if not path.exists():
        return
    news = io.load_csv(path, usecols=["group", "year", "article_count"])
    agg = news.groupby(["year", "group"], as_index=False)["article_count"].sum()
    for _, r in agg.iterrows():
        rows.append(
            {
                "year": int(r["year"]),
                "variable_id": f"news_{r['group']}_articles",
                "value": float(r["article_count"]),
            }
        )


def _event_rows(cfg, rows: List[Dict], drug_rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.events)
    if not path.exists():
        return
    events = io.load_csv(
        path,
        usecols=["event_type", "start_time", "severity", "confidence",
                 "affected_drugs_json", "affected_entities_json"],
    )
    if events.empty:
        return
    years = pd.to_datetime(events["start_time"], errors="coerce").dt.year
    events["year"] = years
    events = events.dropna(subset=["year"])
    events["year"] = events["year"].astype(int)

    by_year = events.groupby("year")
    for year, sub in by_year:
        sev = pd.to_numeric(sub["severity"], errors="coerce")
        conf = pd.to_numeric(sub["confidence"], errors="coerce")
        rows.append({"year": year, "variable_id": "event_count", "value": float(len(sub))})
        if sev.notna().any():
            rows.append({"year": year, "variable_id": "event_severity_mean", "value": float(sev.mean())})
        if conf.notna().any():
            rows.append({"year": year, "variable_id": "event_confidence_mean", "value": float(conf.mean())})

        # Per-drug exposure from affected_drugs_json lists.
        for _, row in sub.iterrows():
            for drug in _json_list(row["affected_drugs_json"]):
                key = normalize_name(drug)
                if not key:
                    continue
                drug_rows.append({"year": year, "drug": key, "variable_id": "shortage_events", "value": 1.0})


def _shortage_rows(cfg, rows: List[Dict], drug_rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.fda_shortages)
    if not path.exists():
        return
    shortages = io.load_csv(
        path,
        usecols=["generic_name", "initial_posting_date", "status"],
    )
    shortages["year"] = (
        pd.to_datetime(shortages["initial_posting_date"], errors="coerce").dt.year
    )
    shortages = shortages.dropna(subset=["year"])
    shortages["year"] = shortages["year"].astype(int)
    if shortages.empty:
        return

    by_year = shortages.groupby("year")
    for year, sub in by_year:
        rows.append({"year": year, "variable_id": "shortage_active_total", "value": float(len(sub))})
        rows.append({"year": year, "variable_id": "shortage_current_total", "value": float((sub["status"] == "Current").sum())})
        for _, row in sub.iterrows():
            key = normalize_name(row["generic_name"])
            if key:
                drug_rows.append({"year": year, "drug": key, "variable_id": "shortage_active", "value": 1.0})


def _enforcement_rows(cfg, rows: List[Dict], drug_rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.fda_enforcement)
    if not path.exists():
        return
    enf = io.load_csv(
        path,
        usecols=["report_date", "product_description", "recalling_firm", "classification"],
    )
    enf["year"] = (
        pd.to_numeric(enf["report_date"], errors="coerce") // 10000
    ).astype("Int64")
    enf = enf.dropna(subset=["year"])
    enf["year"] = enf["year"].astype(int)
    if enf.empty:
        return

    by_year = enf.groupby("year")
    for year, sub in by_year:
        rows.append({"year": year, "variable_id": "recall_count", "value": float(len(sub))})
        rows.append({"year": year, "variable_id": "recall_firms", "value": float(sub["recalling_firm"].nunique())})
        for _, row in sub.iterrows():
            for key in _drug_keys_from_description(row["product_description"]):
                drug_rows.append({"year": year, "drug": key, "variable_id": "recall_count", "value": 1.0})


def _drug_keys_from_description(description: object) -> List[str]:
    """Pull a candidate drug name from an FDA recall product description."""
    if not isinstance(description, str):
        return []
    head = description.split(",")[0].strip()
    key = normalize_name(head)
    return [key] if key else []


def _json_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return []
    return []


def pivot_annual(annual: pd.DataFrame) -> pd.DataFrame:
    """Wide annual event/news feature table keyed by year."""
    if annual.empty:
        return pd.DataFrame(columns=["year"])
    wide = annual.pivot_table(
        index="year", columns="variable_id", values="value", aggfunc="sum"
    ).reset_index()
    wide.columns.name = None
    return wide


def pivot_drug(drug_features: pd.DataFrame) -> pd.DataFrame:
    """Wide per-drug annual features keyed by (year, drug)."""
    if drug_features.empty:
        return pd.DataFrame(columns=["year", "drug"])
    wide = drug_features.pivot_table(
        index=["year", "drug"], columns="variable_id", values="value",
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None
    return wide
