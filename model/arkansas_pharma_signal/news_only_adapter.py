"""Adapter for the separately trained news-only SLM signal surface.

The source model lives in ``existing_models/news_signal_model`` and remains
independently reproducible. This adapter imports only its dated, 20-column
feature output; it never imports a validation target or replaces the existing
article-event layer. Annual sums are joined to the multi-model panel at year
``t`` and therefore can only inform a ``t+1`` target through the existing
chronological training/evaluation code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from . import io

NEWS_ONLY_SIGNAL_IDS = (
    "arkansas_anti_infective_disruption",
    "arkansas_anti_infective_supply_chain",
    "arkansas_anti_infective_supply_policy_pressure",
    "arkansas_anti_infective_supply_disruption",
    "arkansas_anti_infective_national_supply",
    "arkansas_anesthesia_disruption",
    "arkansas_anesthesia_policy",
    "arkansas_anesthesia_supply_chain",
    "arkansas_antiinfective_anesthesia_supply",
    "arkansas_anti_infective_national_policy",
    "arkansas_opioid_policy",
    "arkansas_opioid_demand",
    "arkansas_anti_infective_anesthesia_policy",
    "arkansas_anti_infective_national_demand",
    "arkansas_anti_infective_composite_burden",
    "national_policy_risk",
    "national_demand_risk",
    "national_supply_chain_risk",
    "national_policy_direction",
    "national_supply_stage_direction",
)
NEWS_ONLY_PREFIX = "news_only_"


def _source_path(cfg, source: Optional[Path] = None) -> Path:
    """Resolve an explicit source or the configured repository-relative CSV."""
    if source is not None:
        return Path(source).expanduser().resolve()
    artifact = cfg.artifact_path("news/news_only_catalog_features.csv.gz")
    if artifact.exists():
        return artifact
    return cfg.repo_root / cfg.news_only_features


def load_news_only_features(cfg, source: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate dated news-only SLM features from a CSV artifact."""
    path = _source_path(cfg, source)
    frame = io.load_csv(path)
    expected = ["date", *NEWS_ONLY_SIGNAL_IDS]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"news-only schema mismatch; missing={missing}, extra={extra}")
    frame = frame[expected].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("news-only features require unique parseable dates")
    for column in NEWS_ONLY_SIGNAL_IDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"news-only feature is not numeric: {column}")
    return frame.sort_values("date").reset_index(drop=True)


def build_news_only_annual_features(cfg, source: Optional[Path] = None) -> pd.DataFrame:
    """Aggregate monthly news-only counts into panel-year feature columns."""
    frame = load_news_only_features(cfg, source)
    frame["year"] = frame["date"].dt.year.astype(int)
    annual = frame.groupby("year", as_index=False)[list(NEWS_ONLY_SIGNAL_IDS)].sum()
    annual = annual.rename(columns={name: NEWS_ONLY_PREFIX + name
                                    for name in NEWS_ONLY_SIGNAL_IDS})
    return annual


def materialize_news_only_features(cfg, source: Optional[Path] = None) -> tuple[Path, dict]:
    """Copy a validated SLM output into the architecture's versioned artifact tree."""
    path = _source_path(cfg, source) if source is not None else Path(cfg.repo_root / cfg.news_only_features)
    frame = load_news_only_features(cfg, path)
    out = cfg.artifact_path("news/news_only_catalog_features.csv.gz")
    io.write_csv(frame, out)
    metadata = {
        "artifact": str(out),
        "source_file": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "signal_count": len(NEWS_ONLY_SIGNAL_IDS),
        "signal_ids": list(NEWS_ONLY_SIGNAL_IDS),
        "feature_boundary": "dated news-only SLM output; no pharmacy target columns",
        "source_urls": {
            "news": "https://www.gdeltproject.org/",
            "slm": "https://huggingface.co/google/flan-t5-small",
            "validation": "https://www.cdc.gov/flu/weekly/",
        },
    }
    io.write_metadata(cfg.artifact_path("news/news_only_catalog_features.json"), **metadata)
    return out, metadata
