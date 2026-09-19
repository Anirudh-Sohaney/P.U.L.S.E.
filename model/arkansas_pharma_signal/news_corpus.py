"""Historical article corpus loader with article-time availability controls."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import io

CORPUS_COLUMNS = ["article_id", "title", "body", "published_at", "source",
                  "url", "source_hash", "author", "city", "county_fips",
                  "language", "is_full_text", "duplicate_id", "source_reliability"]


def _text(v) -> str:
    return "" if v is None else str(v).strip()


def _body(record: dict) -> str:
    for key in ("body", "text", "article_text", "content", "description"):
        if _text(record.get(key)):
            return _text(record[key])
    return ""


def load_historical_corpus(cfg, city_to_county: dict[str, str] | None = None) -> pd.DataFrame:
    """Load raw JSONL full text and keep metadata-only rows separately marked.

    No row is invented and no publication after an ``as_of`` cutoff is used by
    :func:`available_at`.  The source hash makes syndicated/duplicate handling
    deterministic even when URLs differ.
    """
    out = []
    seen_ids = set()
    raw_dir = cfg.data_path(cfg.news_dir).parent / "raw"
    for path in sorted(raw_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                body = _body(r)
                title = _text(r.get("title") or r.get("headline"))
                url = _text(r.get("url") or r.get("link"))
                source = _text(r.get("publication") or r.get("media_name") or r.get("source"))
                published = r.get("publication_date") or r.get("date")
                city = _text((r.get("location") or {}).get("city") if isinstance(r.get("location"), dict) else r.get("city"))
                aid = _text(r.get("id")) or hashlib.sha256((url + title).encode()).hexdigest()[:20]
                sh = hashlib.sha256((title + "\n" + body).encode("utf-8")).hexdigest()
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)
                out.append({"article_id": aid, "title": title, "body": body,
                            "published_at": published, "source": source, "url": url,
                            "source_hash": sh, "author": _text(r.get("author")),
                            "city": city, "county_fips": (city_to_county or {}).get(city.lower(), ""),
                            "language": _text(r.get("language")) or "en",
                            "is_full_text": bool(body), "duplicate_id": sh,
                            "source_reliability": 0.5})

    # Metadata-only files remain part of the corpus, but cannot train language tasks.
    meta_path = cfg.data_path(cfg.news_dir) / "arkansas_3dlnews_targeted_article_metadata.csv.gz"
    if meta_path.exists():
        m = io.load_csv(meta_path)
        for _, r in m.iterrows():
            aid = _text(r.get("id"))
            if not aid:
                continue
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            title = _text(r.get("title")); url = _text(r.get("expanded_url") or r.get("url"))
            sh = hashlib.sha256(title.encode()).hexdigest()
            out.append({"article_id": aid, "title": title, "body": "",
                        "published_at": r.get("publication_date"), "source": _text(r.get("publication")),
                        "url": url, "source_hash": sh, "author": "",
                        "city": _text(r.get("city")), "county_fips": (city_to_county or {}).get(_text(r.get("city")).lower(), ""),
                        "language": "en", "is_full_text": False, "duplicate_id": sh,
                        "source_reliability": 0.35})
    frame = pd.DataFrame(out, columns=CORPUS_COLUMNS)
    if frame.empty:
        return frame
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    return frame.drop_duplicates(["article_id"]).reset_index(drop=True)


def available_at(corpus: pd.DataFrame, as_of: object) -> pd.DataFrame:
    """Apply the temporal information barrier for a forecast timestamp."""
    t = pd.Timestamp(as_of)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    published = pd.to_datetime(corpus["published_at"], errors="coerce", utc=True)
    # An unknown publication time cannot be proven available at the cutoff.
    # Keep it in the stored corpus, but exclude it from time-indexed features.
    return corpus[published.notna() & (published <= t)].copy()
