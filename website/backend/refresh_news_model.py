"""Run the locally saved 20-signal article model as a quarantined shadow job.

Inputs are captured source article titles and summaries for the most recently
closed UTC month. If that month has no captured text, the job can produce a
current-month-to-date estimate targeted at month end. The model was trained on
only ten matched months and production feeds do not supply the same full-text
corpus, so writes remain explicitly unusable until a promotion evaluation
passes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

from .news_text_model import load_model, predict
from .signal_store import connect, initialize


SOURCE_NAME = "article_text_20_signal_shadow_v1"
SOURCE_KIND = "article_text_ridge_shadow_v1"
# The inference may use FDA feeds, GDELT, or both. A single source URL would
# misattribute the result; the actual input sources are recorded in quality.
MODEL_SOURCE_URL = ""
MISATTRIBUTED_SOURCE_URL = "https://www.gdeltproject.org/"


def _previous_month(now: datetime) -> tuple[str, str, str]:
    first_this_month = now.date().replace(day=1)
    last_previous_month = first_this_month.fromordinal(first_this_month.toordinal() - 1)
    first_previous_month = last_previous_month.replace(day=1)
    return (first_previous_month.isoformat(), first_this_month.isoformat(),
            last_previous_month.isoformat())


def _next_month_start(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def _select_articles(db, start: str, end: str, observed_by: str) -> list[dict]:
    return [dict(row) for row in db.execute("""WITH source_latest AS (
            SELECT version_hash, url_hash, title, source, source_timestamp,
                first_observed_at, summary_text,
                ROW_NUMBER() OVER (PARTITION BY source_name, url_hash
                    ORDER BY first_observed_at DESC, version_hash DESC) AS source_rank
            FROM news_article_versions
            WHERE source_timestamp >= ? AND source_timestamp < ?
              AND first_observed_at <= ?
        ) SELECT version_hash, url_hash, title, source, source_timestamp,
            first_observed_at, summary_text FROM source_latest
          WHERE source_rank=1
          ORDER BY url_hash, LENGTH(summary_text) DESC, first_observed_at DESC,
                   version_hash""", (start, end, observed_by))]


def refresh_news_model() -> dict:
    """Infer 20 estimates from actual captured news; never promote these values."""
    initialize()
    with connect() as db:
        db.execute("""UPDATE signal_records SET source_url=?
            WHERE source_kind=? AND source_url=?""",
                   (MODEL_SOURCE_URL, SOURCE_KIND, MISATTRIBUTED_SOURCE_URL))
    started = datetime.now(timezone.utc)
    started_text = started.isoformat()
    with connect() as db:
        run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
            VALUES (?, ?, 'running')""", (SOURCE_NAME, started_text)).lastrowid
    try:
        model = load_model()
        month_start, next_month_start, month_end = _previous_month(started)
        with connect() as db:
            articles = _select_articles(db, month_start, next_month_start, started_text)
            definitions = {row["signal_id"]: row["id"] for row in db.execute("""
                SELECT id, signal_id FROM signal_definitions
                WHERE signal_origin='model_news_output'""")}
            estimate_kind = "closed_month_article_text_estimate"
            if not articles:
                current_start = started.date().replace(day=1)
                next_start = _next_month_start(started.date())
                month_start = current_start.isoformat()
                month_end = (next_start - timedelta(days=1)).isoformat()
                articles = _select_articles(db, month_start, started_text, started_text)
                estimate_kind = "month_to_date_month_end_nowcast"
        expected_ids = model["signal_ids"]
        if set(definitions) != set(expected_ids) or len(definitions) != 20:
            raise ValueError("Model output IDs do not match the 20 catalog definitions")
        unique_articles = []
        seen_urls = set()
        for article in articles:
            if article["url_hash"] in seen_urls:
                continue
            seen_urls.add(article["url_hash"])
            text = " ".join((article["title"], article["summary_text"] or "")).strip()
            if text:
                unique_articles.append((article, text))
        if not unique_articles:
            raise ValueError(
                "No captured article text for the previous closed or current UTC month")

        text = "\n".join(article_text for _, article_text in unique_articles)
        output = predict(text, model)
        if set(output) != set(expected_ids) or len(output) != 20:
            raise ValueError("20-signal inference returned an incomplete output set")
        finished = datetime.now(timezone.utc)
        finished_text = finished.isoformat()
        source_names = sorted({row["source"] for row, _ in unique_articles})
        quality = ("model_estimate;shadow_only;not_observed;not_validated_for_publication;"
                   f"input_summary_text;estimate_kind={estimate_kind};"
                   f"article_count={len(unique_articles)};"
                   f"sources={','.join(source_names)};month={month_start[:7]}")
        rows = []
        provenance_rows = []
        input_rows = []
        article_versions = sorted(article["version_hash"] for article, _ in unique_articles)
        for signal_id in expected_ids:
            uid = definitions[signal_id]
            value = output[signal_id]
            row_hash = hashlib.sha256(json.dumps(
                [SOURCE_KIND, model["source_sha256"], uid, month_end,
                 article_versions, value], separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest()
            rows.append((row_hash, uid, finished.date().isoformat(), month_end, value,
                         finished_text, finished_text, quality,
                         estimate_kind, SOURCE_KIND,
                         MODEL_SOURCE_URL))
            provenance_rows.append((row_hash, model["source_sha256"],
                                    len(unique_articles), estimate_kind))
            input_rows.extend((row_hash, position, article["version_hash"])
                              for position, (article, _) in enumerate(unique_articles))
        with connect() as db:
            before = db.total_changes
            db.executemany("""INSERT OR IGNORE INTO signal_records
                (row_hash, signal_uid, signal_date, observation_date, value,
                 source_timestamp, ingested_at, data_quality, forecast_horizon,
                 source_kind, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
            written = db.total_changes - before
            db.executemany("""INSERT OR IGNORE INTO news_model_record_provenance
                (row_hash, model_sha256, input_count, estimate_kind)
                VALUES (?, ?, ?, ?)""", provenance_rows)
            db.executemany("""INSERT OR IGNORE INTO news_model_record_inputs
                (row_hash, input_position, version_hash) VALUES (?, ?, ?)""", input_rows)
            for row_hash, model_sha256, input_count, kind in provenance_rows:
                evidence = db.execute("""SELECT model_sha256, input_count, estimate_kind
                    FROM news_model_record_provenance WHERE row_hash=?""",
                    (row_hash,)).fetchone()
                inputs = db.execute("""SELECT version_hash FROM news_model_record_inputs
                    WHERE row_hash=? ORDER BY input_position""", (row_hash,)).fetchall()
                if (evidence is None or tuple(evidence) != (model_sha256, input_count, kind)
                        or [row[0] for row in inputs] !=
                        [article["version_hash"] for article, _ in unique_articles]):
                    raise ValueError("Stored news-model evidence does not match inference inputs")
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                rows_written=? WHERE id=?""", (finished_text, written, run_id))
        return {"source": SOURCE_NAME, "status": "success", "signals": len(rows),
                "articles": len(unique_articles), "input_sources": source_names,
                "observation_month": month_start[:7], "observation_date": month_end,
                "estimate_kind": estimate_kind, "rows_written": written, "publishable": False,
                "reason": "shadow_model_requires_full_text_and_chronological_validation"}
    except Exception as exc:
        finished_text = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                WHERE id=?""", (finished_text, str(exc)[:500], run_id))
        raise
