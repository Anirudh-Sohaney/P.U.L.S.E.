"""News sources keep independent status and do not hide a failed feed."""

import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
from fastapi.testclient import TestClient

from backend import refresh_public, signal_store
from backend.app import create_app
from backend.config import settings


def test_gdelt_retries_connection_reset_but_not_rate_limit(monkeypatch):
    attempts = []
    pauses = []
    monkeypatch.setattr(refresh_public.time, "sleep", pauses.append)

    def reset_then_succeed(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionResetError("reset by peer")
        return io.BytesIO(json.dumps({"articles": []}).encode())

    monkeypatch.setattr(refresh_public, "urlopen", reset_then_succeed)
    assert refresh_public.fetch_recent_news() == []
    assert len(attempts) == 3
    assert pauses == [1, 2]

    def rate_limited(request, timeout):
        attempts.append(1)
        raise refresh_public.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    attempts.clear()
    pauses.clear()
    monkeypatch.setattr(refresh_public, "urlopen", rate_limited)
    with pytest.raises(refresh_public.HTTPError) as error:
        refresh_public.fetch_recent_news()
    assert error.value.code == 429
    assert len(attempts) == 1
    assert pauses == []


def _article(source: str, url: str) -> dict:
    return {"url_hash": hashlib.sha256(url.encode()).hexdigest(),
            "title": "Drug shortage update", "url": url, "source": source,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "relevance_score": 2, "matched_terms": "drug,shortage"}


def test_fda_drug_feed_excludes_evergreen_approval_indexes(monkeypatch):
    published = format_datetime(datetime.now(timezone.utc))
    feed = f"""<rss><channel>
      <item><title>First Generic Drug Approvals</title>
        <link>https://www.fda.gov/drugs/first-generic-drug-approvals</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Ongoing | Cancer Accelerated Approvals</title>
        <link>https://www.fda.gov/drugs/ongoing-approvals</link>
        <pubDate>{published}</pubDate></item>
      <item><title>FDA Approves Example Drug for Treatment</title>
        <link>https://www.fda.gov/drugs/fda-approves-example-drug</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Example Drug Shortage Update</title>
        <link>https://www.fda.gov/drugs/example-shortage</link>
        <pubDate>{published}</pubDate></item>
    </channel></rss>""".encode()
    monkeypatch.setattr(refresh_public, "urlopen", lambda request, timeout: io.BytesIO(feed))
    titles = [row["title"] for row in refresh_public.fetch_fda_drug_updates()]
    assert titles == ["FDA Approves Example Drug for Treatment", "Example Drug Shortage Update"]


def test_fda_refresh_removes_stored_approval_index_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    signal_store.initialize()
    title = "First Generic Drug Approvals"
    with signal_store.connect() as db:
        db.execute("""INSERT INTO news_articles
            (url_hash, title, url, source, published_at, timestamp_kind,
             fetched_at, relevance_score, matched_terms)
            VALUES (?, ?, ?, 'FDA Drugs RSS', ?, 'rss_pub_date', ?, 1, 'drug')""",
            ("approval-index", title, "https://www.fda.gov/drugs/first-generic-drug-approvals",
             datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()))
    monkeypatch.setattr(refresh_public, "fetch_fda_drug_updates", lambda: [])
    refresh_public.refresh_news(("fda_drugs_rss",))
    assert signal_store.recent_news() == []


def test_fda_recalls_feed_keeps_drug_recall_and_excludes_food(monkeypatch):
    published = format_datetime(datetime.now(timezone.utc))
    feed = f"""<rss><channel>
      <item><title>Greenwich Rx Recalls Compounded Glutathione</title>
        <link>http://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/glutathione</link>
        <description>A compounding pharmacy is recalling the medication.</description>
        <pubDate>{published}</pubDate></item>
      <item><title>Bakery Recalls Bread</title>
        <link>https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/bread</link>
        <description>Undeclared milk in bread.</description>
        <pubDate>{published}</pubDate></item>
    </channel></rss>""".encode()
    monkeypatch.setattr(refresh_public, "urlopen", lambda request, timeout: io.BytesIO(feed))
    rows = refresh_public.fetch_fda_drug_recalls()
    assert len(rows) == 1
    assert rows[0]["title"] == "Greenwich Rx Recalls Compounded Glutathione"
    assert rows[0]["source"] == "FDA Recalls RSS"
    assert rows[0]["url"].startswith("https://www.fda.gov/")


def test_fda_success_keeps_gdelt_failure_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))

    def unavailable():
        raise RuntimeError("rate limited")

    monkeypatch.setattr(refresh_public, "fetch_recent_news", unavailable)
    monkeypatch.setattr(refresh_public, "fetch_fda_drug_updates", lambda: [
        _article("FDA Drugs RSS", "https://www.fda.gov/drug-shortage")])
    monkeypatch.setattr(refresh_public, "fetch_fda_medwatch_updates", lambda: [])
    monkeypatch.setattr(refresh_public, "fetch_fda_drug_recalls", lambda: [])
    result = refresh_public.refresh_news()
    assert [row["status"] for row in result["sources"]] == ["failed", "success", "success", "success"]
    with signal_store.connect() as db:
        runs = [tuple(row) for row in db.execute("""SELECT source_name, status, error
            FROM refresh_runs ORDER BY id""")]
        articles = db.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    assert runs == [("gdelt_recent_news", "failed", "rate limited"),
                    ("fda_drugs_rss", "success", None),
                    ("fda_medwatch_rss", "success", None),
                    ("fda_recalls_rss", "success", None)]
    assert articles == 1
    assert signal_store.recent_news()[0]["timestamp_kind"] == "rss_pub_date"
    checks = {row["source_name"]: row for row in signal_store.freshness()["source_checks"]}
    assert checks["gdelt_recent_news"]["last_status"] == "failed"
    assert checks["fda_drugs_rss"]["last_status"] == "success"
    assert checks["fda_medwatch_rss"]["last_status"] == "success"
    assert checks["fda_recalls_rss"]["last_status"] == "success"
    public = TestClient(create_app()).get("/api/v1/signals/news/recent").json()
    assert public["count"] == 1
    assert public["source_checks"]["gdelt_recent_news"]["last_status"] == "failed"
    assert public["source_checks"]["fda_drugs_rss"]["last_status"] == "success"
    assert public["source_checks"]["fda_medwatch_rss"]["last_status"] == "success"
    assert public["source_checks"]["fda_recalls_rss"]["last_status"] == "success"
    assert public["source_checks"]["gdelt_recent_news"]["last_success_at"] is None


def test_both_news_failures_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))

    def unavailable():
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(refresh_public, "fetch_recent_news", unavailable)
    monkeypatch.setattr(refresh_public, "fetch_fda_drug_updates", unavailable)
    monkeypatch.setattr(refresh_public, "fetch_fda_medwatch_updates", unavailable)
    monkeypatch.setattr(refresh_public, "fetch_fda_drug_recalls", unavailable)
    with pytest.raises(RuntimeError, match="All public news sources failed: gdelt_recent_news: source unavailable"):
        refresh_public.refresh_news()
    with signal_store.connect() as db:
        statuses = [row[0] for row in db.execute(
            "SELECT status FROM refresh_runs ORDER BY id")]
    assert statuses == ["failed", "failed", "failed", "failed"]


def test_existing_news_dates_gain_explicit_source_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path))
    with sqlite3.connect(tmp_path / "signals.sqlite3") as db:
        db.execute("""CREATE TABLE news_articles (
            url_hash TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
            source TEXT NOT NULL, published_at TEXT NOT NULL,
            fetched_at TEXT NOT NULL, relevance_score INTEGER NOT NULL,
            matched_terms TEXT NOT NULL)""")
        for source, url in (("FDA Drugs RSS", "https://www.fda.gov/example"),
                            ("example.org", "https://example.org/article")):
            article = _article(source, url)
            db.execute("""INSERT INTO news_articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                       (article["url_hash"], article["title"], article["url"], source,
                        article["published_at"], article["published_at"],
                        article["relevance_score"], article["matched_terms"]))
    signal_store.initialize()
    assert {row["source"]: row["timestamp_kind"] for row in signal_store.recent_news()} == {
        "FDA Drugs RSS": "rss_pub_date", "example.org": "gdelt_first_seen"}


def test_fda_drug_events_can_name_the_medicine_without_generic_keywords(monkeypatch):
    published = format_datetime(datetime.now(timezone.utc) - timedelta(days=1))
    feed = f"""<rss><channel>
      <item><title>FDA approves pirtobrutinib for untreated leukemia</title>
        <link>https://www.fda.gov/drugs/approval-pirtobrutinib</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Greenwich Rx issues recall of compounded glutathione</title>
        <link>https://www.fda.gov/safety/recall-glutathione</link>
        <pubDate>{published}</pubDate></item>
      <item><title>What's New Related to Drugs</title>
        <link>https://www.fda.gov/drugs/whats-new</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Oncology Approval Notifications</title>
        <link>https://www.fda.gov/drugs/approval-notifications</link>
        <pubDate>{published}</pubDate></item>
    </channel></rss>""".encode()
    monkeypatch.setattr(refresh_public, "urlopen", lambda request, timeout: io.BytesIO(feed))
    rows = refresh_public.fetch_fda_drug_updates()
    assert [row["title"] for row in rows] == [
        "FDA approves pirtobrutinib for untreated leukemia",
        "Greenwich Rx issues recall of compounded glutathione",
    ]
    assert all(row["relevance_score"] >= 1 and row["matched_terms"] for row in rows)


def test_medwatch_keeps_drug_recalls_without_device_alerts(monkeypatch):
    published = format_datetime(datetime.now(timezone.utc) - timedelta(days=1))
    old = format_datetime(datetime.now(timezone.utc) - timedelta(days=5))
    feed = f"""<rss><channel>
      <item><title>Recall of compounded glutathione injection</title>
        <link>http://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/glutathione</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Injection pump device recall</title>
        <link>https://www.fda.gov/medical-devices/medical-device-recalls-and-early-alerts/pump</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Food recall of tablets-shaped candy</title>
        <link>https://www.fda.gov/food/outbreaks-foodborne-illness/candy</link>
        <pubDate>{published}</pubDate></item>
      <item><title>Old recall of insulin vials</title>
        <link>https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts/insulin</link>
        <pubDate>{old}</pubDate></item>
    </channel></rss>""".encode()
    monkeypatch.setattr(refresh_public, "urlopen", lambda request, timeout: io.BytesIO(feed))
    rows = refresh_public.fetch_fda_medwatch_updates()
    assert [row["title"] for row in rows] == ["Recall of compounded glutathione injection"]
    assert rows[0]["url"].startswith("https://www.fda.gov/")
