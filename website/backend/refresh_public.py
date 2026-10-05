"""Daily ingestion of verifiable recent public news, separate from model signals."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

from .signal_store import connect, initialize


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FDA_DRUGS_RSS_URL = "https://www.fda.gov/AboutFDA/ContactFDA/StayInformed/RSSFeeds/Drugs/rss.xml"
FDA_MEDWATCH_RSS_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml"
FDA_RECALLS_RSS_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml"
NEWS_QUERY = '("drug shortage" OR "medicine shortage" OR "pharmacy demand" OR "medication supply")'
RELEVANT_TERMS = ("drug", "medicine", "medication", "pharmacy", "shortage", "supply", "recall")
FDA_EVENT_TERMS = ("shortage", "recall", "approves", "warning", "safety alert",
                   "concerns", "discontinuation", "supply", "manufacturing")
MEDWATCH_DRUG_TERMS = ("drug", "medicine", "medication", "pharma", "compounded",
                       "tablet", "capsule", "injection", "injectable", "infusion",
                   "vial", "ophthalmic", "insulin", "antibiotic", "prescription")
DRUG_RECALL_TERMS = ("drug", "medicine", "medication", "pharmacy", "pharmaceutical",
                     "compounded", "prescription", "injectable", "injection", "tablet",
                     "tablets", "vial", "vials", "insulin", "antibiotic")


def fetch_recent_news(*, timeout: int = 25) -> list[dict]:
    params = urlencode({"query": NEWS_QUERY, "mode": "artlist", "format": "json",
                        "timespan": "3d", "maxrecords": 100, "sort": "datedesc"})
    request = Request(f"{GDELT_DOC_URL}?{params}", headers={
        "User-Agent": "PULSE-public-signal-monitor/1.0",
        "Accept": "application/json",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:  # fixed HTTPS URL
                payload = json.load(response)
            break
        except HTTPError as exc:
            # A rate limit is a source policy, not a transient connection fault.
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, OSError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("GDELT response has no article list")
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    output: list[dict] = []
    for item in articles:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        parsed_url = urlparse(url)
        if not title or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        seen_raw = str(item.get("seendate") or "")
        try:
            seen = datetime.strptime(seen_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if seen < cutoff or seen > datetime.now(timezone.utc) + timedelta(minutes=5):
            continue
        matches = [term for term in RELEVANT_TERMS if term in title.lower()]
        if not matches:
            continue
        output.append({"url_hash": hashlib.sha256(url.encode()).hexdigest(),
                       "title": title[:500], "url": url,
                       "source": str(item.get("domain") or parsed_url.netloc)[:160],
                       "published_at": seen.isoformat(), "relevance_score": len(matches),
                       "matched_terms": ",".join(matches)})
    return output


def fetch_fda_drug_updates(*, timeout: int = 25) -> list[dict]:
    """Verified FDA RSS fallback when a general news source is unavailable."""
    request = Request(FDA_DRUGS_RSS_URL, headers={"User-Agent": "PULSE-public-signal-monitor/1.0"})
    with urlopen(request, timeout=timeout) as response:  # fixed HTTPS URL
        root = ElementTree.fromstring(response.read(2 * 1024 * 1024))
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    output = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip().replace("http://www.fda.gov/", "https://www.fda.gov/", 1)
        title_lower = title.lower()
        events = [term for term in FDA_EVENT_TERMS if term in title_lower]
        terms = [term for term in RELEVANT_TERMS if term in title_lower]
        if (not title or not events or "approval notifications" in title_lower
                or not url.startswith("https://www.fda.gov/")):
            continue
        try:
            published = parsedate_to_datetime((item.findtext("pubDate") or "").strip())
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=ZoneInfo("America/New_York"))
        published = published.astimezone(timezone.utc)
        if published < cutoff or published > datetime.now(timezone.utc) + timedelta(minutes=5):
            continue
        output.append({"url_hash": hashlib.sha256(url.encode()).hexdigest(),
                       "title": title[:500], "url": url, "source": "FDA Drugs RSS",
                       "published_at": published.isoformat(),
                       "relevance_score": max(1, len(terms)),
                       "matched_terms": ",".join(terms or events[:1])})
    return output


def fetch_fda_medwatch_updates(*, timeout: int = 25) -> list[dict]:
    """Drug-related FDA MedWatch safety alerts, excluding device-only notices."""
    request = Request(FDA_MEDWATCH_RSS_URL, headers={
        "User-Agent": "PULSE-public-signal-monitor/1.0"})
    with urlopen(request, timeout=timeout) as response:
        root = ElementTree.fromstring(response.read(2 * 1024 * 1024))
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    output = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip().replace(
            "http://www.fda.gov/", "https://www.fda.gov/", 1)
        lowered = title.lower()
        path = urlparse(url).path.lower()
        matches = [term for term in MEDWATCH_DRUG_TERMS if term in lowered]
        if (not title or not matches or not url.startswith("https://www.fda.gov/")
                or path.startswith("/medical-devices/")
                or not (path.startswith("/safety/") or path.startswith("/drugs/"))):
            continue
        try:
            published = parsedate_to_datetime((item.findtext("pubDate") or "").strip())
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=ZoneInfo("America/New_York"))
        published = published.astimezone(timezone.utc)
        if published < cutoff or published > datetime.now(timezone.utc) + timedelta(minutes=5):
            continue
        output.append({"url_hash": hashlib.sha256(url.encode()).hexdigest(),
                       "title": title[:500], "url": url, "source": "FDA MedWatch RSS",
                       "published_at": published.isoformat(),
                       "relevance_score": len(matches),
                       "matched_terms": ",".join(matches)})
    return output


def fetch_fda_drug_recalls(*, timeout: int = 25) -> list[dict]:
    """Keep drug-specific announcements from FDA's broader recalls feed."""
    request = Request(FDA_RECALLS_RSS_URL, headers={
        "User-Agent": "PULSE-public-signal-monitor/1.0"})
    with urlopen(request, timeout=timeout) as response:
        root = ElementTree.fromstring(response.read(2 * 1024 * 1024))
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    output = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        url = (item.findtext("link") or "").strip().replace(
            "http://www.fda.gov/", "https://www.fda.gov/", 1)
        if not title or not url.startswith("https://www.fda.gov/safety/"):
            continue
        lowered = f"{title} {description}".lower()
        matches = [term for term in DRUG_RECALL_TERMS
                   if re.search(rf"\b{re.escape(term)}\b", lowered)]
        if not matches:
            continue
        try:
            published = parsedate_to_datetime((item.findtext("pubDate") or "").strip())
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=ZoneInfo("America/New_York"))
        published = published.astimezone(timezone.utc)
        if published < cutoff or published > datetime.now(timezone.utc) + timedelta(minutes=5):
            continue
        output.append({"url_hash": hashlib.sha256(url.encode()).hexdigest(),
                       "title": title[:500], "url": url, "source": "FDA Recalls RSS",
                       "published_at": published.isoformat(),
                       "relevance_score": len(matches),
                       "matched_terms": ",".join(matches)})
    return output


def refresh_news(sources: tuple[str, ...] | None = None) -> dict:
    initialize()
    results = []
    fetchers = {"gdelt_recent_news": fetch_recent_news,
                "fda_drugs_rss": fetch_fda_drug_updates,
                "fda_medwatch_rss": fetch_fda_medwatch_updates,
                "fda_recalls_rss": fetch_fda_drug_recalls}
    selected = sources or tuple(fetchers)
    if any(name not in fetchers for name in selected):
        raise ValueError("Unknown public news source")
    for source_name in selected:
        fetcher = fetchers[source_name]
        started = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
                VALUES (?, ?, 'running')""", (source_name, started)).lastrowid
        try:
            articles = fetcher()
            fetched = datetime.now(timezone.utc).isoformat()
            with connect() as db:
                if source_name == "fda_drugs_rss":
                    db.execute("""DELETE FROM news_articles WHERE source='FDA Drugs RSS'
                        AND (instr(lower(title), 'approval notifications') > 0
                        OR NOT (lower(title) LIKE '%shortage%' OR lower(title) LIKE '%recall%'
                          OR lower(title) LIKE '%approves%'
                          OR lower(title) LIKE '%warning%' OR lower(title) LIKE '%safety alert%'
                          OR lower(title) LIKE '%concerns%' OR lower(title) LIKE '%discontinuation%'
                          OR lower(title) LIKE '%supply%' OR lower(title) LIKE '%manufacturing%'))""")
                before = db.total_changes
                db.executemany("""INSERT INTO news_articles
                    (url_hash, title, url, source, published_at, timestamp_kind,
                     fetched_at, relevance_score, matched_terms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url_hash) DO UPDATE SET title=excluded.title,
                      source=excluded.source, published_at=excluded.published_at,
                      timestamp_kind=excluded.timestamp_kind,
                      fetched_at=excluded.fetched_at,
                      relevance_score=excluded.relevance_score,
                      matched_terms=excluded.matched_terms""",
                    [(row["url_hash"], row["title"], row["url"], row["source"],
                      row["published_at"],
                      "rss_pub_date" if source_name != "gdelt_recent_news" else "gdelt_first_seen",
                      fetched, row["relevance_score"], row["matched_terms"])
                     for row in articles])
                written = db.total_changes - before
                db.execute("""UPDATE refresh_runs SET finished_at=?, status='success',
                    rows_written=? WHERE id=?""", (fetched, written, run_id))
            results.append({"source": source_name, "status": "success",
                            "fetched": len(articles), "rows_written": written})
        except Exception as exc:
            with connect() as db:
                db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                    WHERE id=?""", (datetime.now(timezone.utc).isoformat(), str(exc)[:500], run_id))
            results.append({"source": source_name, "status": "failed", "error": str(exc)[:500]})
    if all(result["status"] == "failed" for result in results):
        failures = "; ".join(f"{result['source']}: {result['error']}" for result in results)
        raise RuntimeError(f"All public news sources failed: {failures}")
    return {"sources": results, "rows_written": sum(result.get("rows_written", 0)
                                                     for result in results)}


def refresh_gdelt_news() -> dict:
    return refresh_news(("gdelt_recent_news",))


def refresh_fda_drugs_news() -> dict:
    return refresh_news(("fda_drugs_rss",))


def refresh_fda_medwatch_news() -> dict:
    return refresh_news(("fda_medwatch_rss",))


def refresh_fda_recalls_news() -> dict:
    return refresh_news(("fda_recalls_rss",))
