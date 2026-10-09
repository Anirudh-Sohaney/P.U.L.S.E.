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

from .signal_store import connect, initialize, news_version_hash


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FDA_DRUGS_RSS_URL = "https://www.fda.gov/AboutFDA/ContactFDA/StayInformed/RSSFeeds/Drugs/rss.xml"
FDA_MEDWATCH_RSS_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml"
FDA_RECALLS_RSS_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml"
FDA_PRESS_RSS_URL = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
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
FDA_PRESS_EVENT_TERMS = ("approv", "recall", "shortage", "safety", "warning",
                         "treatment", "research", "drug", "medicine", "medication")
NEWS_DEFAULT_WINDOW = timedelta(days=3)
NEWS_CATCHUP_LIMIT = timedelta(days=90)
GDELT_QUERY_WINDOW = timedelta(days=3)
GDELT_MAX_RECORDS = 250
GDELT_MAX_REQUESTS = 64
GDELT_REQUEST_INTERVAL_SECONDS = 10


def _fda_rss_root(url: str, *, timeout: int) -> ElementTree.Element:
    """Retry transient FDA feed transport failures without retrying rate limits."""
    request = Request(url, headers={"User-Agent": "PULSE-public-signal-monitor/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:  # fixed FDA HTTPS URL
                return ElementTree.fromstring(response.read(2 * 1024 * 1024))
        except HTTPError as exc:
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, OSError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("FDA RSS retry loop ended without a response")


def _evidence_row(row: dict, source_name: str, fetched_at: str) -> tuple:
    """Retain each distinct source-visible article version and first capture time."""
    timestamp_kind = "gdelt_first_seen" if source_name == "gdelt_recent_news" else "rss_pub_date"
    fields = {"url_hash": row["url_hash"], "source_name": source_name,
              "title": row["title"], "url": row["url"], "source": row["source"],
              "source_timestamp": row["published_at"], "timestamp_kind": timestamp_kind,
              "relevance_score": row["relevance_score"],
              "matched_terms": row["matched_terms"],
              "summary_text": str(row.get("summary_text") or "")[:4000]}
    version_hash = news_version_hash(fields)
    return (version_hash, fields["url_hash"], source_name, fields["title"],
            fields["url"], fields["source"], fields["source_timestamp"],
            timestamp_kind, fetched_at, fields["relevance_score"],
            fields["matched_terms"], fields["summary_text"])


def _refresh_error(exc: Exception, failed_at: datetime) -> str:
    """Keep a server-provided 429 retry deadline with the failed source run."""
    message = str(exc)[:400]
    if not isinstance(exc, HTTPError) or exc.code != 429 or exc.headers is None:
        return message[:500]
    retry_after = exc.headers.get("Retry-After")
    if not retry_after:
        return message[:500]
    retry_at = None
    try:
        retry_at = failed_at + timedelta(seconds=max(0, int(retry_after)))
    except (ValueError, OverflowError):
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            retry_at = retry_at.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            pass
    if retry_at is None:
        return message[:500]
    marker = f" [retry_after_until={retry_at.isoformat()}]"
    return f"{message[:500 - len(marker)]}{marker}"


def _gdelt_articles(start: datetime, end: datetime, *, timeout: int) -> list[dict]:
    params = urlencode({"query": NEWS_QUERY, "mode": "artlist", "format": "json",
                        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                        "maxrecords": GDELT_MAX_RECORDS, "sort": "datedesc"})
    request = Request(f"{GDELT_DOC_URL}?{params}", headers={
        "User-Agent": "PULSE-public-signal-monitor/1.0", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:  # fixed HTTPS URL
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, OSError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise ValueError("GDELT response has no article list")
    return articles


def fetch_recent_news(*, timeout: int = 25, cutoff: datetime | None = None) -> list[dict]:
    now = datetime.now(timezone.utc)
    start = cutoff or now - NEWS_DEFAULT_WINDOW
    if start.tzinfo is None:
        raise ValueError("GDELT cutoff must be timezone-aware")
    start = start.astimezone(timezone.utc)
    if start > now or now - start > NEWS_CATCHUP_LIMIT:
        raise ValueError("GDELT catch-up exceeds the supported 90-day window")
    seen_articles: dict[str, dict] = {}
    requests = 0

    def collect(window_start: datetime, window_end: datetime) -> None:
        nonlocal requests
        requests += 1
        if requests > GDELT_MAX_REQUESTS:
            raise RuntimeError("GDELT catch-up exceeded its 64-request safety limit")
        if requests > 1:
            # GDELT has no published per-client quota. Its operator notes that
            # small QPS changes can trigger 429s, so keep catch-up deliberately
            # slow; FDA feeds remain independent fallbacks during this wait.
            time.sleep(GDELT_REQUEST_INTERVAL_SECONDS)
        articles = _gdelt_articles(window_start, window_end, timeout=timeout)
        if len(articles) >= GDELT_MAX_RECORDS:
            if window_end - window_start <= timedelta(minutes=15):
                raise RuntimeError("GDELT article window still reaches its result cap at 15 minutes")
            middle = window_start + (window_end - window_start) / 2
            collect(window_start, middle + timedelta(seconds=1))
            collect(middle - timedelta(seconds=1), window_end)
            return
        for item in articles:
            article = _parse_gdelt_article(item, start, now)
            if article:
                seen_articles[article["url_hash"]] = article

    window_start = start
    while window_start < now:
        window_end = min(window_start + GDELT_QUERY_WINDOW, now)
        collect(window_start, window_end)
        if window_end == now:
            break
        window_start = window_end - timedelta(seconds=1)
    return list(seen_articles.values())


def _parse_gdelt_article(item: dict, cutoff: datetime, now: datetime) -> dict | None:
    title = str(item.get("title") or "").strip()
    url = str(item.get("url") or "").strip()
    parsed_url = urlparse(url)
    if not title or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return None
    seen_raw = str(item.get("seendate") or "")
    try:
        seen = datetime.strptime(seen_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if seen < cutoff or seen > now + timedelta(minutes=5):
        return None
    matches = [term for term in RELEVANT_TERMS if term in title.lower()]
    if not matches:
        return None
    return {"url_hash": hashlib.sha256(url.encode()).hexdigest(),
            "title": title[:500], "url": url,
            "source": str(item.get("domain") or parsed_url.netloc)[:160],
            "published_at": seen.isoformat(), "relevance_score": len(matches),
            "matched_terms": ",".join(matches),
            "summary_text": str(item.get("snippet") or "")[:4000]}


def _effective_cutoff(cutoff: datetime | None) -> datetime:
    if cutoff is None:
        return datetime.now(timezone.utc) - NEWS_DEFAULT_WINDOW
    if cutoff.tzinfo is None:
        raise ValueError("News cutoff must be timezone-aware")
    return cutoff.astimezone(timezone.utc)


def fetch_fda_drug_updates(*, timeout: int = 25,
                           cutoff: datetime | None = None) -> list[dict]:
    """Verified FDA RSS fallback when a general news source is unavailable."""
    root = _fda_rss_root(FDA_DRUGS_RSS_URL, timeout=timeout)
    cutoff = _effective_cutoff(cutoff)
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
                       "matched_terms": ",".join(terms or events[:1]),
                       "summary_text": (item.findtext("description") or "").strip()[:4000]})
    return output


def fetch_fda_medwatch_updates(*, timeout: int = 25,
                               cutoff: datetime | None = None) -> list[dict]:
    """Drug-related FDA MedWatch safety alerts, excluding device-only notices."""
    root = _fda_rss_root(FDA_MEDWATCH_RSS_URL, timeout=timeout)
    cutoff = _effective_cutoff(cutoff)
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
                       "matched_terms": ",".join(matches),
                       "summary_text": (item.findtext("description") or "").strip()[:4000]})
    return output


def fetch_fda_drug_recalls(*, timeout: int = 25,
                           cutoff: datetime | None = None) -> list[dict]:
    """Keep drug-specific announcements from FDA's broader recalls feed."""
    root = _fda_rss_root(FDA_RECALLS_RSS_URL, timeout=timeout)
    cutoff = _effective_cutoff(cutoff)
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
                       "matched_terms": ",".join(matches),
                       "summary_text": description[:4000]})
    return output


def fetch_fda_drug_press(*, timeout: int = 25,
                         cutoff: datetime | None = None) -> list[dict]:
    """Drug-related FDA press announcements; these are context, not demand data."""
    root = _fda_rss_root(FDA_PRESS_RSS_URL, timeout=timeout)
    cutoff = _effective_cutoff(cutoff)
    output = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        url = (item.findtext("link") or "").strip().replace(
            "http://www.fda.gov/", "https://www.fda.gov/", 1)
        lowered_title = title.lower()
        matches = [term for term in DRUG_RECALL_TERMS
                   if re.search(rf"\b{re.escape(term)}\b", f"{title} {description}".lower())]
        if (not title or not matches or not any(term in lowered_title for term in FDA_PRESS_EVENT_TERMS)
                or not url.startswith("https://www.fda.gov/news-events/press-announcements/")):
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
                       "title": title[:500], "url": url, "source": "FDA Press RSS",
                       "published_at": published.isoformat(),
                       "relevance_score": len(matches),
                       "matched_terms": ",".join(matches),
                       "summary_text": description[:4000]})
    return output


def refresh_news(sources: tuple[str, ...] | None = None) -> dict:
    initialize()
    results = []
    fetchers = {"gdelt_recent_news": fetch_recent_news,
                "fda_drugs_rss": fetch_fda_drug_updates,
                "fda_medwatch_rss": fetch_fda_medwatch_updates,
                "fda_recalls_rss": fetch_fda_drug_recalls,
                "fda_press_rss": fetch_fda_drug_press}
    selected = sources or tuple(fetchers)
    if any(name not in fetchers for name in selected):
        raise ValueError("Unknown public news source")
    for source_name in selected:
        fetcher = fetchers[source_name]
        now = datetime.now(timezone.utc)
        started = now.isoformat()
        with connect() as db:
            last_success = db.execute("""SELECT MAX(finished_at) FROM refresh_runs
                WHERE source_name=? AND status='success'""", (source_name,)).fetchone()[0]
            run_id = db.execute("""INSERT INTO refresh_runs(source_name, started_at, status)
                VALUES (?, ?, 'running')""", (source_name, started)).lastrowid
        try:
            cutoff = now - NEWS_DEFAULT_WINDOW
            if last_success:
                cutoff = min(cutoff, datetime.fromisoformat(last_success).astimezone(
                    timezone.utc) - timedelta(hours=1))
            articles = fetcher(cutoff=cutoff)
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
                before_evidence = db.total_changes
                db.executemany("""INSERT OR IGNORE INTO news_article_versions
                    (version_hash, url_hash, source_name, title, url, source,
                     source_timestamp, timestamp_kind, first_observed_at,
                     relevance_score, matched_terms, summary_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [_evidence_row(row, source_name, fetched) for row in articles])
                evidence_added = db.total_changes - before_evidence
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
                            "fetched": len(articles), "rows_written": written,
                            "evidence_added": evidence_added})
        except Exception as exc:
            failed_at = datetime.now(timezone.utc)
            error = _refresh_error(exc, failed_at)
            with connect() as db:
                db.execute("""UPDATE refresh_runs SET finished_at=?, status='failed', error=?
                    WHERE id=?""", (failed_at.isoformat(), error, run_id))
            results.append({"source": source_name, "status": "failed", "error": error})
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


def refresh_fda_press_news() -> dict:
    return refresh_news(("fda_press_rss",))
