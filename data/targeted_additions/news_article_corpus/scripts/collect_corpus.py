"""Collect and normalize a dated, mixed-topic GDELT article corpus.

Outputs model-compatible JSONL plus a provenance manifest. Article pages are
retrieved only as publicly accessible pages; failed/paywalled pages remain
metadata-only rather than being fabricated.
"""
from __future__ import annotations
import concurrent.futures
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import trafilatura

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "data/targeted_additions/news_article_corpus"
OUT = BASE / "data"
RAW = BASE / "raw"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
YEARS = range(2023, 2026)
UA = "PULSE-research-corpus/1.0 (public-news feature research)"

QUERIES = {
    "drug_shortage": '("drug shortage" OR "medicine shortage")',
    "pharmaceutical": "(pharmaceutical OR pharma)",
    "pharmacy": "(pharmacy OR pharmacist OR prescription)",
    "fda_medicine": '(FDA AND (drug OR medicine OR recall))',
    "healthcare": "(hospital OR healthcare OR Medicaid)",
    "opioid": "(opioid OR naloxone OR overdose)",
    "drug_prices": '("drug prices" OR "prescription costs")',
    "manufacturing_supply": '(drug AND (manufacturing OR supply-chain OR factory))',
    "sports_control": "(sports OR football OR basketball OR baseball)",
    "arts_entertainment": "(film OR music OR theater OR entertainment)",
    "technology": "(technology OR software OR smartphone)",
    "travel_food": "(travel OR restaurant OR cooking)",
}
RELATED = set(list(QUERIES)[:8])
TARGETS = {"related": 260, "unrelated": 140}


def gdelt(query, year):
    params = {"query": f"{query} sourcelang:english", "mode": "artlist",
              "format": "json", "maxrecords": "250",
              "startdatetime": f"{year}0101000000",
              "enddatetime": f"{year}1231235959"}
    for attempt in range(4):
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=20)
            if r.ok:
                return r.json().get("articles", [])
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return []


def clean_url(url):
    return str(url or "").strip()


def fetch_article(item):
    url = clean_url(item.get("url"))
    title = str(item.get("title") or "").strip()
    text = ""
    status = None
    if url:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=8,
                             allow_redirects=True)
            status = r.status_code
            if r.ok and "text/html" in r.headers.get("content-type", "").lower():
                text = trafilatura.extract(r.text, include_comments=False,
                                           include_tables=False, favor_precision=True) or ""
        except requests.RequestException:
            pass
    text = re.sub(r"\s+", " ", text).strip()
    article_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    duplicate_id = hashlib.sha256((title + "\n" + text).encode("utf-8")).hexdigest()
    return {
        "article_id": article_id, "title": title, "body": text,
        "published_at": item.get("seendate"),
        "source": item.get("domain") or item.get("sourcecountry") or "",
        "url": url, "source_hash": duplicate_id, "author": "", "city": "",
        "county_fips": "", "language": "en", "is_full_text": bool(text),
        "duplicate_id": duplicate_id, "source_reliability": 0.5 if text else 0.25,
        "topic_label": item["topic_label"], "query_key": item["query_key"],
        "http_status": status,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    candidates = {}
    jobs = [(key, query, year) for key, query in QUERIES.items() for year in YEARS]
    def run_job(job):
        key, query, year = job
        return key, year, gdelt(query, year)
    cached = [RAW / f"gdelt_{key}_{year}.json" for key, _, year in jobs]
    if all(path.exists() for path in cached):
        query_results = [(key, year, json.loads(path.read_text(encoding="utf-8")))
                         for (key, _, year), path in zip(jobs, cached)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            query_results = list(pool.map(run_job, jobs))
    for key, year, articles in query_results:
            path = RAW / f"gdelt_{key}_{year}.json"
            path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
            for item in articles:
                url = clean_url(item.get("url"))
                if not url: continue
                item = dict(item); item["query_key"] = key
                item["topic_label"] = "related" if key in RELATED else "unrelated"
                candidates.setdefault(url, item)
    by_label = {label: [x for x in candidates.values() if x["topic_label"] == label]
                for label in TARGETS}
    # Stratify selection by year and deterministically shuffle within each stratum.
    selected = []
    for label, target in TARGETS.items():
        group = by_label[label]
        for x in group:
            x["_year"] = str(x.get("seendate", ""))[:4]
            x["_sort"] = hashlib.sha256((x["url"] + label).encode()).hexdigest()
        group.sort(key=lambda x: (x["_year"], x["_sort"]))
        buckets = {str(y): [x for x in group if x["_year"] == str(y)] for y in YEARS}
        per_year = target // len(YEARS)
        chosen = sum((buckets[str(y)][:per_year] for y in YEARS), [])
        remainder = [x for x in group if x not in chosen]
        chosen += remainder[: target - len(chosen)]
        selected.extend(chosen[:target])
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        rows = list(pool.map(fetch_article, selected))
    frame = pd.DataFrame(rows)
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    frame = frame[frame["published_at"].notna()].copy()
    frame = frame.drop_duplicates("duplicate_id").sort_values(["published_at", "article_id"])
    for destination in (OUT / "articles.jsonl.gz", RAW / "articles.jsonl.gz"):
        frame.to_json(destination, orient="records", lines=True,
                      compression="gzip", force_ascii=False)
    counts = frame.groupby([frame.published_at.dt.year.rename("year"), "topic_label"]).size().unstack(fill_value=0)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT DOC 2.0 Article List API",
        "source_url": API,
        "coverage": {"start": "2023-01-01", "end": "2025-12-31"},
        "requested_targets": TARGETS,
        "rows": int(len(frame)),
        "full_text_rows": int(frame.is_full_text.sum()),
        "metadata_only_rows": int((~frame.is_full_text).sum()),
        "duplicate_rows_removed": int(len(selected) - len(frame)),
        "topic_counts": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in counts.iterrows()},
        "schema": list(frame.columns),
        "model_loader_path": "raw/articles.jsonl.gz",
        "retrieval_note": "GDELT metadata was queried by calendar year; article pages were fetched separately. Failed or non-HTML pages are retained as metadata-only.",
        "license_note": "GDELT metadata is retained with URLs and provenance. Article body text is locally cached for research extraction and should not be redistributed without checking each publisher's terms.",
    }
    (BASE / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__": main()
