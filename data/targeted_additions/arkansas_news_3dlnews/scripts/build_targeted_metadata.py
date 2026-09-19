#!/usr/bin/env python3
"""Normalize 3DLNews Arkansas newspaper metadata + keyword counts.

Reads raw/preprocessed_newspaper_articles_AR_{year}.jsonl.gz, matches article
content against keyword groups, and writes:
  data/arkansas_3dlnews_targeted_article_metadata.csv.gz
  data/arkansas_3dlnews_monthly_keyword_counts.csv.gz
Article full text is never stored; only metadata + keyword-hit summary.
"""
import csv
import gzip
import io
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
RETRIEVED = datetime.now(timezone.utc).isoformat()
SOURCE_ID = "3DLNews2_Arkansas"
SOURCE_URL = "https://g-e840e1.746abe.8540.data.globus.org/1-Google/1-Newspaper/preprocessed_state/AR"
NOTES = "Metadata only; article full text dropped. Keyword match on full text at build time."

KEYWORD_GROUPS = {
    "drug_shortage": [
        r"drug shortage", r"medicine shortage", r"medication shortage",
        r"shortage of (drug|medicine|medication)", r"pharmaceutical shortage",
    ],
    "pharmaceutical_supply_chain": [
        r"supply chain", r"(drug|medicine|medication|pharmaceutical|pharma) supply",
    ],
    "pharmacy_closure": [
        r"pharmac(y|ies) (clos|shut)", r"clos(ing|ed|es|e) (the )?(pharmacy|pharmacies)",
        r"closing (its|their|our) (drugstore|pharmacy)",
    ],
    "medication_access": [
        r"access to (medication|medicine|drug|prescription|treatment)",
        r"medication access", r"medicine access", r"afford(ing|able|s)? (their |his |her )?(medication|medicine|prescription|drugs)",
        r"can'?t afford", r"cannot afford", r"unaffordable",
    ],
    "fda_recall": [
        r"fda.{0,40}recall", r"drug.{0,20}recall", r"recall.{0,20}(drug|medication|medicine|pharmaceutical|injectable)",
        r"recalled", r"voluntary recall",
    ],
    "manufacturing_disruption": [
        r"manufactur", r"factory", r"production (disrupt|halt|delay|issue|problem|shortag|suspend)",
        r"manufactur.{0,40}(disrupt|halt|delay|issue|problem|shortag)",
    ],
    "active_pharmaceutical_ingredient": [
        r"active pharmaceutical ingredient", r"pharmaceutical ingredient",
        r"ingredient (shortag|supply)", r"active ingredient",
    ],
    "arkansas_pharmacy": [
        r"arkansas (state board of )?pharmacy", r"arkansas pharmacist",
        r"pharmac(y|ies) in arkansas", r"arkansas pharmacies",
        r"arkansas association of pharmacist",
    ],
    "arkansas_hospital": [
        r"arkansas hospital", r"arkansas hospitals", r"hospital in arkansas",
        r"hospitals in arkansas", r"arkansas medical", r"arkansas physician",
    ],
    "arkansas_medicaid": [
        r"arkansas medicaid", r"medicaid", r"healthcare independence program",
        r"arkansas works",
    ],
    "influenza": [r"influenza", r"\bflu\b", r"h1n1", r"flu (season|shot|vaccine)"],
    "rsv": [r"\brsv\b", r"respiratory syncytial"],
    "covid": [r"covid", r"coronavirus", r"sars-cov-2", r"omicron"],
    "tornado": [r"tornado", r"twister"],
    "flood": [r"flood"],
    "winter_storm": [
        r"winter storm", r"ice storm", r"snow ?storm", r"blizzard",
        r"winter weather", r"arctic (blast|air)", r"freezing",
    ],
}

GROUP_RE = {name: [re.compile(p, re.I) for p in pats] for name, pats in KEYWORD_GROUPS.items()}


def match_groups(text):
    return [name for name, pats in GROUP_RE.items() if any(p.search(text) for p in pats)]


def parse_date(date_str, year):
    if date_str:
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d"), dt.year, dt.month
        except ValueError:
            pass
    return None, int(year), None


def main():
    os.makedirs(DATA, exist_ok=True)
    meta_out = os.path.join(DATA, "arkansas_3dlnews_targeted_article_metadata.csv.gz")
    count_out = os.path.join(DATA, "arkansas_3dlnews_monthly_keyword_counts.csv.gz")

    meta_fields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes",
                   "year", "publication_date", "month", "title", "publication",
                   "media_type", "url", "expanded_url", "id", "city", "state",
                   "is_news_article", "response_code", "keyword_hit_summary"]
    count_fields = ["source_id", "source_url", "retrieved_at_utc", "extraction_notes",
                    "group", "year", "month", "article_count"]

    m_handle = gzip.open(meta_out, "wt", newline="")
    c_handle = gzip.open(count_out, "wt", newline="")
    m_writer = csv.DictWriter(m_handle, fieldnames=meta_fields)
    c_writer = csv.DictWriter(c_handle, fieldnames=count_fields)
    m_writer.writeheader()
    c_writer.writeheader()

    monthly = Counter()
    meta_rows = 0
    for year in range(2013, 2023):
        path = os.path.join(RAW, f"preprocessed_newspaper_articles_AR_{year}.jsonl.gz")
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = rec.get("content") or ""
                title = rec.get("title") or ""
                text = content + "\n" + title
                groups = match_groups(text)
                if not groups:
                    continue
                pub_date, pyear, pmonth = parse_date(rec.get("publication_date"), year)
                for g in groups:
                    monthly[(g, pyear, pmonth)] += 1
                if not title:
                    continue
                loc = rec.get("location") or {}
                m_writer.writerow({
                    "source_id": SOURCE_ID,
                    "source_url": SOURCE_URL,
                    "retrieved_at_utc": RETRIEVED,
                    "extraction_notes": NOTES,
                    "year": pyear,
                    "publication_date": pub_date,
                    "month": pmonth,
                    "title": title.strip()[:1000],
                    "publication": rec.get("media_name") or "",
                    "media_type": rec.get("media_type") or "",
                    "url": rec.get("link") or "",
                    "expanded_url": rec.get("expanded_url") or "",
                    "id": rec.get("id") or "",
                    "city": loc.get("city") or "",
                    "state": loc.get("state") or "",
                    "is_news_article": rec.get("is_news_article"),
                    "response_code": rec.get("response_code"),
                    "keyword_hit_summary": "|".join(groups),
                })
                meta_rows += 1
        print(f"processed {year}", flush=True)

    for (group, year, month), n in sorted(monthly.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        c_writer.writerow({"source_id": SOURCE_ID, "source_url": SOURCE_URL,
                           "retrieved_at_utc": RETRIEVED, "extraction_notes": NOTES,
                           "group": group, "year": year, "month": month, "article_count": n})

    m_handle.close()
    c_handle.close()
    print(f"metadata_rows={meta_rows} monthly_buckets={len(monthly)}")
    print("DONE")


if __name__ == "__main__":
    main()
