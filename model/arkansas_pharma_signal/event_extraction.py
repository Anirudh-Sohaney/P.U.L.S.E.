"""Deterministic, auditable event extraction baseline over real article text.

This is the fallback extractor for environments without a transformer runtime.
Every emitted field is tied to a matched article span and is intentionally
conservative; it is an intelligence-layer signal, not a forecast label.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from .entities import normalize_name
from .event_schema import EVENT_COLUMNS, validate_events

EVENT_RULES = {
    "shortage": (r"\b(shortage|shortages|supply disruption|unavailable|out of stock)\b", "supply_negative", "demand_positive"),
    "recall": (r"\b(recall|recalled|contamination|contaminated|warning letter)\b", "supply_negative", "demand_positive"),
    # BAND's expert outbreak articles often describe the event with a
    # clinical consequence ("tested positive", "infected", "killed", or
    # "contagious") without using the literal word outbreak.  These terms
    # deliberately remain observable triggers; downstream uncertainty and
    # geography fields keep the signal from being treated as confirmation.
    "disease_outbreak": (r"\b(outbreak|epidemic|surge|cases|hospitalization|respiratory|"
                          r"positive|infected|infection|contagious|transmission|"
                          r"transmitted|cluster|sick|illness|died|death|fatal|deaths|"
                          r"killed|hospitalized|confirmed|symptoms|victims)\b",
                          "supply_neutral", "demand_positive"),
    "disaster": (r"\b(flood|tornado|hurricane|wildfire|disaster|storm)\b", "supply_negative", "demand_positive"),
    "policy_trade": (r"\b(tariff|sanction|regulation|legislation|Medicaid|FDA approval)\b", "supply_uncertain", "demand_uncertain"),
}
NEGATION = re.compile(r"\b(no|not|without|denies|denied|unlikely|never)\b", re.I)
UNCERTAINTY = re.compile(r"\b(may|might|could|possible|possibly|potential|reportedly|alleged)\b", re.I)


def _span(text: str, match: re.Match, width: int = 240) -> str:
    lo = max(0, match.start() - width // 2)
    hi = min(len(text), match.end() + width // 2)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def _drug_mentions(text: str, by_first_token: dict[str, list[str]] | None) -> list[str]:
    if not by_first_token:
        return []
    lower = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lower))
    found = []
    for token in tokens:
        for name in by_first_token.get(token, []):
            if re.search(r"\b" + re.escape(name) + r"\b", lower):
                found.append(name)
    return list(dict.fromkeys(found))


def extract_events(corpus: pd.DataFrame, drug_names: list[str] | None = None,
                   city_to_county: dict[str, dict] | None = None) -> pd.DataFrame:
    """Extract events only from available full-text article bodies/titles."""
    drug_names = sorted({normalize_name(x) for x in (drug_names or []) if normalize_name(x)},
                        key=len, reverse=True)
    drug_by_first = {}
    for name in drug_names:
        drug_by_first.setdefault(name.split()[0], []).append(name)
    rows = []
    for _, article in corpus[corpus["is_full_text"].fillna(False)].iterrows():
        text = f"{article.get('title', '')}. {article.get('body', '')}".strip()
        if not text:
            continue
        article_drugs = ",".join(_drug_mentions(text, drug_by_first))
        for event_type, (pattern, supply, demand) in EVENT_RULES.items():
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            span = _span(text, match)
            context = text[max(0, match.start() - 100):match.end() + 100]
            negated = bool(NEGATION.search(context))
            uncertain = bool(UNCERTAINTY.search(context))
            city = str(article.get("city", "") or "")
            geo = (city_to_county or {}).get(city.lower(), {})
            article_id = str(article.get("article_id", ""))
            event_id = hashlib.sha256(f"{article_id}:{event_type}:{match.start()}".encode()).hexdigest()[:20]
            rows.append({
                "event_id": event_id, "article_id": article_id, "event_type": event_type,
                "event_subtype": event_type, "event_start": article.get("published_at"),
                "event_end": "", "location": city, "county_fips": geo.get("county_fips", ""),
                "disease": "", "variant": "", "drug": article_drugs,
                "therapeutic_class": "", "supplier": "", "parent_company": "",
                "factory": "", "api_source": "", "supply_impact": supply,
                "demand_impact": demand, "severity": 0.5 if uncertain else 0.7,
                "probability": 0.45 if uncertain else 0.75, "lead_time_days": "",
                "evidence_span": span, "negation_status": "negated" if negated else "affirmed",
                "uncertainty_status": "uncertain" if uncertain else "asserted",
                "causal_status": "unknown", "source": article.get("source", ""),
                "source_timestamp": article.get("published_at"),
                "effective_date": article.get("published_at"), "confidence": 0.55 if uncertain else 0.7,
                "evidence_type": "article_span_rule",
            })
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.drop_duplicates("event_id").reset_index(drop=True)
    validate_events(events)
    return events
