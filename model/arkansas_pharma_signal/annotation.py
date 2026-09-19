"""Create a real-text annotation manifest; never fills expert labels."""

from __future__ import annotations

import hashlib

import pandas as pd

ANNOTATION_COLUMNS = ["annotation_id", "article_id", "published_at", "source", "title", "body",
                      "candidate_event_type", "label_status", "annotator", "event_type_gold",
                      "entities_gold_json", "relations_gold_json", "temporal_gold_json",
                      "negation_gold", "impact_gold", "evidence_span_gold", "notes"]


def build_annotation_manifest(corpus: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    """Select up to n real full-text articles, stratified by rule candidate class."""
    c = corpus[corpus["is_full_text"].fillna(False)].copy()
    if c.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    text = (c["title"].fillna("") + " " + c["body"].fillna("")).str.lower()
    c["candidate_event_type"] = "other"
    for label, terms in {
        "shortage": ["shortage", "out of stock", "unavailable"],
        "recall": ["recall", "contamination", "warning letter"],
        "disease_outbreak": ["outbreak", "surge", "hospitalization", "cases"],
        "disaster": ["flood", "tornado", "wildfire", "storm"],
        "policy_trade": ["tariff", "sanction", "regulation", "medicaid"],
    }.items():
        c.loc[text.str.contains("|".join(terms), regex=True), "candidate_event_type"] = label
    pieces = []
    per_class = max(1, n // max(c["candidate_event_type"].nunique(), 1))
    for _, group in c.groupby("candidate_event_type", sort=True):
        pieces.append(group.sort_values(["published_at", "article_id"]).head(per_class))
    selected = pd.concat(pieces).drop_duplicates("article_id").head(n)
    if len(selected) < min(n, len(c)):
        remaining = c[~c["article_id"].isin(selected["article_id"])]
        selected = pd.concat([selected, remaining]).drop_duplicates("article_id").head(n)
    rows = []
    for _, r in selected.iterrows():
        aid = str(r["article_id"])
        rows.append({"annotation_id": hashlib.sha256(aid.encode()).hexdigest()[:20],
                     "article_id": aid, "published_at": r.get("published_at"),
                     "source": r.get("source", ""), "title": r.get("title", ""),
                     "body": r.get("body", ""), "candidate_event_type": r.get("candidate_event_type", "other"),
                     "label_status": "unlabeled", "annotator": "", "event_type_gold": "",
                     "entities_gold_json": "", "relations_gold_json": "", "temporal_gold_json": "",
                     "negation_gold": "", "impact_gold": "", "evidence_span_gold": "", "notes": ""})
    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
