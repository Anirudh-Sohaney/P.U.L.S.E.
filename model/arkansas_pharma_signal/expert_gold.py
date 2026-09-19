"""Build an external expert-labeled event benchmark.

The benchmark is transfer evidence for news/event representation learning. It
is deliberately not a human-Arkansas pharmacy target: the PADI-web tables are
animal-health article-event annotations and the CIRAD table is animal-health
sentence annotation. Source and unit type remain in every row so the combined
count cannot be mistaken for one homogeneous label set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PADI_FILES = {
    "padi_avian_influenza": "targeted_additions/padi_expert/data/avian_influenza.tab",
    "padi_african_swine_fever": "targeted_additions/padi_expert/data/african_swine_fever.tab",
    "padi_west_nile_virus": "targeted_additions/padi_expert/data/west_nile_virus.tab",
}
CIRAD_PATH = "targeted_additions/epidemiology_annotation/cirad_padiweb_annotation.ods"
GOLD_COLUMNS = [
    "unit_id", "article_id", "source_dataset", "unit_type", "published_at",
    "title", "text", "label", "event_type", "information_type",
    "event_positive", "disease", "location",
    "source_url", "annotation_provenance",
]


def normalize_padi_label(value: object) -> tuple[str, bool]:
    """Map PADI's documented relevance classes to a binary event benchmark."""
    label = " ".join(str(value or "").strip().casefold().split())
    if "irrelevant" in label:
        return "irrelevant", False
    if "event" in label:
        return "current_or_risk_event", True
    if "general" in label:
        return "general_information", False
    if label:
        return label, False
    return "missing", False


def _unit_id(source: str, article_id: str, ordinal: int) -> str:
    raw = f"{source}\x1f{article_id}\x1f{ordinal}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def load_padi_expert_rows(root: Path) -> pd.DataFrame:
    """Load the three public PADI-web article-event tables without deduping events."""
    frames = []
    for source, relative in PADI_FILES.items():
        path = root / "data" / relative
        if not path.exists():
            raise FileNotFoundError(f"PADI expert table missing: {path}")
        raw = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
        columns = {str(column).casefold(): column for column in raw.columns}
        def col(name: str) -> pd.Series:
            actual = columns.get(name.casefold())
            return raw[actual].astype("string") if actual else pd.Series("", index=raw.index, dtype="string")

        rows = []
        for ordinal, (_, record) in enumerate(raw.iterrows()):
            article_id = str(record.get(columns.get("alertid", ""), "")).strip()
            title = str(record.get(columns.get("title", ""), "")).strip()
            url = str(record.get(columns.get("url", ""), "")).strip()
            label, positive = normalize_padi_label(
                record.get(columns.get("manualclass", ""), ""))
            if label == "missing" or not (article_id or url or title):
                continue
            disease = str(record.get(columns.get("diseasename", ""), "")).strip()
            location = str(record.get(
                columns.get("placename", columns.get("continent/country", "")), "")).strip()
            frames.append({
                "unit_id": _unit_id(source, article_id or url or title, ordinal),
                "article_id": article_id or url or title,
                "source_dataset": source,
                "unit_type": "article_event",
                "published_at": "",
                "title": title,
                "text": title,
                "label": label,
                "event_type": "",
                "information_type": "",
                "event_positive": bool(positive),
                "disease": disease,
                "location": location,
                "source_url": url,
                "annotation_provenance": "animal-health; two epidemiologists; PADI-web manual relevance annotation",
            })
    return pd.DataFrame(frames, columns=GOLD_COLUMNS)


def load_cirad_expert_rows(root: Path) -> pd.DataFrame:
    """Load CIRAD sentence consensus labels as a separate unit type."""
    from .external_validation import _read_ods_table

    path = root / "data" / CIRAD_PATH
    if not path.exists():
        raise FileNotFoundError(f"CIRAD annotation corpus missing: {path}")
    rows = []
    source = "cirad_padiweb_sentences"
    raw_rows = (_read_ods_table(path, "annot_sentences_pairs") +
                _read_ods_table(path, "annot_sentences_single"))
    for ordinal, record in enumerate(raw_rows):
        article_id = str(record.get("id_article", "")).strip()
        text = str(record.get("sentence_text", "")).strip()
        label = str(record.get("event_type", "")).strip()
        if not article_id or not text or not label:
            continue
        rows.append({
            "unit_id": _unit_id(source, article_id, ordinal),
            "article_id": article_id,
            "source_dataset": source,
            "unit_type": "sentence_event",
            "published_at": "",
            "title": "",
            "text": text,
            "label": label,
            "event_type": label,
            "information_type": str(record.get("information_type", "")).strip(),
            "event_positive": label in {"CE", "RE"},
            "disease": "",
            "location": "",
            "source_url": "",
            "annotation_provenance": "animal-health; CIRAD/PADI-web expert consensus or single-annotator label",
        })
    return pd.DataFrame(rows, columns=GOLD_COLUMNS)


def build_expert_event_gold(root: Path) -> pd.DataFrame:
    """Return the auditable combined external event benchmark."""
    frame = pd.concat([
        load_padi_expert_rows(root), load_cirad_expert_rows(root),
    ], ignore_index=True)
    if frame.empty or frame["unit_id"].duplicated().any():
        raise ValueError("expert event benchmark must contain unique labeled units")
    return frame


def expert_event_gold_metadata(frame: pd.DataFrame) -> dict:
    """Summarize provenance and unit counts for the publishability audit."""
    return {
        "rows": int(len(frame)),
        "unique_articles": int(frame["article_id"].nunique()),
        "source_datasets": sorted(frame["source_dataset"].unique().tolist()),
        "unit_types": frame["unit_type"].value_counts().to_dict(),
        "positive_rows": int(frame["event_positive"].astype(bool).sum()),
        "annotation_provenance": "external animal-health transfer benchmark; not Arkansas human-pharmacy truth",
        "training_use": "evaluation_only",
    }
