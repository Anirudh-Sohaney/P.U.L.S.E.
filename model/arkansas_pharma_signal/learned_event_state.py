"""Research-only supervised event-state classifier.

The CIRAD/PADI-web annotations provide expert sentence labels for current and
risk events.  They are useful for testing a learned replacement candidate for
the deterministic event trigger, but they are animal-health data and are not
production truth for Arkansas pharmacy news.  This module therefore exposes
training and scoring utilities without silently joining scores to Layer 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .news_relevance import _binary_metrics, vectorize_text
from .regression import LogisticRidge


POSITIVE_LABELS = {"CE", "RE"}
ANNOTATION_RELATIVE_PATH = "data/targeted_additions/epidemiology_annotation/cirad_padiweb_annotation.ods"


def load_cirad_event_sentences(root: Path) -> list[dict[str, str]]:
    """Load CIRAD labels through the shared ODS reader."""
    from .external_validation import _read_ods_table

    path = root / ANNOTATION_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"CIRAD annotation corpus missing: {path}")
    from_pairs = _read_ods_table(path, "annot_sentences_pairs")
    from_single = _read_ods_table(path, "annot_sentences_single")
    rows = []
    for row in from_pairs + from_single:
        text = str(row.get("sentence_text", "")).strip()
        article = str(row.get("id_article", "")).strip()
        label = str(row.get("event_type", "")).strip()
        if text and article and label:
            rows.append({"text": text, "article_id": article, "label": label})
    if not rows:
        raise ValueError("CIRAD annotation corpus contains no usable sentences")
    return rows


def _grouped_split(rows: list[dict[str, str]], test_fraction: float = 0.25):
    """Split by article ID so sentences from one article cannot leak."""
    articles = sorted({row["article_id"] for row in rows})
    cut = max(1, min(len(articles) - 1, int(len(articles) * (1 - test_fraction))))
    train_ids, test_ids = set(articles[:cut]), set(articles[cut:])
    train = [row for row in rows if row["article_id"] in train_ids]
    test = [row for row in rows if row["article_id"] in test_ids]
    return train, test


def train_cirad_event_state(root: Path, dimension: int = 512) -> Dict:
    """Train a research candidate and report an article-grouped holdout."""
    rows = load_cirad_event_sentences(root)
    train, test = _grouped_split(rows)
    y_train = np.asarray([row["label"] in POSITIVE_LABELS for row in train], dtype=float)
    y_test = np.asarray([row["label"] in POSITIVE_LABELS for row in test], dtype=float)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("CIRAD grouped split must contain both event classes")
    model = LogisticRidge(alpha=10.0, max_iter=40).fit(
        vectorize_text([row["text"] for row in train], dimension), y_train,
        [f"hash_{i}" for i in range(dimension)])
    probabilities = model.predict_proba(vectorize_text([row["text"] for row in test], dimension))
    return {
        "model": model.to_dict(),
        "feature_dimension": dimension,
        "feature_type": "hashed_word_unigram_bigram_count_normalized",
        "source": ANNOTATION_RELATIVE_PATH,
        "positive_definition": "CIRAD CE or RE (current or risk event)",
        "training_use": "research_only_animal_health_transfer",
        "split": {"article_count": len({row["article_id"] for row in rows}),
                  "train_articles": len({row["article_id"] for row in train}),
                  "test_articles": len({row["article_id"] for row in test}),
                  "train_rows": len(train), "test_rows": len(test)},
        "metrics": _binary_metrics(y_test, probabilities),
    }


def event_state_probability(text: str, artifact: Dict) -> float:
    """Return a bounded research score; empty text is neutral."""
    if not str(text or "").strip():
        return 0.5
    dimension = int(artifact.get("feature_dimension", 512))
    model = LogisticRidge.from_dict(artifact["model"])
    return float(model.predict_proba(vectorize_text([text], dimension))[0])

