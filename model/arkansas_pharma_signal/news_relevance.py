"""Weakly supervised learned relevance scoring for incoming news text.

The PADI-web labels are article-relevance labels, not disease, shortage, or
forecast labels. This module trains only on that weak corpus and keeps the
result as a Layer 1 relevance probability for downstream learned models.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .regression import LogisticRidge

RELATIVE_FILES = (
    "data/targeted_additions/padi_web_weak/relevant_articles.tab",
    "data/targeted_additions/padi_web_weak/irrelevant_articles.tab",
)
DEFAULT_DIMENSION = 512
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_weak_news(root: Path) -> pd.DataFrame:
    """Load the public weak labels with publication timestamps intact."""
    frames = []
    for relative in RELATIVE_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"weak news source missing: {path}")
        frame = pd.read_csv(path, sep="\t", usecols=["title", "source", "publishedat"],
                            low_memory=False)
        frame["label"] = 1 if path.name == "relevant_articles.tab" else 0
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["published_at"] = pd.to_datetime(out["publishedat"], errors="coerce", utc=True)
    out["text"] = (out["title"].fillna("").astype(str) + " "
                   + out["source"].fillna("").astype(str)).str.strip()
    out = out[out["published_at"].notna() & out["text"].ne("")].copy()
    return out.sort_values("published_at").reset_index(drop=True)


def _bucket(token: str, dimension: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % dimension


def vectorize_text(texts, dimension: int = DEFAULT_DIMENSION) -> np.ndarray:
    """Hash word unigrams and adjacent bigrams into a fixed ML feature space."""
    matrix = np.zeros((len(texts), dimension), dtype=float)
    for i, text in enumerate(texts):
        tokens = TOKEN_RE.findall(str(text).lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for token in features:
            matrix[i, _bucket(token, dimension)] += 1.0
        if features:
            matrix[i] /= float(len(features))
    return matrix


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = p >= 0.5
    yb = y.astype(bool)
    tp = int(np.sum(pred & yb)); tn = int(np.sum(~pred & ~yb))
    fp = int(np.sum(pred & ~yb)); fn = int(np.sum(~pred & yb))
    return {
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "balanced_accuracy": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "positive_rate": float(np.mean(pred)),
    }


def train_news_relevance(root: Path, train_end: str = "2022-03-31",
                         validation_end: str = "2022-05-31",
                         dimension: int = DEFAULT_DIMENSION) -> Dict:
    data = load_weak_news(root)
    train_cut = pd.Timestamp(train_end, tz="UTC")
    val_cut = pd.Timestamp(validation_end, tz="UTC")
    train = data[data["published_at"] <= train_cut]
    validation = data[(data["published_at"] > train_cut) & (data["published_at"] <= val_cut)]
    test = data[data["published_at"] > val_cut]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("weak-news chronological split is empty")
    x_train = vectorize_text(train["text"], dimension)
    x_val = vectorize_text(validation["text"], dimension)
    x_test = vectorize_text(test["text"], dimension)
    names = [f"hash_{i}" for i in range(dimension)]
    model = LogisticRidge(alpha=10.0, max_iter=40).fit(
        x_train, train["label"].to_numpy(float), names)
    metrics = {
        "validation": _binary_metrics(validation["label"].to_numpy(float), model.predict_proba(x_val)),
        "test": _binary_metrics(test["label"].to_numpy(float), model.predict_proba(x_test)),
    }
    artifact = {
        "model": model.to_dict(), "feature_dimension": dimension,
        "feature_type": "hashed_word_unigram_bigram_count_normalized",
        "source_files": list(RELATIVE_FILES), "training_use": "weak_supervision_only",
        "label_definition": "PADI-web article relevance label; not event or forecast truth",
        "split": {"train_end": train_end, "validation_end": validation_end,
                  "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "metrics": metrics,
    }
    artifact["independent_cirad"] = evaluate_cirad_relevance(root, artifact)
    return artifact


def relevance_probability(text: str, artifact: Dict) -> float:
    """Infer relevance; empty/unknown text returns a neutral probability."""
    if not str(text or "").strip():
        return 0.5
    dimension = int(artifact.get("feature_dimension", DEFAULT_DIMENSION))
    model = LogisticRidge.from_dict(artifact["model"])
    return float(model.predict_proba(vectorize_text([text], dimension))[0])


def score_corpus(corpus: pd.DataFrame, artifact: Dict) -> pd.DataFrame:
    """Score article text for Layer 1, retaining only provenance keys."""
    if "article_id" not in corpus.columns:
        raise ValueError("corpus must contain article_id")
    text = (corpus.get("title", pd.Series("", index=corpus.index)).fillna("").astype(str)
            + " " + corpus.get("body", pd.Series("", index=corpus.index)).fillna("").astype(str))
    dimension = int(artifact.get("feature_dimension", DEFAULT_DIMENSION))
    model = LogisticRidge.from_dict(artifact["model"])
    scores = model.predict_proba(vectorize_text(text.tolist(), dimension))
    return pd.DataFrame({"article_id": corpus["article_id"].astype(str),
                         "relevance_probability": scores})


def evaluate_cirad_relevance(root: Path, artifact: Dict) -> Dict:
    """Evaluate weakly-trained relevance on expert CIRAD sentences only."""
    from .external_validation import _read_ods_table
    path = root / "data/targeted_additions/epidemiology_annotation/cirad_padiweb_annotation.ods"
    if not path.exists():
        return {"available": False, "reason": "CIRAD annotation corpus missing"}
    rows = (_read_ods_table(path, "annot_sentences_pairs") +
            _read_ods_table(path, "annot_sentences_single"))
    y = np.asarray([str(r.get("event_type", "")) in {"CE", "RE"} for r in rows], dtype=float)
    x = vectorize_text([r.get("sentence_text", "") for r in rows],
                       int(artifact.get("feature_dimension", DEFAULT_DIMENSION)))
    model = LogisticRidge.from_dict(artifact["model"])
    return {"available": True, "source": "CIRAD PADI-web expert annotation",
            "sentence_count": len(rows), "positive_definition": "CE or RE",
            "metrics": _binary_metrics(y, model.predict_proba(x)),
            "training_use": "evaluation_only"}
