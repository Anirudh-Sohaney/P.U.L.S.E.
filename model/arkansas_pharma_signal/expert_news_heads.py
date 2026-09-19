"""Grouped, multi-head NLP evaluation on the expert news annotation corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .expert_gold import load_cirad_expert_rows
from .news_relevance import vectorize_text
from .regression import LogisticRidge


HEADS = {
    "event_current": lambda row: row["event_type"] == "CE",
    "event_risk": lambda row: row["event_type"] == "RE",
    "event_current_or_risk": lambda row: row["event_type"] in {"CE", "RE"},
    "info_disease": lambda row: row["information_type"] == "DE",
    "info_prevention_control": lambda row: row["information_type"] == "PCM",
    "info_irrelevant": lambda row: row["information_type"] == "IR",
    "info_case_report": lambda row: row["information_type"] == "CRF",
    "info_geography": lambda row: row["information_type"] == "GE",
    "info_transmission": lambda row: row["information_type"] == "TP",
    "info_epidemiology": lambda row: row["information_type"] == "EPC",
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = actual.astype(bool)
    predicted = predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    return {
        "test_rows": int(len(actual)),
        "actual_positive_rows": int(actual.sum()),
        "predicted_positive_rows": int(predicted.sum()),
        "accuracy": float((actual == predicted).mean()),
        "true_positive_precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / max(int(actual.sum()), 1)),
    }


def evaluate_expert_news_heads(root: Path, *, train_fraction: float = 0.70) -> dict:
    """Train ten text heads with an article-disjoint deterministic split."""
    frame = load_cirad_expert_rows(root)
    article_ids = sorted(frame["article_id"].unique())
    split = max(1, min(len(article_ids) - 1, int(len(article_ids) * train_fraction)))
    train_articles = set(article_ids[:split])
    train = frame[frame["article_id"].isin(train_articles)]
    test = frame[~frame["article_id"].isin(train_articles)]
    x_train = vectorize_text(train["text"].tolist())
    x_test = vectorize_text(test["text"].tolist())
    results = []
    for name, labeler in HEADS.items():
        y_train = np.asarray([labeler(row) for row in train.to_dict("records")], dtype=float)
        y_test = np.asarray([labeler(row) for row in test.to_dict("records")], dtype=float)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        model = LogisticRidge(alpha=10.0, max_iter=80).fit(x_train, y_train)
        predicted = model.predict(x_test).astype(bool)
        result = _metrics(y_test, predicted)
        majority = np.full(len(y_test), y_train.mean() >= 0.5)
        result.update({"signal": name, "majority_accuracy": float((y_test == majority).mean()),
                       "beats_majority": bool((y_test == predicted).mean() > (y_test == majority).mean()),
                       "passes_accuracy": bool(result["accuracy"] >= 0.70),
                       "passes_true_positive_precision": bool(
                           result["true_positive_precision"] is not None and
                           result["true_positive_precision"] >= 0.70)})
        result["promotion_candidate"] = bool(
            result["beats_majority"] and
            (result["passes_accuracy"] or result["passes_true_positive_precision"]))
        results.append(result)
    return {
        "protocol": "article_grouped_cirad_multitask_text_heads_v1",
        "source": "CIRAD/PADI-web expert sentence annotations",
        "evaluation_scope": "animal-health news NLP transfer benchmark; not pharmacy truth",
        "feature_type": "hashed word unigrams and adjacent bigrams",
        "article_count": len(article_ids),
        "train_article_count": len(train_articles),
        "test_article_count": len(article_ids) - len(train_articles),
        "train_rows": len(train), "test_rows": len(test),
        "heads": results,
        "evaluated_head_count": len(results),
        "passing_count": int(sum(row["promotion_candidate"] for row in results)),
    }
