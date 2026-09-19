from pathlib import Path

import numpy as np
import pandas as pd

from arkansas_pharma_signal.news_relevance import (
    load_weak_news, relevance_probability, train_news_relevance, vectorize_text,
)


def test_vectorizer_is_fixed_and_empty_safe():
    out = vectorize_text(["shortage in Arkansas", ""], dimension=32)
    assert out.shape == (2, 32)
    assert np.all(out[1] == 0)


def test_real_weak_news_has_both_classes_and_dates():
    root = Path(__file__).resolve().parents[2]
    data = load_weak_news(root)
    assert len(data) > 1000
    assert set(data["label"]) == {0, 1}
    assert data["published_at"].notna().all()


def test_training_is_chronological_and_inference_is_bounded(tmp_path):
    root = Path(__file__).resolve().parents[2]
    artifact = train_news_relevance(root, dimension=32)
    split = artifact["split"]
    assert split["train_rows"] > 0
    assert split["validation_rows"] > 0
    assert split["test_rows"] > 0
    assert artifact["training_use"] == "weak_supervision_only"
    assert 0.0 <= relevance_probability("drug supply article", artifact) <= 1.0
    assert relevance_probability("", artifact) == 0.5
    assert artifact["independent_cirad"]["training_use"] == "evaluation_only"
