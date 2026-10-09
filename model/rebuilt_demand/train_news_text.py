"""Explore an article-conditioned replacement for the archived 20 outputs.

This is a deliberately separate experiment from the temporal replacement.
Only months with actual local GDELT full text are eligible, and the resulting
10 monthly examples are too few to justify production promotion.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from train import MODEL_DIR, NEWS_PATH, OUT_DIR
from arkansas_pharma_signal.news_only_adapter import NEWS_ONLY_SIGNAL_IDS

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data/targeted_additions/news_article_corpus/data/articles.jsonl.gz"


def load_monthly_corpus(path: Path = CORPUS_PATH) -> pd.DataFrame:
    monthly: dict[pd.Timestamp, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            article = json.loads(line)
            body = str(article.get("body") or "").strip()
            title = str(article.get("title") or "").strip()
            if not article.get("is_full_text") or not body or not article.get("published_at"):
                continue
            date = pd.to_datetime(article["published_at"], unit="ms", utc=True).tz_convert(None)
            month = date.to_period("M").to_timestamp()
            monthly.setdefault(month, []).append(f"{title}. {body}")
    return pd.DataFrame({"date": sorted(monthly),
                         "text": ["\n".join(monthly[month]) for month in sorted(monthly)],
                         "article_count": [len(monthly[month]) for month in sorted(monthly)]})


def signal_errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    per_signal_mae = np.mean(np.abs(actual - predicted), axis=0)
    per_signal_wape = np.abs(actual - predicted).sum(axis=0) / np.maximum(np.abs(actual).sum(axis=0), 1e-9)
    return {
        "months": int(actual.shape[0]),
        "signals": int(actual.shape[1]),
        "mean_signal_mae": float(per_signal_mae.mean()),
        "mean_signal_wape": float(per_signal_wape.mean()),
        "pooled_wape": float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1e-9)),
    }


def main() -> None:
    article_months = load_monthly_corpus()
    archive = pd.read_csv(NEWS_PATH, parse_dates=["date"])
    archive["date"] = archive.date.dt.to_period("M").dt.to_timestamp()
    matched = article_months.merge(archive, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if list(archive.columns) != ["date", *NEWS_ONLY_SIGNAL_IDS]:
        raise ValueError("Archived 20-signal output schema mismatch")
    n = len(matched)
    train_end, validation_end = int(n * .60), int(n * .80)
    if n < 8:
        raise ValueError(f"Article-conditioned 20-output training requires at least 8 months; found {n}")
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=1, max_features=1500, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(matched.text.iloc[:train_end])
    x_validation = vectorizer.transform(matched.text.iloc[train_end:validation_end])
    x_test = vectorizer.transform(matched.text.iloc[validation_end:])
    y = matched[list(NEWS_ONLY_SIGNAL_IDS)].to_numpy(dtype=float)
    candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y[:train_end])
        prediction = model.predict(x_validation)
        candidates.append({"alpha": alpha, "validation": signal_errors(
            y[train_end:validation_end], prediction
        )})
    best = min(candidates, key=lambda result: result["validation"]["mean_signal_wape"])
    final_vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=1, max_features=1500, sublinear_tf=True,
    )
    x_all = final_vectorizer.fit_transform(matched.text)
    final_model = Ridge(alpha=best["alpha"]).fit(x_all, y)
    x_final = final_vectorizer.transform(matched.text.iloc[validation_end:])
    test_prediction = np.maximum(final_model.predict(x_final), 0)
    test_metrics = signal_errors(y[validation_end:], test_prediction)
    persistence = matched[list(NEWS_ONLY_SIGNAL_IDS)].shift(1).iloc[validation_end:].to_numpy(dtype=float)
    persistence_metrics = signal_errors(y[validation_end:], persistence)

    model_dir = MODEL_DIR / "initial_20_signals"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": final_vectorizer, "model": final_model,
                 "signal_ids": list(NEWS_ONLY_SIGNAL_IDS)}, model_dir / "article_text_model.joblib")
    prediction_rows = pd.DataFrame(test_prediction, columns=NEWS_ONLY_SIGNAL_IDS)
    prediction_rows.insert(0, "date", matched.date.iloc[validation_end:].dt.strftime("%Y-%m-%d").to_numpy())
    actual_rows = pd.DataFrame(y[validation_end:], columns=[f"actual__{key}" for key in NEWS_ONLY_SIGNAL_IDS])
    pd.concat([prediction_rows.reset_index(drop=True), actual_rows], axis=1).to_csv(
        OUT_DIR / "article_text_20_signal_test_predictions.csv", index=False
    )
    result = {
        "model": "TF-IDF word/bigram features with multi-output Ridge regression",
        "target": "same-month archived 20 signal values; text is available after month end",
        "corpus_path": str(CORPUS_PATH),
        "article_records": 354,
        "full_text_records": int(article_months.article_count.sum()),
        "matched_months": n,
        "matched_date_range": [str(matched.date.min().date()), str(matched.date.max().date())],
        "months_with_text": int(n),
        "train_months": train_end,
        "validation_months": validation_end - train_end,
        "test_months": n - validation_end,
        "validation_candidates": candidates,
        "selected_alpha": best["alpha"],
        "test": test_metrics,
        "persistence_test": persistence_metrics,
        "production_status": "exploratory_only: 10 sparse month aggregates are not enough for a dependable news-to-20-signal model",
        "saved_model": str(model_dir / "article_text_model.joblib"),
    }
    (OUT_DIR / "article_text_20_signal_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
