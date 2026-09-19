# Mixed 2023–2025 news article corpus

This corpus contains dated English-language article records collected through
the GDELT DOC 2.0 Article List API for the same calendar years as the supplied
synthetic pharmacy sales. It intentionally contains two classes:

- `related`: pharmaceutical, medical, pharmacy, disease, shortage, policy,
  pricing, and supply-chain news;
- `unrelated`: sports, arts/entertainment, technology, travel, and food news.

The core file is `data/articles.jsonl.gz`. Each record uses the field names
expected by `model/arkansas_pharma_signal/news_corpus.py`, including
`article_id`, `title`, `body`, `published_at`, `source`, `url`, `language`,
`is_full_text`, `duplicate_id`, and `source_reliability`. The two audit fields
`topic_label` and `query_key` are retained for filtering and signal-cleanliness
experiments.

The collector stores raw GDELT responses under `raw/`, fetches article pages,
extracts readable text with Trafilatura, normalizes whitespace, hashes content
for deduplication, and preserves metadata-only records when a page cannot be
retrieved. It never invents article text or dates.

The corpus is a research input, not a labeled demand target. `related` and
`unrelated` are query-intent labels, not human editorial judgments. GDELT's
`seendate` is the discovery/seen timestamp; users requiring publication-time
backtesting should replace it with a verified publisher publication timestamp
where available.

Rebuild from the repository root with:

```bash
python3 data/targeted_additions/news_article_corpus/scripts/collect_corpus.py
```

Source: <https://www.gdeltproject.org/>.
