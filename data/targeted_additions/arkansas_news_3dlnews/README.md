# 3DLNews2 — Arkansas Newspaper Articles (targeted keyword subset)

## Source
- Dataset: **3DLNews2** (Media Cloud successor), Google News newspaper archive, preprocessed per-state article files.
- Provider host: `g-e840e1.746abe.8540.data.globus.org` (Globus data portal)
- Base path: `/1-Google/1-Newspaper/preprocessed_state/AR/`
- File pattern: `preprocessed_newspaper_articles_AR_{YEAR}.jsonl.gz`, **YEAR 2013–2022** (10 files).
- Exact URLs in `source_manifest.json`.

## Retrieval
- Retrieved at (UTC): **2026-08-12**
- Downloader: `scripts/download_3dlnews_ar.py` (curl, 4-way parallel, retries to `raw/`).
- All 10 yearly files downloaded successfully into `raw/`.

## Files
- `raw/preprocessed_newspaper_articles_AR_{year}.jsonl.gz` — source JSONL (gzipped), one JSON object per line.
- `data/arkansas_3dlnews_targeted_article_metadata.csv.gz` — **1,246 metadata rows** (articles with ≥1 keyword hit **and** a title).
- `data/arkansas_3dlnews_monthly_keyword_counts.csv.gz` — **507 group×month buckets** (article counts by keyword group and publication year-month).

**Article full text is NOT stored** — only metadata plus a keyword-hit summary. Raw files in `raw/` retain full text if downstream need re-scoring.

## Normalisation (`scripts/build_targeted_metadata.py`)
- Parses JSONL safely (per-line `json.loads`, tolerant of bad lines/encodings).
- Keyword match runs case-insensitively against article `content` **and** `title`; groups are non-exclusive (article can hit several groups).
- Metadata rows require a non-empty `title`; matched articles lacking a title are counted in the monthly file only (per spec).
- Dates: `publication_date` → `publication_date`, `year`, `month`; blank dates keep file-year and empty month.
- Provenance columns on every output: `source_id`, `source_url`, `retrieved_at_utc`, `extraction_notes`.

## Keyword groups (exactly 16)
`drug_shortage`, `pharmaceutical_supply_chain`, `pharmacy_closure`, `medication_access`, `fda_recall`, `manufacturing_disruption`, `active_pharmaceutical_ingredient`, `arkansas_pharmacy`, `arkansas_hospital`, `arkansas_medicaid`, `influenza`, `rsv`, `covid`, `tornado`, `flood`, `winter_storm`.

## Metadata fields
`source_id, source_url, retrieved_at_utc, extraction_notes, year, publication_date, month, title, publication, media_type, url, expanded_url, id, city, state, is_news_article, response_code, keyword_hit_summary`

## Monthly fields
`source_id, source_url, retrieved_at_utc, extraction_notes, group, year, month, article_count`

## Counts by group (total article-hits, monthly file)
covid 630 · manufacturing_disruption 208 · flood 163 · fda_recall 152 · tornado 88 · winter_storm 74 · pharmaceutical_supply_chain 51 · arkansas_medicaid 41 · arkansas_hospital 33 · influenza 23 · medication_access 21 · active_pharmaceutical_ingredient 5 · arkansas_pharmacy 3 · pharmacy_closure 2 · **drug_shortage 0** · **rsv 0**

## Raw record counts by year (total articles, all topics)
2013: 14 · 2014: 11 · 2015: 31 · 2016: 1,568 · 2017: 1,688 · 2018: 1,832 · 2019: 1,948 · 2020: 2,150 · 2021: 2,257 · 2022: 2,190 — **Total: 13,689**.

## Metadata rows by publication year
2013: 5 · 2014: 2 · 2015: 4 · 2016: 68 · 2017: 71 · 2018: 68 · 2019: 97 · 2020: 356 · 2021: 355 · 2022: 214 · 2023: 4 · 2024: 1 · 2025: 1

## Known limitations
- **Coverage is thin before 2016** (2013–2015 files hold only 14/11/31 records — the 3DLNews2 archive has few Arkansas papers digitised for those years).
- **Date range drift**: a handful of articles carry `publication_date` outside the file's year (2023–2025 rows appear inside 2016–2022 files); dates are preserved as published and bucketed by actual year-month.
- 84 of 1,246 metadata rows have a blank `publication_date` (counted under file-year with empty month).
- **No title → no metadata row.** ~untracked matched articles appear only in the monthly counts.
- Keyword matching is broad/substring-based; expect false positives, e.g. `fda_recall` matches military "recalled", `flood` matches "flooded" in any context, `manufactur*` matches any manufacturing text. Precision for `drug_shortage` and `rsv` is nil — **zero hits** in the corpus.
- `response_code` preserved per row: some 200 responses are CAPTCHA/404 pages (preprocessed archive artifacts); rows with non-200 codes should be filtered by consumers.
- Raw files are 3DLNews2 preprocessed outputs (as-is); `source_metadata`/`page_details` blobs were dropped in normalisation.
