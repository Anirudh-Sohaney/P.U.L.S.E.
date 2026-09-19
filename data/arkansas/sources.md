# Arkansas Sources

Source for all Arkansas news articles in `data.json` (years 2013-2022):

- 3DLNews2 (A Dataset of 4 Million Long-Form News Articles): https://github.com/wm-newslab/3DLNews2
- Paper: https://arxiv.org/abs/2408.04716
- Extracted from Google 1-Newspaper collection for state `AR` (`preprocessed_state/AR/preprocessed_newspaper_articles_AR_{YEAR}.jsonl.gz`), via Globus HTTPS.
- Pipeline: `data/scripts/` (`download_3dlnews.py`, `extract_records.py`, `download_html.py`, `extract_text.py`).
