#!/usr/bin/env python3
"""Extract clean text from downloaded HTML files using trafilatura (parallel), write JSON."""
import json, gzip, os, re, sys
from multiprocessing import Pool
import trafilatura

RAW = "/home/ubuntu/meditrack_proj/medi_track/data/raw"
HTMLDIR = os.path.join(RAW, "html")
RECORDS = os.path.join(RAW, "records_2013_2022.jsonl")
OUT_ARK = "/home/ubuntu/meditrack_proj/medi_track/data/arkansas/data.json"
OUT_INT = "/home/ubuntu/meditrack_proj/medi_track/data/international/data.json"

AR_KEYWORDS = ["arkansas", "little rock", "arlington", "fort smith", "hot springs", "pine bluff", "jonesboro", "conway", "rogers", "springdale", "bentonville", "north little rock", "fayetteville"]

def classify(rec):
    st = (rec.get('state') or '').strip().lower()
    if st in ('ar', 'arkansas'):
        return 'ark'
    if st:
        return 'int'
    t = (rec.get('title') or '').lower()
    for kw in AR_KEYWORDS:
        if kw in t:
            return 'ark'
    return 'int'

def get_text(rel):
    path = os.path.join(HTMLDIR, rel)
    try:
        with gzip.open(path, 'rt', errors='ignore') as f:
            html = f.read()
    except Exception:
        return None
    try:
        txt = trafilatura.extract(html, include_comments=False, include_tables=False, include_links=False)
    except Exception:
        return None
    if not txt:
        return None
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt if len(txt) >= 20 else None

def process(rec):
    txt = get_text(rec['html'])
    if not txt:
        return None
    return (rec['date'], rec['title'], txt, classify(rec))

if __name__ == '__main__':
    records = [json.loads(l) for l in open(RECORDS) if l.strip()]
    print(f"processing {len(records)} records", flush=True)
    with Pool(4) as p:
        results = p.map(process, records, chunksize=500)
    print("extraction done", flush=True)

    ark, intl = {}, {}
    seen = {}
    skipped_empty = dedup = 0
    for r in results:
        if r is None:
            skipped_empty += 1
            continue
        date, title, txt, bucket = r
        norm = re.sub(r'[^\w]+', '', title.lower())
        if norm in seen:
            dedup += 1
            continue
        seen[norm] = bucket
        target = ark if bucket == 'ark' else intl
        target.setdefault(date, {})[title] = txt

    def write(path, data):
        ordered = {d: data[d] for d in sorted(data.keys())}
        with open(path, 'w') as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)

    write(OUT_ARK, ark)
    write(OUT_INT, intl)
    na = sum(len(v) for v in ark.values())
    ni = sum(len(v) for v in intl.values())
    print(f"arkansas={na} international={ni} total={na+ni} dedup={dedup} skipped_empty={skipped_empty}", flush=True)