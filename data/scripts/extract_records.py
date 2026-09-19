#!/usr/bin/env python3
"""Parse preprocessed 3DLNews2 files for 2000-2001, filter valid articles, list needed HTML files."""
import json, gzip, glob, os, re, sys

RAW = "/home/ubuntu/meditrack_proj/medi_track/data/raw"
OUT = "/home/ubuntu/meditrack_proj/medi_track/data/raw/records_2013_2022.jsonl"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def parse_date(v):
    if not v: return None
    s = str(v).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if not m: return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (2013 <= y <= 2022): return None
    if not (1 <= mo <= 12 and 1 <= d <= 31): return None
    return f"{y:04d}-{mo:02d}-{d:02d}"

records = []
html_needed = set()
for year in range(2013, 2023):
    for f in sorted(glob.glob(os.path.join(RAW, f"preprocessed_newspaper_articles_*_{year}.jsonl.gz"))):
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                date = parse_date(d.get('publication_date'))
                if not date: continue
                title = (d.get('title') or '').strip()
                if not title: continue
                hf = d.get('html_filename')
                if not hf: continue
                rec = {
                    'date': date,
                    'title': title,
                    'html': hf,
                    'state': (d.get('location') or {}).get('state'),
                    'media': d.get('media_name'),
                }
                records.append(rec)
                html_needed.add(hf)

with open(OUT, 'w') as fh:
    for r in records:
        fh.write(json.dumps(r) + '\n')
with open(os.path.join(RAW, 'needed_html.txt'), 'w') as fh:
    for h in sorted(html_needed):
        fh.write(h + '\n')
print(f"records: {len(records)}  unique html: {len(html_needed)}")
