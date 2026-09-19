# Archived FDA Drug Shortage Panel

This directory stores dated Wayback Machine captures of the FDA downloadable
drug-shortage CSV endpoint and a deterministic builder for a monthly national
shortage-status panel.

The panel is a supply-risk target, not Arkansas pharmacy inventory. It has no
county allocation and does not establish that a product was available when it
was absent from a snapshot. Products without an observed resolution are
right-censored at the last captured month.

## Rebuild

```bash
python data/targeted_additions/fda_shortage_archive/scripts/build_fda_shortage_panel.py \
  --raw-dir data/targeted_additions/fda_shortage_archive/raw \
  --captures data/targeted_additions/fda_shortage_archive/captures.json \
  --output data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv \
  --metadata data/targeted_additions/fda_shortage_archive/data/metadata.json
```

`captures.json` is the CDX response reduced to timestamp, URL, digest, and
capture metadata. `raw/SHA256SUMS` pins downloaded bytes. The source endpoint
is `https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm`.

Archived FDA status is national and manufacturer-reported. It should be used
as a middle-layer supply signal or external validation target, not as a local
shortage observation. Current FDA data and missing archive observations remain
distinct from this historical panel.

## Evaluation

The compact rolling evaluation avoids dense supplier/NDC identity columns:

```bash
PYTHONPATH=model python model/scripts/evaluate_fda_archive.py \
  --panel data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv \
  --output model/artifacts/evaluation/fda_archive_shortage_metrics.json
```

The Arkansas-exposure bridge is evaluated separately:

```bash
PYTHONPATH=model python model/scripts/evaluate_arkansas_exposure.py \
  --panel data/targeted_additions/fda_shortage_archive/data/arkansas_medicaid_shortage_exposure.csv \
  --output model/artifacts/evaluation/arkansas_exposure_shortage_metrics.json
```

For the non-persistence onset task:

```bash
PYTHONPATH=model python model/scripts/evaluate_arkansas_exposure.py \
  --onset-only \
  --panel data/targeted_additions/fda_shortage_archive/data/arkansas_medicaid_shortage_exposure.csv \
  --output model/artifacts/evaluation/arkansas_exposure_onset_metrics.json
```

This comparison is required because active shortage states are persistent. A
high score that does not beat previous-quarter persistence is not treated as
new predictive skill.

The exposure panel is built from the real Arkansas Medicaid SDUD records under
`data/S_D/data/combined`:

```bash
PYTHONPATH=model python data/targeted_additions/fda_shortage_archive/scripts/build_arkansas_medicaid_exposure.py \
  --combined-dir data/S_D/data/combined \
  --archive-panel data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv \
  --output data/targeted_additions/fda_shortage_archive/data/arkansas_medicaid_shortage_exposure.csv \
  --metadata data/targeted_additions/fda_shortage_archive/data/arkansas_medicaid_shortage_exposure.json
```

The resulting target is Arkansas Medicaid utilization exposure paired with a
national FDA shortage state. Unmatched rows retain `fda_shortage_active=0` and
`fda_archive_row_observed=0`; this means no FDA event was observed, not that
the product was available in Arkansas.
