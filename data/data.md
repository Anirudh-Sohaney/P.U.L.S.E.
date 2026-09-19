# Data Layout and Signal Contract

The repository now contains more than the original date-to-article JSON
format. Data is organized into source families under `data/`, with normalized
tables in `data/targeted_additions/` and the assembled feature, event, entity,
and relationship tables in `data/final_data/`.

## Normalized tables

Each normalized table should preserve `source_id`, `source_url`,
`retrieved_at_utc`, and `extraction_notes`. Dates are UTC unless a source
explicitly documents another timezone. Source-specific README files describe
the schema, coverage, suppression rules, and refresh method.

Important signal families include:

- Arkansas and national demand proxies from CMS Part D, Medicaid SDUD, and
  HHS provider-NDC spending;
- FDA NDC identity, shortage, enforcement, recall, and establishment data;
- CDC FluView, NNDSS, wastewater, hospital respiratory, NSSP, RESP-NET, and
  related disease surveillance;
- Arkansas pharmacy locations, provider geography, DEA ARCOS distribution,
  weather, disasters, economic, trade, sanctions, and supply-chain data;
- timestamped Arkansas news and broader event-extraction corpora.

## Modeling rules

Features must be point-in-time safe: a forecast origin may use only records
published or observed by that origin. Derived signals must retain their source,
geography, cadence, target meaning, and missingness. Public datasets are
external proxies, not direct pharmacy on-hand, fill, backorder, or allocation
observations.

The model keeps Arkansas as its highest-resolution priority while allowing
national, neighboring-state, and global signals to provide context. New
signals are evaluated against chronological baselines before being promoted;
poorly labeled, stale, sparse, or leakage-prone inputs remain documented as
research-only rather than being presented as operational evidence.
