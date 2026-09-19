# Publishable Evaluation Suite

This suite is the canonical whole-system testing setup for public-data
experiments. It preserves the valid grain of each target instead of creating a
false county x supplier x drug x week inventory label.

Build it from the repository root:

```bash
PYTHONPATH=model .venv/bin/python model/scripts/build_publishable_test_dataset.py --root .
```

The generated `manifest.json` records source URLs, target semantics, hashes,
period splits, missingness, and right-censoring policy. The primary near-term
table merges Arkansas Medicaid NDC9-quarter exposure with national FDA shortage
evidence and strictly prior-quarter CDC FluView and Arkansas news context.
County demand is annual; supplier/NDC shortage continuation is monthly and
national. The suite also includes a separate quarterly DEA ARCOS ZIP3-by-
controlled-substance distribution proxy. Neither proxy is presented as direct
pharmacy inventory.

## Current benchmark result

Version `2026-08-16.v5` contains 51,208 state NDC9-quarter transitions,
245,371 county-drug-year transitions, and 76,910 supplier-NDC-month
transitions plus the ARCOS ZIP3 distribution transitions. The state demand
ridge improves test WAPE over persistence from
`0.8599` to `0.8465` (1.55% relative improvement), which does not meet the
10% promotion gate. County demand persistence is stronger than the ridge
(`0.1187` versus `0.1795` WAPE). The shortage task has severe class imbalance:
persistence reaches `0.8098` balanced accuracy and `0.4279` AUPRC, while the
logistic model reaches `0.5729` balanced accuracy and `0.1659` AUPRC. Supplier
continuation is nearly all positive and is retained as a censor-aware
diagnostic, not as evidence of shortage-onset skill.

These are benchmark-head results, not an end-to-end claim for the 300M-class
multimodal model. The full repository regression suite must pass before a
training artifact can be compared against this protocol.
