# PADI-web Expert Event Tables

This directory stores the three public PADI-web article-event tables used for
external event-representation evaluation. The labels were manually produced
with help from two epidemiologists. They cover animal-health surveillance and
are not Arkansas human-pharmacy ground truth.

The normalized artifact is built with:

```text
PYTHONPATH=model python -m arkansas_pharma_signal.cli --root . build-expert-event-gold
```

The builder preserves article-event rows, labels, source dataset, and unit
type. It does not join these labels to pharmacy targets or production inputs.
