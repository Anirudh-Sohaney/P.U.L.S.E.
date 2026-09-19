# Per-drug untouched-test benchmark

This report is a product-level companion to `current_benchmark_results.md`.
It uses the same chronological 60%/20%/20% split, unchanged XGBoost settings,
and training-only top-five signal selection. Each product is scored separately
on the final period beginning 2025-05-26.

The strongest documented acute-care scenarios show the following WAPE results:

| Product | Sales-only WAPE | Signal WAPE | WAPE reduction |
|---|---:|---:|---:|
| Azithromycin 250 mg tablet | 54.46% | 19.62% | 63.97% |
| Doxycycline 100 mg capsule | 70.70% | 31.11% | 55.99% |
| Cephalexin 500 mg capsule | 54.41% | 25.99% | 52.23% |
| Amoxicillin-clavulanate 875/125 mg tablet | 46.46% | 28.57% | 38.51% |

These products are exposed to the documented seasonal respiratory, shortage,
and dated supply/outbreak-response scenarios described in `synthetic_guide.md`.
The figures are specific to this deterministic synthetic benchmark; they are
not estimates of real Arkansas dispensing performance.

## Reproduction

```bash
python3 data/synthetic_pharmacy_data/per_drug_benchmark.py
```

The command writes the complete 30-product table to
`per_drug_benchmark_metrics.csv`.
