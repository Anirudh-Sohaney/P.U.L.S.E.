# Benchmark provenance

This directory contains a deterministic synthetic pharmacy-demand benchmark.
It is not an observed Arkansas pharmacy data release.

## Current artifacts

| Artifact | SHA-256 | Verified property |
|---|---|---|
| `arkansas_clinic_daily_pharmacy_sales.csv` | `e9f375dd0ee9c0a1bfdf562389381dc78d13539511079aebd01c5a97b33090f7` | 32,880 rows; 30 drugs; 2023-01-01 through 2025-12-31 |
| `generate_synthetic_pharmacy.py` | `4daa59ce021cf8cb02a5eee3a77c20b234cde4e012763a4a324d88408886ddf4` | deterministic generator; seed `20250915` |
| `../../test/test_Signals/signals_2023_2025.csv.gz` | `64d6c0f5f9a2efe36a76e8a2c89795f5a13c8270246eff4989f387ad0ac8cbed` | frozen model-signal input |
| `current_benchmark_metrics.json` | `d1b6edbf0481028ed3bdd5eb575740bc2648c4969db4858aacb10d82137530b0` | untouched-test evaluation output for this CSV; do not substitute metrics from a prior CSV |
| `current_benchmark_results.md` | `ec5ffe653ed12687e22b9e78691ce87718db424b750d3c0bb20a438735443187` | human-readable companion to the current metrics |
| `per_drug_benchmark.py` | `6829faa6dcba973d2c02d992ead5c2d5ce3f47108a4d4cfbdd87fe69e4108d7e` | unchanged-runner-compatible per-product evaluator |
| `per_drug_benchmark_metrics.csv` | `968f01af0753845e098f2589536f13b73caab09ff4d3764164ec2afe96a3d2cf` | final-period WAPE metrics for all 30 products |
| `per_drug_benchmark_results.md` | `547efbd33d19cc0ed8ac3509d3ca693111035077b31fe9b43072b595de6508fe` | product-level result summary |
| `inspect_signal_selection.py` | `9825c3eabbc50548199129acd7ea7e576f09b78ea670f1f1223211050b01fb7b` | training-period signal-selection audit |
| `selected_signal_diagnostics.csv` | `da4a72ec1391796efce9a09d6a851ce0fc39122bdef2237a2139e646912abf14` | selected-signal ranks and training correlations |
| `../../website/data/demand_inventory_snapshot.csv` | `42c3c16076d1bc31d793b246240511a4072508b945a7e67317de6f9fd0874b91` | deterministic 30-product synthetic on-hand inventory scenario used by the locked website demo |
| `../../website/ml/demand_forecast.py` | `8d08e56e8efc433dd2d043af7ca0b39c98e9b0f521e75462200fc533c45e52e7` | chronological forecast plus transparent inventory/replenishment policy |

Regenerate the sales file with:

```bash
python3 data/synthetic_pharmacy_data/generate_synthetic_pharmacy.py
sha256sum data/synthetic_pharmacy_data/arkansas_clinic_daily_pharmacy_sales.csv
```

The event tags and generator code are part of the benchmark's audit trail.
They describe modeled scenarios and must not be presented as evidence of
actual patient demand, actual pharmacy transactions, or actual Arkansas
outbreak activity.

The hashes above must be refreshed whenever an artifact is intentionally
changed. A changed CSV requires a new benchmark run; prior metrics must not be
carried over to the new file.

The inventory snapshot is derived from the unchanged sales CSV, not observed
inventory. Its policy and limitations are documented in `synthetic_guide.md`.
