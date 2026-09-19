# Recent Arkansas CMS SDUD Refreshes

This directory records the official CMS annual State Drug Utilization Data
files used by the opt-in `--live-sdud-url` refresh path. The source manifest
records the exact URLs, coverage, transformation, and limitations.

The files are quarterly Medicaid utilization targets, not real-time pharmacy
inventory observations and not operational feature inputs. The 2024 source is
only available through Q2. Missing quarters are intentionally left missing;
the model's consecutive-quarter target contract excludes transitions across
those gaps.

The CLI accepts repeated flags, for example:

```bash
PYTHONPATH=model .venv/bin/python -m arkansas_pharma_signal.cli \
  --root . forecast-quarterly \
  --live-sdud-url https://download.medicaid.gov/data/sdud2023.csv \
  --live-sdud-url https://download.medicaid.gov/data/sdud2024.csv \
  --live-sdud-url https://download.medicaid.gov/data/sdud2025_updatedjuly2026.csv
```

The historical cache is not modified by this command. Forecast metadata
records the panel period, period age, and every refresh URL.
