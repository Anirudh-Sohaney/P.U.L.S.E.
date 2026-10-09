"""Produce next-14-day per-drug forecasts from local dated input files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from train import BASE_COLUMNS, MODEL_DIR, OUT_DIR, make_sales_frame, safe_name, signal_matrix


def predict(sales_path: Path, signal_path: Path, output_path: Path,
            news_forecast_path: Path | None = None,
            recipe: str = "legacy-published") -> pd.DataFrame:
    if recipe == "legacy-published":
        manifest_path = OUT_DIR / "legacy_recipe_demand_model.json"
        model_dir = MODEL_DIR / "legacy_recipe"
    elif recipe == "validation-selected":
        manifest_path = OUT_DIR / "selected_demand_model.json"
        model_dir = MODEL_DIR
    else:
        raise ValueError(f"Unknown model recipe: {recipe}")
    if not manifest_path.exists():
        raise FileNotFoundError("Train the rebuilt demand model before forecasting")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = make_sales_frame(sales_path)
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    if news_forecast_path is None:
        news_forecast_path = MODEL_DIR / "initial_20_signals" / "next_month_20_signals.csv"
    news_forecast = pd.read_csv(news_forecast_path, parse_dates=["date"]) if news_forecast_path.exists() else None
    signals, _ = signal_matrix(dates, signal_path, news_forecast)
    frame = frame.join(signals, on="date")
    feature_map = manifest["features_by_drug"]
    rows = []
    for drug, part in frame.groupby("drug_name", sort=True):
        if drug not in feature_map:
            continue
        features = feature_map[drug]
        signal_features = [col for col in features if col not in BASE_COLUMNS]
        additions = {}
        for name in signal_features:
            signal_name, lag_text = name.rsplit("__lag", 1)
            lag = int(lag_text)
            if signal_name not in part:
                raise ValueError(f"Input signal table is missing selected feature {signal_name}")
            additions[name] = part[signal_name].shift(lag)
        work = pd.concat([part, pd.DataFrame(additions, index=part.index)], axis=1)
        latest = work.dropna(subset=features).tail(1)
        if latest.empty:
            continue
        model = XGBRegressor()
        model.load_model(str(model_dir / f"demand_{safe_name(drug)}.json"))
        forecast = max(float(model.predict(latest[features])[0]), 0.0)
        origin = pd.Timestamp(latest.date.iloc[0])
        rows.append({
            "drug_name": drug,
            "forecast_origin": origin.date().isoformat(),
            "target_window_start": (origin + pd.DateOffset(days=1)).date().isoformat(),
            "target_window_end": (origin + pd.DateOffset(days=14)).date().isoformat(),
            "predicted_units_next_14d": forecast,
            "model_configuration": manifest["selected_configuration"],
            "evidence_status": "synthetic_benchmark_model; not validated on observed pharmacy sales",
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No forecasts were produced; check input drug names and history length")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, required=True, help="Daily sales CSV with date, drug_name, units_sold and price/stockout columns")
    parser.add_argument("--signals", type=Path, required=True, help="Dated signal CSV with the catalog schema used for training")
    parser.add_argument("--news-forecast", type=Path, help="20-output model CSV; defaults to its latest local prediction when present")
    parser.add_argument("--recipe", choices=["legacy-published", "validation-selected"],
                        default="legacy-published",
                        help="Forecast with the fixed historical recipe or the winner of the 90-configuration validation grid")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "demand_14d_predictions.csv")
    args = parser.parse_args()
    result = predict(args.sales, args.signals, args.out, args.news_forecast, args.recipe)
    print(json.dumps({"output": str(args.out), "forecasts": len(result)}, indent=2))


if __name__ == "__main__":
    main()
