"""Materialize model signal values usable with the 2023-2025 clinic sales test."""
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL_NEWS = ROOT / "model/artifacts/news/news_only_catalog_features.csv.gz"
MODEL_FEATURES = ROOT / "data/final_data/features/external_state_features.csv.gz"
FINAL_OUTPUTS = ROOT / "model/final_predictions.json"
OUTPUT = OUT / "signals_2023_2025.csv.gz"

OUT_COLS = ["signal_date", "period_end", "cadence", "signal_id", "value", "unit",
            "geography_level", "geography_id", "geography_name", "source_id",
            "source_name", "source_url", "source_timestamp", "forecast_horizon",
            "data_quality", "missingness", "source_release_frequency", "signal_origin"]
OUT_COLS += ["entity_type", "entity_key", "state_definition", "model_output_type"]


def main():
    frames = []
    news = pd.read_csv(MODEL_NEWS, parse_dates=["date"])
    news = news[news.date.dt.year.between(2023, 2025)].copy()
    signal_cols = [c for c in news.columns if c != "date"]
    n = news.melt(id_vars="date", value_vars=signal_cols, var_name="signal_id", value_name="value")
    n["signal_date"] = n["date"]
    n["period_end"] = n.signal_date + pd.offsets.MonthEnd(0)
    n["cadence"] = "monthly"
    n["unit"] = "model_signal_score"
    n["geography_level"] = "state"
    n["geography_id"] = "AR"
    n["geography_name"] = "Arkansas"
    n["source_id"] = "news_only_slm"
    n["source_name"] = "Validated news-only SLM signal artifact"
    n["source_url"] = "https://www.gdeltproject.org/"
    n["source_timestamp"] = pd.NaT
    n["forecast_horizon"] = "current_month"
    n["data_quality"] = "derived"
    n["missingness"] = "not_imputed"
    n["source_release_frequency"] = "monthly"
    n["signal_origin"] = "model_news_output"
    n["entity_type"] = ""
    n["entity_key"] = ""
    n["state_definition"] = ""
    n["model_output_type"] = "upstream_signal"
    frames.append(n[OUT_COLS])

    # Add the model's derived demand outputs: one state per CMS drug and per
    # qualified Arkansas ATC class. These are forecast outputs, not extra
    # upstream news signals, so their entity keys are retained for joins.
    payload = json.loads(FINAL_OUTPUTS.read_text(encoding="utf-8"))
    derived = []
    cms = payload.get("cms_part_d_signals_1300", {})
    for drug, item in cms.items():
        derived.append({"signal_date": "2024-01-01", "period_end": "2024-12-31",
                        "cadence": "annual", "signal_id": f"cms_part_d_demand_state::{drug}",
                        "value": item.get("predicted_demand_state"), "unit": "five_state_demand",
                        "geography_level": "state", "geography_id": "AR",
                        "geography_name": "Arkansas", "source_id": "final_predictions",
                        "source_name": "PULSE production CMS Part D demand output",
                        "source_url": "", "source_timestamp": pd.NaT,
                        "forecast_horizon": "2025", "data_quality": "model_output",
                        "missingness": "not_imputed", "source_release_frequency": "annual",
                        "signal_origin": "derived_demand_output", "entity_type": "drug",
                        "entity_key": drug, "state_definition": item.get("state_definition", ""),
                        "model_output_type": "predicted_demand_state"})
    for atc, item in payload.get("arkansas_atc_signals_18", {}).items():
        derived.append({"signal_date": "2024-12-01", "period_end": "2024-12-31",
                        "cadence": "monthly", "signal_id": f"arkansas_atc_demand_state::{atc}",
                        "value": item.get("predicted_demand_state"), "unit": "three_state_demand",
                        "geography_level": "state", "geography_id": "AR",
                        "geography_name": "Arkansas", "source_id": "final_predictions",
                        "source_name": "PULSE production Arkansas ATC demand output",
                        "source_url": "", "source_timestamp": pd.NaT,
                        "forecast_horizon": "2025-01", "data_quality": "model_output",
                        "missingness": "not_imputed", "source_release_frequency": "monthly",
                        "signal_origin": "derived_demand_output", "entity_type": "therapeutic_class",
                        "entity_key": atc, "state_definition": item.get("state_definition", ""),
                        "model_output_type": "predicted_demand_state"})
    frames.append(pd.DataFrame(derived, columns=OUT_COLS))

    use = ["value", "unit", "variable_id", "geography_level", "geography_id",
           "geography_name", "observation_time", "period_end", "forecast_horizon",
           "source_id", "source_name", "source_url", "source_timestamp",
           "release_frequency", "data_quality", "missingness"]
    chunks = []
    for x in pd.read_csv(MODEL_FEATURES, usecols=use, chunksize=200_000, low_memory=False):
        date = pd.to_datetime(x["observation_time"], errors="coerce")
        x = x[date.dt.year.between(2023, 2025) & x.geography_level.isin(["state", "national"])]
        if x.empty: continue
        freq = x.release_frequency.astype(str).str.lower()
        x["cadence"] = ""
        x.loc[freq.str.contains("week|daily_or_weekly"), "cadence"] = "weekly"
        x.loc[freq.str.contains("month"), "cadence"] = "monthly"
        x.loc[freq.str.contains("annual|year"), "cadence"] = "annual"
        x = x[x.cadence.ne("")].copy()
        x["signal_date"] = pd.to_datetime(x.observation_time, errors="coerce")
        x["signal_id"] = x.variable_id
        x["signal_origin"] = "model_external_state_feature"
        x["source_release_frequency"] = x.release_frequency
        x["entity_type"] = ""
        x["entity_key"] = ""
        x["state_definition"] = ""
        x["model_output_type"] = "upstream_signal"
        x = x.rename(columns={"observation_time": "_observation_time"})
        chunks.append(x[OUT_COLS])
    if chunks: frames.append(pd.concat(chunks, ignore_index=True))
    result = pd.concat(frames, ignore_index=True)
    result["signal_date"] = pd.to_datetime(result.signal_date, errors="coerce").dt.strftime("%Y-%m-%d")
    result["period_end"] = pd.to_datetime(result.period_end, errors="coerce").dt.strftime("%Y-%m-%d")
    result = result.sort_values(["signal_date", "cadence", "signal_id", "geography_level", "geography_id"])
    result.to_csv(OUTPUT, index=False, compression="gzip")
    print({"rows": len(result), "path": str(OUTPUT), "cadences": result.cadence.value_counts().to_dict(),
           "origins": result.signal_origin.value_counts().to_dict(),
           "dates": [result.signal_date.min(), result.signal_date.max()]})


if __name__ == "__main__": main()
