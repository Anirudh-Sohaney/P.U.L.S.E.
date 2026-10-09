"""Train the local 20-output signal forecaster and compare demand models.

The 20-signal forecaster is an explicitly labeled temporal replacement: the
historical output table exists, but the original article inference code and
weights do not. Demand-model comparison uses the full 1,312-column dated
signal table and a chronological 60/20/20 split of synthetic sales.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from arkansas_pharma_signal.news_only_adapter import NEWS_ONLY_SIGNAL_IDS  # noqa: E402

SALES_PATH = ROOT / "data/synthetic_pharmacy_data/arkansas_clinic_daily_pharmacy_sales.csv"
SIGNAL_PATH = ROOT / "test/test_Signals/signals_2023_2025.csv.gz"
NEWS_PATH = ROOT / "website/catalog/news_only_catalog_features.csv.gz"
OUT_DIR = Path(__file__).resolve().parent
MODEL_DIR = OUT_DIR / "models"

BASE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28", "lag_56",
    "mean_3", "std_3", "mean_7", "std_7", "mean_14", "std_14",
    "mean_28", "std_28", "mean_56", "std_56", "dow", "weekofyear", "month",
    "sin_year", "cos_year", "price_lag1", "stockout_lag1",
]


@dataclass(frozen=True)
class Variant:
    name: str
    signal_group: str
    count: int
    lags: tuple[int, ...]
    selection_lag: int = 1


@dataclass(frozen=True)
class Profile:
    name: str
    depth: int
    estimators: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    colsample: float
    objective: str = "reg:squarederror"
    tweedie_variance_power: float | None = None
    max_delta_step: float = 0.0
    seed: int = 20261008


VARIANTS = [
    Variant("sales_only", "none", 0, ()),
    Variant("all_1312_lag1", "all", 1312, (1,)),
    Variant("top5_all_lag1", "all", 5, (1,)),
    Variant("top5_all_lags_1_7_14", "all", 5, (1, 7, 14)),
    Variant("top5_all_lags_1_7_14_28", "all", 5, (1, 7, 14, 28)),
    Variant("top10_all_lags_1_7_14", "all", 10, (1, 7, 14)),
    Variant("top5_news20_lags_1_7_14", "model_news_output", 5, (1, 7, 14)),
    Variant("top5_derived_lags_1_7_14", "derived_demand_output", 5, (1, 7, 14)),
    Variant("top5_external_lags_1_7_14", "model_external_state_feature", 5, (1, 7, 14)),
    Variant("top5_all_lags_1_7_14_legacy_rank", "all", 5, (1, 7, 14), selection_lag=0),
]

PROFILES = [
    Profile("shallow_regularized", 2, 120, .04, 8, 10, .70),
    Profile("shallow_more_trees", 2, 300, .03, 5, 5, .80),
    Profile("medium_balanced", 3, 220, .04, 5, 5, .80),
    Profile("medium_less_regularized", 3, 350, .05, 2, 1, .90),
    Profile("deep_regularized", 5, 220, .03, 8, 10, .80),
    Profile("shallow_poisson", 2, 120, .04, 8, 10, .70, "count:poisson", max_delta_step=.7),
    Profile("shallow_tweedie", 2, 120, .04, 8, 10, .70, "reg:tweedie", 1.3),
    Profile("shallow_pseudo_huber", 2, 120, .04, 8, 10, .70, "reg:pseudohubererror"),
    Profile("legacy_published_recipe", 2, 220, .04, 8, 10, .50, seed=20250915),
]


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")[:110]


def make_sales_frame(sales_path: Path = SALES_PATH) -> pd.DataFrame:
    frame = pd.read_csv(sales_path, parse_dates=["date"])
    frame = frame.sort_values(["drug_name", "date"]).reset_index(drop=True)
    group = frame.groupby("drug_name", sort=False)["units_sold"]
    frame["target_14d"] = group.transform(
        lambda series: series.shift(-1).rolling(14, min_periods=14).sum().shift(-13)
    )
    for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
        frame[f"lag_{lag}"] = group.shift(lag)
    shifted = group.shift(1)
    for window in [3, 7, 14, 28, 56]:
        shifted_group = shifted.groupby(frame["drug_name"])
        frame[f"mean_{window}"] = shifted_group.transform(
            lambda series: series.rolling(window, min_periods=window).mean()
        )
        frame[f"std_{window}"] = shifted_group.transform(
            lambda series: series.rolling(window, min_periods=window).std()
        )
    frame["dow"] = frame.date.dt.dayofweek
    frame["weekofyear"] = frame.date.dt.isocalendar().week.astype(int)
    frame["month"] = frame.date.dt.month
    doy = frame.date.dt.dayofyear
    frame["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    frame["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    frame["price_lag1"] = frame.groupby("drug_name").unit_price_usd.shift(1)
    frame["stockout_lag1"] = frame.groupby("drug_name").stockout_flag.shift(1)
    return frame


def signal_matrix(dates: pd.DatetimeIndex, signal_path: Path = SIGNAL_PATH,
                  news_forecast: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(signal_path, parse_dates=["signal_date", "period_end"], low_memory=False)
    if news_forecast is not None and len(news_forecast):
        forecast = news_forecast.copy()
        if "date" not in forecast or not set(NEWS_ONLY_SIGNAL_IDS).issubset(forecast.columns):
            raise ValueError("20-signal forecast must contain date and all 20 archived signal IDs")
        forecast_date = pd.to_datetime(forecast.date.iloc[-1])
        forecast_end = forecast_date.to_period("M").end_time.normalize()
        rows = []
        for signal_id in NEWS_ONLY_SIGNAL_IDS:
            templates = source.loc[
                source.signal_origin.eq("model_news_output") & source.signal_id.eq(signal_id)
            ].sort_values("period_end")
            if templates.empty:
                raise ValueError(f"The 1,312-signal catalog is missing initial output {signal_id}")
            row = templates.iloc[-1].copy()
            row["signal_date"] = forecast_end
            row["period_end"] = forecast_end
            row["value"] = pd.to_numeric(forecast[signal_id].iloc[-1], errors="coerce")
            row["source_name"] = "rebuilt_temporal_20_signal_model"
            row["source_timestamp"] = forecast_date
            row["data_quality"] = "model_forecast_from_archived_outputs"
            rows.append(row)
        source = pd.concat([source, pd.DataFrame(rows)], ignore_index=True)
    source["value"] = pd.to_numeric(source.value, errors="coerce")
    source = source.dropna(subset=["value", "period_end"]).copy()
    key_cols = ["signal_origin", "signal_id", "cadence", "geography_level", "geography_id", "entity_key"]
    for col in key_cols:
        source[col] = source[col].fillna("").astype(str)
    source["feature_key"] = source[key_cols].agg("|".join, axis=1).map(safe_name)
    source = source.sort_values(["feature_key", "period_end", "signal_date"])
    source = source.drop_duplicates(["feature_key", "period_end"], keep="last")
    metadata = source[["feature_key", "signal_origin", "signal_id"]].drop_duplicates("feature_key")
    columns: dict[str, pd.Series] = {}
    for key, part in source.groupby("feature_key", sort=True):
        series = part.set_index("period_end").value.sort_index()
        series = series[~series.index.duplicated(keep="last")]
        columns[key] = series.reindex(dates, method="ffill")
    matrix = pd.DataFrame(columns, index=dates).fillna(0.0)
    if matrix.shape[1] != source.signal_id.nunique():
        raise ValueError(
            f"Expected one dated feature per catalog signal ID; got {matrix.shape[1]} features "
            f"for {source.signal_id.nunique()} IDs. Resolve key collisions before training."
        )
    return matrix, metadata


def date_splits(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    return unique[int(len(unique) * .60)], unique[int(len(unique) * .80)]


def metric_pack(actual, predicted) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    p = np.maximum(p, 0)
    den = np.maximum(np.abs(y) + np.abs(p), 1e-9)
    rel = np.abs(y - p) / np.maximum(y, 1.0)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "wape": float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9)),
        "smape": float(np.mean(2 * np.abs(y - p) / den)),
        "within_5pct": float(np.mean(rel <= .05)),
        "within_10pct": float(np.mean(rel <= .10)),
        "within_20pct": float(np.mean(rel <= .20)),
        "n": int(len(y)),
    }


def select_features(train: pd.DataFrame, signal_columns: list[str], metadata: pd.DataFrame,
                    group: str, count: int, selection_lag: int = 1) -> list[str]:
    if group == "none":
        return []
    if group == "all":
        candidates = signal_columns
    else:
        candidates = metadata.loc[metadata.signal_origin.eq(group), "feature_key"].tolist()
    scores = []
    y = train.target_14d.to_numpy(dtype=float)
    for name in candidates:
        values = train[name].shift(selection_lag).to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(y)
        if valid.sum() < 30 or np.std(values[valid]) < 1e-12:
            continue
        score = np.corrcoef(values[valid], y[valid])[0, 1]
        if np.isfinite(score):
            scores.append((name, abs(float(score))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in scores[:count]]


def add_signal_lags(frame: pd.DataFrame, selected: list[str], lags: tuple[int, ...]) -> tuple[pd.DataFrame, list[str]]:
    additions = {}
    for column in selected:
        for lag in lags:
            additions[f"{column}__lag{lag}"] = frame[column].shift(lag)
    if not additions:
        return frame, []
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1), list(additions)


def make_model(profile: Profile) -> XGBRegressor:
    params = dict(
        n_estimators=profile.estimators,
        max_depth=profile.depth,
        learning_rate=profile.learning_rate,
        min_child_weight=profile.min_child_weight,
        subsample=.85,
        colsample_bytree=profile.colsample,
        reg_lambda=profile.reg_lambda,
        objective=profile.objective,
        tree_method="hist",
        n_jobs=1,
        random_state=profile.seed,
        verbosity=0,
        max_delta_step=profile.max_delta_step,
    )
    if profile.tweedie_variance_power is not None:
        params["tweedie_variance_power"] = profile.tweedie_variance_power
    return XGBRegressor(**params)


def evaluate_split(frame: pd.DataFrame, signals: pd.DataFrame, metadata: pd.DataFrame,
                   variants: list[Variant], profiles: list[Profile], val_start: pd.Timestamp,
                   test_start: pd.Timestamp, split: str, save_selected: tuple[str, str] | None = None,
                   model_dir: Path = MODEL_DIR):
    records, predictions, chosen_by_drug = [], {}, {}
    for drug, raw_part in frame.groupby("drug_name", sort=True):
        print(f"[{split}] {drug}: fitting {len(variants) * len(profiles)} configurations", flush=True)
        part = raw_part.join(signals, on="date")
        part = part.dropna(subset=BASE_COLUMNS + ["target_14d"]).copy()
        boundary = val_start if split == "validation" else test_start
        # Each target covers t+1 through t+14. Drop the final 14 training
        # origins so no training label contains a validation/test outcome.
        train_mask = part.date < boundary - pd.DateOffset(days=14)
        if split == "validation":
            eval_mask = (part.date >= val_start) & (part.date < test_start - pd.DateOffset(days=14))
        else:
            eval_mask = part.date >= test_start
        train = part.loc[train_mask].copy()
        eval_rows = part.loc[eval_mask].copy()
        if len(train) < 100 or not len(eval_rows):
            continue
        chosen_by_drug[drug] = {}
        for variant in variants:
            selected = select_features(train, list(signals.columns), metadata,
                                       variant.signal_group, variant.count, variant.selection_lag)
            work, signal_cols = add_signal_lags(part, selected, variant.lags)
            feat = BASE_COLUMNS + signal_cols
            fit = work.loc[train_mask].dropna(subset=feat)
            score_rows = work.loc[eval_mask].dropna(subset=feat)
            if len(fit) < 100 or not len(score_rows):
                continue
            chosen_by_drug[drug][variant.name] = selected
            for profile in profiles:
                model = make_model(profile)
                model.fit(fit[feat], fit.target_14d, verbose=False)
                pred = model.predict(score_rows[feat])
                key = f"{variant.name}__{profile.name}"
                predictions.setdefault(key, {"y": [], "p": [], "drug": [], "date": []})
                predictions[key]["y"].extend(score_rows.target_14d.to_numpy())
                predictions[key]["p"].extend(pred)
                predictions[key]["drug"].extend([drug] * len(score_rows))
                predictions[key]["date"].extend(score_rows.date.to_numpy())
                if save_selected == (variant.name, profile.name) and split == "test":
                    model_dir.mkdir(parents=True, exist_ok=True)
                    full_train = work.loc[train_mask].dropna(subset=feat)
                    final_model = make_model(profile).fit(full_train[feat], full_train.target_14d, verbose=False)
                    final_model.save_model(str(model_dir / f"demand_{safe_name(drug)}.json"))
                    chosen_by_drug[drug]["final_features"] = feat
                    chosen_by_drug[drug]["final_signals"] = selected
        if split == "validation":
            seasonal_pred = eval_rows["mean_14"].to_numpy() * 14
        else:
            seasonal_pred = eval_rows["mean_14"].to_numpy() * 14
        predictions.setdefault("seasonal_naive", {"y": [], "p": [], "drug": [], "date": []})
        predictions["seasonal_naive"]["y"].extend(eval_rows.target_14d.to_numpy())
        predictions["seasonal_naive"]["p"].extend(seasonal_pred)
        predictions["seasonal_naive"]["drug"].extend([drug] * len(eval_rows))
        predictions["seasonal_naive"]["date"].extend(eval_rows.date.to_numpy())

    for key, values in predictions.items():
        per_drug = []
        packed = pd.DataFrame({"drug": values["drug"], "y": values["y"], "p": values["p"]})
        for drug, subset in packed.groupby("drug"):
            per_drug.append({"drug_name": drug, **metric_pack(subset.y, subset.p)})
        pooled = metric_pack(values["y"], values["p"])
        records.append({
            "split": split,
            "variant": key,
            "drugs": len(per_drug),
            "mean_drug_wape": float(np.mean([row["wape"] for row in per_drug])),
            "mean_drug_mae": float(np.mean([row["mae"] for row in per_drug])),
            "pooled_wape": pooled["wape"],
            "pooled_mae": pooled["mae"],
            "pooled_rmse": pooled["rmse"],
            "pooled_smape": pooled["smape"],
            "within_20pct": pooled["within_20pct"],
            "origins": pooled["n"],
        })
    return records, predictions, chosen_by_drug


def train_twenty_signal_forecaster() -> dict:
    """Fit 20 next-month signal regressors from the dated archived output."""
    source = pd.read_csv(NEWS_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if list(source.columns) != ["date", *NEWS_ONLY_SIGNAL_IDS]:
        raise ValueError("The archived 20-signal table does not match the expected output schema")
    lagged = {
        f"{signal}__lag{lag}": source[signal].shift(lag)
        for signal in NEWS_ONLY_SIGNAL_IDS for lag in [1, 2, 3, 6, 12]
    }
    news = pd.concat([source, pd.DataFrame(lagged)], axis=1)
    news = pd.concat([news, pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * news.date.dt.month / 12),
        "month_cos": np.cos(2 * np.pi * news.date.dt.month / 12),
    })], axis=1)
    target_dates = news.date.shift(-1)
    feature_cols = [c for c in news if "__lag" in c] + ["month_sin", "month_cos"]
    train_end = int(len(news) * .80)
    fit = news.iloc[:train_end].dropna(subset=feature_cols)
    test = news.iloc[train_end:].dropna(subset=feature_cols)
    if len(fit) < 20 or len(test) < 5:
        raise ValueError("Not enough monthly history for a chronological 20-signal evaluation")
    scores, models = {}, {}
    for signal in NEWS_ONLY_SIGNAL_IDS:
        if len(models) % 5 == 0:
            print(f"[initial-20] trained {len(models)}/{len(NEWS_ONLY_SIGNAL_IDS)} outputs", flush=True)
        model = XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=.04,
            min_child_weight=3, subsample=.9, colsample_bytree=.8,
            reg_lambda=5, objective="reg:squarederror", tree_method="hist",
            n_jobs=1, random_state=20261008, verbosity=0,
        )
        model.fit(fit[feature_cols], fit[signal], verbose=False)
        prediction = model.predict(test[feature_cols])
        persistence = test[f"{signal}__lag1"].to_numpy()
        scores[signal] = {
            "test_rows": int(len(test)),
            "xgb_mae": float(np.mean(np.abs(test[signal].to_numpy() - prediction))),
            "persistence_mae": float(np.mean(np.abs(test[signal].to_numpy() - persistence))),
        }
        final = XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=.04,
            min_child_weight=3, subsample=.9, colsample_bytree=.8,
            reg_lambda=5, objective="reg:squarederror", tree_method="hist",
            n_jobs=1, random_state=20261008, verbosity=0,
        )
        final.fit(news.dropna(subset=feature_cols)[feature_cols],
                  news.dropna(subset=feature_cols)[signal], verbose=False)
        models[signal] = final
    model_dir = MODEL_DIR / "initial_20_signals"
    model_dir.mkdir(parents=True, exist_ok=True)
    for signal, model in models.items():
        model.save_model(str(model_dir / f"{signal}.json"))
    prediction_date = (news.date.max().to_period("M") + 1).to_timestamp()
    feature_row = {}
    for signal in NEWS_ONLY_SIGNAL_IDS:
        for lag in [1, 2, 3, 6, 12]:
            feature_row[f"{signal}__lag{lag}"] = float(news[signal].iloc[-lag])
    feature_row["month_sin"] = float(np.sin(2 * np.pi * prediction_date.month / 12))
    feature_row["month_cos"] = float(np.cos(2 * np.pi * prediction_date.month / 12))
    prediction_features = pd.DataFrame([feature_row], columns=feature_cols)
    out = {signal: float(model.predict(prediction_features)[0]) for signal, model in models.items()}
    pd.DataFrame([{"date": prediction_date, **out}]).to_csv(model_dir / "next_month_20_signals.csv", index=False)
    result = {
        "classification": "temporal replacement trained on archived outputs; not recovered article-to-signal model",
        "source_rows": int(len(news)),
        "source_date_range": [str(news.date.min().date()), str(news.date.max().date())],
        "holdout_rows": int(len(test)),
        "holdout_start": str(test.date.min().date()),
        "signal_count": len(NEWS_ONLY_SIGNAL_IDS),
        "signal_ids": list(NEWS_ONLY_SIGNAL_IDS),
        "mean_xgb_mae": float(np.mean([row["xgb_mae"] for row in scores.values()])),
        "mean_persistence_mae": float(np.mean([row["persistence_mae"] for row in scores.values()])),
        "per_signal": scores,
        "next_month_output": str(model_dir / "next_month_20_signals.csv"),
    }
    (OUT_DIR / "initial_20_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_demand_comparison() -> dict:
    frame = make_sales_frame()
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    signals, metadata = signal_matrix(dates)
    val_start, test_start = date_splits(dates)
    if len(VARIANTS) * len(PROFILES) < 40:
        raise AssertionError("Comparison grid must contain at least 40 configurations")
    config_map = {
        f"{variant.name}__{profile.name}": {"variant": asdict(variant), "profile": asdict(profile)}
        for variant in VARIANTS for profile in PROFILES
    }
    validation, _, _ = evaluate_split(frame, signals, metadata, VARIANTS, PROFILES,
                                      val_start, test_start, "validation")
    val_models = [row for row in validation if row["variant"] != "seasonal_naive"]
    selected = min(val_models, key=lambda row: (row["mean_drug_wape"], row["pooled_wape"]))
    selected_variant, selected_profile = selected["variant"].rsplit("__", 1)
    chosen_variant_obj = next(item for item in VARIANTS if item.name == selected_variant)
    chosen_profile_obj = next(item for item in PROFILES if item.name == selected_profile)
    sales_only = next(item for item in VARIANTS if item.name == "sales_only")
    test, test_predictions, selections = evaluate_split(
        frame, signals, metadata, [chosen_variant_obj, sales_only], [chosen_profile_obj],
        val_start, test_start, "test", (selected_variant, selected_profile),
    )
    val_by = {row["variant"]: row for row in validation}
    test_by = {row["variant"]: row for row in test}
    legacy_variant = next(item for item in VARIANTS if item.name == "top5_all_lags_1_7_14_legacy_rank")
    legacy_profile = next(item for item in PROFILES if item.name == "legacy_published_recipe")
    legacy_test, legacy_predictions, legacy_selections = evaluate_split(
        frame, signals, metadata, [legacy_variant, sales_only], [legacy_profile],
        val_start, test_start, "test",
        (legacy_variant.name, legacy_profile.name), MODEL_DIR / "legacy_recipe",
    )
    legacy_by = {row["variant"]: row for row in legacy_test}
    test_by.update(legacy_by)
    comparison = []
    for name, params in config_map.items():
        comparison.append({**params, "configuration": name,
                           "validation": val_by.get(name, {}), "final_test": test_by.get(name, {})})
    with (OUT_DIR / "comparison_results.json").open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, indent=2)
    pd.DataFrame([
        {"configuration": row["configuration"], "variant": row["variant"]["name"],
         "profile": row["profile"]["name"], **{f"validation_{k}": v for k, v in row["validation"].items() if k not in {"variant", "split"}},
         **{f"test_{k}": v for k, v in row["final_test"].items() if k not in {"variant", "split"}}}
        for row in comparison
    ]).to_csv(OUT_DIR / "comparison_results.csv", index=False)
    best_test = test_by[selected["variant"]]
    selected_preds = test_predictions[selected["variant"]]
    baseline_preds = test_predictions[f"sales_only__{selected_profile}"]
    details = []
    pack = pd.DataFrame({"drug_name": selected_preds["drug"],
                         "date": selected_preds["date"],
                         "actual": selected_preds["y"], "predicted": selected_preds["p"]})
    baseline_pack = pd.DataFrame({"drug_name": baseline_preds["drug"],
                                  "date": baseline_preds["date"],
                                  "baseline_actual": baseline_preds["y"],
                                  "baseline_predicted": baseline_preds["p"]})
    paired = pack.merge(baseline_pack, on=["drug_name", "date"], validate="one_to_one")
    for drug, rows in paired.groupby("drug_name", sort=True):
        score = metric_pack(rows.actual, rows.predicted)
        baseline_score = metric_pack(rows.baseline_actual, rows.baseline_predicted)
        details.append({"drug_name": drug, **score,
                        "sales_only_wape": baseline_score["wape"],
                        "wape_improvement_vs_sales_only": 1 - score["wape"] / max(baseline_score["wape"], 1e-12)})
    pd.DataFrame(details).to_csv(OUT_DIR / "selected_model_per_drug_test.csv", index=False)
    legacy_signal = legacy_predictions["top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe"]
    legacy_sales = legacy_predictions["sales_only__legacy_published_recipe"]
    legacy_pairs = pd.DataFrame({
        "drug_name": legacy_signal["drug"], "date": legacy_signal["date"],
        "actual": legacy_signal["y"], "predicted": legacy_signal["p"],
    }).merge(pd.DataFrame({
        "drug_name": legacy_sales["drug"], "date": legacy_sales["date"],
        "sales_actual": legacy_sales["y"], "sales_predicted": legacy_sales["p"],
    }), on=["drug_name", "date"], validate="one_to_one")
    legacy_details = []
    for drug, rows in legacy_pairs.groupby("drug_name", sort=True):
        signal_score = metric_pack(rows.actual, rows.predicted)
        sales_score = metric_pack(rows.sales_actual, rows.sales_predicted)
        legacy_details.append({
            "drug_name": drug, **signal_score,
            "sales_only_wape": sales_score["wape"],
            "wape_improvement_vs_sales_only": 1 - signal_score["wape"] / max(sales_score["wape"], 1e-12),
        })
    pd.DataFrame(legacy_details).to_csv(OUT_DIR / "legacy_recipe_per_drug_test.csv", index=False)
    selected_manifest = {
        "selected_configuration": selected["variant"],
        "validation_mean_drug_wape": selected["mean_drug_wape"],
        "test": best_test,
        "signal_count": int(signals.shape[1]),
        "signal_rows": int(len(pd.read_csv(SIGNAL_PATH, usecols=["signal_id"]))),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
        "validation_start": str(val_start.date()),
        "final_test_start": str(test_start.date()),
        "drug_count": int(frame.drug_name.nunique()),
        "variants": len(VARIANTS),
        "profiles": len(PROFILES),
        "configuration_count": len(config_map),
        "selection_rule": "lowest unweighted mean per-drug validation WAPE; 14-day purge before validation/test boundaries; no final-test-based tuning",
        "target_horizon_days": 14,
        "purge_days": 14,
        "signals_by_drug": {drug: selections[drug].get(selected_variant, []) for drug in selections},
        "features_by_drug": {drug: selections[drug].get("final_features", []) for drug in selections},
        "sales_only_comparison": test_by[f"sales_only__{selected_profile}"],
        "legacy_recipe_test": test_by["top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe"],
        "legacy_recipe_sales_only": test_by["sales_only__legacy_published_recipe"],
        "wape_improvement_vs_sales_only": float(
            1 - best_test["pooled_wape"] / test_by[f"sales_only__{selected_profile}"]["pooled_wape"]
        ),
        "saved_models": str(MODEL_DIR),
    }
    legacy_manifest = {
        "selected_configuration": "top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe",
        "selection_status": "fixed historical primary recipe; purged replication",
        "test": test_by["top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe"],
        "sales_only_comparison": test_by["sales_only__legacy_published_recipe"],
        "wape_improvement_vs_sales_only": float(
            1 - test_by["top5_all_lags_1_7_14_legacy_rank__legacy_published_recipe"]["pooled_wape"]
            / test_by["sales_only__legacy_published_recipe"]["pooled_wape"]
        ),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
        "final_test_start": str(test_start.date()),
        "target_horizon_days": 14,
        "purge_days": 14,
        "signal_count": int(signals.shape[1]),
        "drug_count": int(frame.drug_name.nunique()),
        "signals_by_drug": {drug: legacy_selections[drug].get(legacy_variant.name, []) for drug in legacy_selections},
        "features_by_drug": {drug: legacy_selections[drug].get("final_features", []) for drug in legacy_selections},
        "saved_models": str(MODEL_DIR / "legacy_recipe"),
    }
    (OUT_DIR / "legacy_recipe_demand_model.json").write_text(json.dumps(legacy_manifest, indent=2), encoding="utf-8")
    (OUT_DIR / "selected_demand_model.json").write_text(json.dumps(selected_manifest, indent=2), encoding="utf-8")
    return {"manifest": selected_manifest, "validation": validation, "test": test,
            "comparison": comparison, "per_drug": details, "legacy_per_drug": legacy_details,
            "signal_cols": int(signals.shape[1])}


def write_comparison_report(demand: dict, initial: dict) -> None:
    comparison = demand["comparison"]
    rows = []
    for item in comparison:
        val = item["validation"]
        tst = item["final_test"]
        rows.append((item["configuration"], item["variant"]["signal_group"],
                     item["variant"]["count"], ",".join(map(str, item["variant"]["lags"])) or "none",
                     item["variant"]["selection_lag"],
                     item["profile"]["depth"], item["profile"]["estimators"],
                     item["profile"]["learning_rate"], item["profile"]["min_child_weight"],
                     item["profile"]["reg_lambda"], item["profile"]["colsample"],
                     item["profile"]["objective"], item["profile"]["tweedie_variance_power"],
                     item["profile"]["max_delta_step"], item["profile"]["seed"],
                     val.get("mean_drug_wape"),
                     tst.get("mean_drug_wape"), tst.get("pooled_wape")))
    lines = [
        "# Model comparison: rebuilt demand models",
        "",
        "## Protocol",
        "",
        f"The grid has {len(VARIANTS)} feature-input variants × {len(PROFILES)} XGBoost profiles = {len(VARIANTS) * len(PROFILES)} configurations. Each of 30 synthetic drug series gets its own model. The first 60% of dates select signals and fit validation models; the next 20% selects the configuration by unweighted mean per-drug WAPE. The validation winner is then refit on the first 80% and evaluated once on the final 20%. A 14-day purge before each boundary prevents the next-14-day training labels from overlapping validation or final-test outcomes. No configuration was selected from final-test results.",
        "",
        f"The synthetic data covers {demand['manifest']['date_range'][0]} through {demand['manifest']['date_range'][1]}. Validation begins {demand['manifest']['validation_start']}; final test begins {demand['manifest']['final_test_start']}. The signal table expands to {demand['signal_cols']:,} dated candidate columns with {demand['manifest']['signal_rows']:,} source rows.",
        "",
        "Signal rankings use only training dates and absolute Pearson correlation between each candidate’s one-day lag and the next-14-day demand target, except the explicitly named legacy-rank variant, which ranks raw values to reproduce the earlier benchmark's rule. Signal lags are 1, 7, 14, and optionally 28 days. The all-signals row uses 1,312 lagged features. All signal inputs are forward-filled only after their recorded period end and begin at zero before the first observation.",
        "",
        "All profiles use subsample=0.85, histogram tree building, and one worker. The grid includes squared-error, Poisson count, Tweedie, and pseudo-Huber objectives alongside five tree-depth/regularization/estimator profiles. The historical profile retains seed 20250915; all new profiles use 20261008. Sales features are prior-day sales lags through 56 days, rolling means and standard deviations through 56 days, calendar terms, and lagged price/stockout indicators.",
        "",
        "Metrics are aggregated over forecast origins; `mean drug WAPE` is the unweighted average of per-drug WAPE. `pooled WAPE` weights errors by total demand. Predictions are clamped at zero. Synthetic event tags are never model features.",
        "",
        f"## {len(VARIANTS) * len(PROFILES)} configuration results",
        "",
        "| Configuration | Signal source | Count | Signal lags | Rank lag | Depth | Trees | Learning rate | Min child weight | Lambda | Column sample | Objective | Tweedie power | Max delta step | Seed | Validation mean drug WAPE |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for name, group, count, lags, rank_lag, depth, trees, lr, child, reg_lambda, colsample, objective, tweedie_power, delta_step, seed, val, tst, pooled in rows:
        fmt = lambda x: "n/a" if x is None else f"{x:.2%}"
        power = "n/a" if tweedie_power is None else f"{tweedie_power:.1f}"
        lines.append(f"| {name} | {group} | {count} | {lags} | {rank_lag} | {depth} | {trees} | {lr:.3f} | {child:g} | {reg_lambda:g} | {colsample:.2f} | {objective} | {power} | {delta_step:g} | {seed} | {fmt(val)} |")
    selected = demand["manifest"]["selected_configuration"]
    test = demand["manifest"]["test"]
    baseline = next(row for row in demand["test"] if row["variant"] == "seasonal_naive")
    lines.extend([
        "",
        "## Selected model and per-drug results",
        "",
        f"Validation selected `{selected}` at {demand['manifest']['validation_mean_drug_wape']:.2%} mean per-drug WAPE. It achieved {test['mean_drug_wape']:.2%} mean per-drug WAPE and {test['pooled_wape']:.2%} pooled WAPE on the final period. The same-profile sales-only XGBoost scored {demand['manifest']['sales_only_comparison']['pooled_wape']:.2%} pooled WAPE; selected signals changed pooled WAPE by {demand['manifest']['wape_improvement_vs_sales_only']:.2%}. Seasonal-naive scored {baseline['pooled_wape']:.2%} pooled WAPE.",
        "",
        "The repository already contains an earlier report in `../data/synthetic_pharmacy_data/current_benchmark_results.md`: top-five signals at 1-, 7-, and 14-day lags were reported at 29.40% WAPE versus 32.35% for sales-only on this synthetic final period. That runner did not purge the 14-day target horizon at the train/test boundary, so its numbers are exploratory and may be optimistic; its XGBoost recipe and selection implementation also differ. This purged expanded grid is reported below and its validation winner is compared with a matched sales-only baseline on final dates.",
        f"The prior fixed recipe is also rerun directly with a 14-day purge: top-five raw-value correlation selection, 1/7/14-day signal lags, and the original 220-tree depth-2 XGBoost settings. It scores {demand['manifest']['legacy_recipe_test']['pooled_wape']:.2%} pooled WAPE versus {demand['manifest']['legacy_recipe_sales_only']['pooled_wape']:.2%} for the same-profile sales-only baseline. Its configuration is included in the comparison table; this isolates the effect of the boundary purge from the newer grid's tuning choices.",
        "The historical recipe is saved as a separate inference option in `rebuilt_demand/models/legacy_recipe/` with its own feature manifest, `rebuilt_demand/legacy_recipe_demand_model.json`. Its paired per-drug results are saved in `rebuilt_demand/legacy_recipe_per_drug_test.csv`.",
        "",
        "| Drug | Historical recipe WAPE | Matched sales-only WAPE | WAPE reduction |",
        "|---|---:|---:|---:|",
    ])
    for item in demand["legacy_per_drug"]:
        lines.append(f"| {item['drug_name']} | {item['wape']:.2%} | {item['sales_only_wape']:.2%} | {item['wape_improvement_vs_sales_only']:.2%} |")
    lines.extend([
        "",
        "| Drug | Signal model WAPE | Sales-only WAPE | WAPE reduction | MAE | RMSE | Origins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in demand["per_drug"]:
        lines.append(f"| {item['drug_name']} | {item['wape']:.2%} | {item['sales_only_wape']:.2%} | {item['wape_improvement_vs_sales_only']:.2%} | {item['mae']:.3f} | {item['rmse']:.3f} | {item['n']} |")
    lines.extend([
        "",
        "The full configuration objects and per-split metrics are in `rebuilt_demand/comparison_results.json`; the compact table is in `rebuilt_demand/comparison_results.csv`. Per-drug test metrics for the validation-selected model and fixed historical recipe are in `rebuilt_demand/selected_model_per_drug_test.csv` and `rebuilt_demand/legacy_recipe_per_drug_test.csv`.",
        "",
        "## Initial 20-signal replacement",
        "",
        "The original article-to-signal runner and weights were not found in the inspected Git history. The available 20-column table is a dated output artifact, not per-article training labels. The 3DLNews `.gz` files are Git LFS pointers, but a separate GDELT article corpus is present: 354 records, 292 with full text, across only 10 distinct publication months from 2023-10 through 2025-12. Those sparse month aggregates are insufficient for a defensible article-conditioned model over the 97 monthly target rows. The replacement therefore forecasts each of the same 20 output columns one month ahead from their own historical lags and calendar features. It does not claim to recover news understanding or reproduce the original model.",
        "",
        "The historical traces support that boundary: commit `d5e11df` contains a downstream `prod_pipeline.py` that consumes dated 20-signal outputs; `d6ca4b7` archives a README, completion gate, and monthly signal table without an upstream runner or weights; and `6d13c14` summarizes already-extracted `lm_event`/`lm_stage`/`lm_direction` fields. The local news artifact metadata mentions GDELT and FLAN-T5-small, but no matching inference implementation or weights were found. The investigation is detailed in `docs/NEWS_ONLY_INTEGRATION.md`.",
        "",
        f"It trains on {initial['source_rows']} monthly output rows from {initial['source_date_range'][0]} to {initial['source_date_range'][1]} and evaluates the last {initial['holdout_rows']} rows beginning {initial['holdout_start']}. Mean holdout MAE is {initial['mean_xgb_mae']:.4f} versus {initial['mean_persistence_mae']:.4f} for persistence. Per-signal metrics and the 20 next-month output values are saved in `rebuilt_demand/initial_20_metrics.json` and `rebuilt_demand/models/initial_20_signals/next_month_20_signals.csv`.",
        "",
        "A separate text-to-output experiment is in `rebuilt_demand/train_news_text.py`. It uses the local GDELT articles and same-month archived outputs; only 10 months overlap (six train, two validation, two test). Its two-month test pooled WAPE was 11.91% versus 87.07% for persistence, with pooled MAE 0.944 versus 6.9. Because the test covers only two months and uses same-month article aggregates, the result is exploratory and not a reliable ahead-of-time forecast. Reproduction outputs are `rebuilt_demand/article_text_20_signal_metrics.json`, `rebuilt_demand/article_text_20_signal_test_predictions.csv`, and `rebuilt_demand/models/initial_20_signals/article_text_model.joblib`.",
        "",
        "## Interpretation and limitations",
        "",
        "All demand scores are on the repository's deterministic single-site synthetic panel. Performance does not establish accuracy on observed pharmacy sales. The 1,312 table includes model-derived outputs, so the benchmark measures their conditional association with synthetic demand, not independent external predictive value. The historical and rebuilt WAPE results use the same documented final date window and WAPE definition, but differ in feature-selection details and XGBoost settings; compare the recipes before attributing the score difference to one change.",
        "",
    ])
    (OUT_DIR.parent / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def rewrite_report_from_saved_results() -> None:
    """Rebuild the human-readable report without fitting any models."""
    comparison = json.loads((OUT_DIR / "comparison_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT_DIR / "selected_demand_model.json").read_text(encoding="utf-8"))
    frame = make_sales_frame()
    seasonal_rows = frame.loc[frame.date.ge(pd.Timestamp(manifest["final_test_start"]))].dropna(
        subset=["target_14d", "mean_14"]
    )
    seasonal_metrics = metric_pack(seasonal_rows.target_14d, seasonal_rows.mean_14 * 14)
    seasonal = {"pooled_wape": seasonal_metrics["wape"]}
    demand = {
        "comparison": comparison,
        "manifest": manifest,
        "test": [{"variant": "seasonal_naive", **seasonal}],
        "per_drug": pd.read_csv(OUT_DIR / "selected_model_per_drug_test.csv").to_dict("records"),
        "legacy_per_drug": pd.read_csv(OUT_DIR / "legacy_recipe_per_drug_test.csv").to_dict("records"),
        "signal_cols": manifest["signal_count"],
    }
    initial = json.loads((OUT_DIR / "initial_20_metrics.json").read_text(encoding="utf-8"))
    write_comparison_report(demand, initial)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-20-signal", action="store_true", help="Skip refitting the temporal 20-output replacement")
    parser.add_argument("--report-only", action="store_true", help="Regenerate comparison.md from saved metrics without fitting")
    args = parser.parse_args()
    if args.report_only:
        rewrite_report_from_saved_results()
        print(f"wrote {OUT_DIR.parent / 'comparison.md'}")
        return
    initial = None if args.skip_20_signal else train_twenty_signal_forecaster()
    demand = run_demand_comparison()
    if initial is None:
        initial = json.loads((OUT_DIR / "initial_20_metrics.json").read_text(encoding="utf-8"))
    write_comparison_report(demand, initial)
    print(json.dumps({
        "selected": demand["manifest"]["selected_configuration"],
        "validation_mean_drug_wape": demand["manifest"]["validation_mean_drug_wape"],
        "test_mean_drug_wape": demand["manifest"]["test"]["mean_drug_wape"],
        "test_pooled_wape": demand["manifest"]["test"]["pooled_wape"],
        "configurations": demand["manifest"]["configuration_count"],
        "signal_columns": demand["signal_cols"],
        "initial_20_signal_mean_mae": initial["mean_xgb_mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
