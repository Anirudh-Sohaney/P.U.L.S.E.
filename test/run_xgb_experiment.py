"""Per-drug, direct 14-day XGBoost experiment on the supplied synthetic sales.

The script is deliberately self-contained and reads model signals only from
the existing artifact; it never writes to data/ or model/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SALES = ROOT / "data/synthetic_pharmacy_data/arkansas_clinic_daily_pharmacy_sales.csv"
NEWS = ROOT / "model/artifacts/news/news_only_catalog_features.csv.gz"
OUT = Path(__file__).resolve().parent / "experiment_metrics.json"


def make_frame(include_news=False):
    raw = pd.read_csv(SALES, parse_dates=["date"])
    raw = raw.sort_values(["drug_name", "date"]).reset_index(drop=True)
    # Keep the target direct and non-overlapping with the information at origin t.
    g = raw.groupby("drug_name", sort=False)["units_sold"]
    raw["target_14d"] = g.transform(lambda s: s.shift(-1).rolling(14, min_periods=14).sum().shift(-13))
    for lag in [1, 2, 3, 7, 14, 21, 28, 56]:
        raw[f"lag_{lag}"] = g.shift(lag)
    for window in [3, 7, 14, 28, 56]:
        shifted = g.shift(1)
        raw[f"mean_{window}"] = shifted.groupby(raw["drug_name"]).transform(lambda s: s.rolling(window, min_periods=window).mean())
        raw[f"std_{window}"] = shifted.groupby(raw["drug_name"]).transform(lambda s: s.rolling(window, min_periods=window).std())
    raw["dow"] = raw.date.dt.dayofweek
    raw["weekofyear"] = raw.date.dt.isocalendar().week.astype(int)
    raw["month"] = raw.date.dt.month
    raw["day_of_year"] = raw.date.dt.dayofyear
    raw["sin_year"] = np.sin(2 * np.pi * raw.day_of_year / 365.25)
    raw["cos_year"] = np.cos(2 * np.pi * raw.day_of_year / 365.25)
    raw["price_lag1"] = raw.groupby("drug_name").unit_price_usd.shift(1)
    raw["stockout_lag1"] = raw.groupby("drug_name").stockout_flag.shift(1)
    if include_news:
        news = pd.read_csv(NEWS, parse_dates=["date"])
        # Signals dated at month start are used only from the next calendar month.
        news["signal_month"] = news.date.dt.to_period("M")
        signal_cols = [c for c in news.columns if c not in {"date", "signal_month"}]
        news["signal_month"] = news.signal_month + 1
        news = news[["signal_month"] + signal_cols].rename(columns={c: f"news_{c}" for c in signal_cols})
        raw["signal_month"] = raw.date.dt.to_period("M")
        raw = raw.merge(news, on="signal_month", how="left").drop(columns="signal_month")
    return raw


def metrics(y, p):
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 0)
    denom = np.maximum(np.abs(y) + np.abs(p), 1e-9)
    rel = np.abs(y - p) / np.maximum(y, 1.0)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "wape": float(np.abs(y-p).sum()/max(np.abs(y).sum(), 1e-9)),
            "smape": float((2*np.abs(y-p)/denom).mean()),
            "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) > 0 else None,
            "within_5pct": float((rel <= .05).mean()), "within_10pct": float((rel <= .10).mean()),
            "within_20pct": float((rel <= .20).mean()), "n": int(len(y))}


def run_variant(name, df, feature_cols):
    cutoff = df.date.min() + (df.date.max() - df.date.min()) * .70
    preds, actual, baseline, drugs = [], [], [], []
    for drug, part in df.groupby("drug_name", sort=True):
        part = part.dropna(subset=feature_cols + ["target_14d"]).copy()
        train, test = part[part.date <= cutoff], part[part.date > cutoff]
        if len(train) < 100 or len(test) == 0: continue
        model = XGBRegressor(n_estimators=350, max_depth=3, learning_rate=.035,
                             min_child_weight=5, subsample=.85, colsample_bytree=.85,
                             reg_lambda=5, objective="reg:squarederror", tree_method="hist",
                             n_jobs=2, random_state=20250915)
        model.fit(train[feature_cols], train.target_14d, verbose=False)
        p = model.predict(test[feature_cols])
        # A transparent operational baseline: most recent observed 14-day total.
        b = (test["mean_14"] * 14).to_numpy()
        preds.extend(p); actual.extend(test.target_14d); baseline.extend(b); drugs.extend([drug]*len(test))
    overall = metrics(actual, preds)
    base = metrics(actual, baseline)
    per_drug = pd.DataFrame({"drug":drugs,"y":actual,"p":preds}).groupby("drug").apply(lambda x: metrics(x.y,x.p), include_groups=False).to_dict()
    return {"variant":name, "features":feature_cols, "cutoff":str(cutoff.date()), "overall":overall,
            "baseline":base, "baseline_wape_improvement":float(1-overall["wape"]/base["wape"]),
            "per_drug":per_drug, "drugs_evaluated":len(per_drug)}


def main():
    base = make_frame(False)
    lag = [c for c in base.columns if c.startswith("lag_") and int(c.split("_")[1]) <= 28]
    stats = [c for c in base.columns if c.startswith(("mean_", "std_")) and int(c.split("_")[1]) <= 28]
    calendar = ["dow", "weekofyear", "month", "sin_year", "cos_year", "price_lag1", "stockout_lag1"]
    variants = [("recent_lags", lag), ("lags_stats_calendar", lag+stats+calendar),
                ("long_history_stats_calendar", [c for c in base.columns if c.startswith(("lag_", "mean_", "std_"))]+calendar)]
    results = []
    for name, cols in variants: results.append(run_variant(name, base, cols))
    news = make_frame(True)
    news_cols = [c for c in news.columns if c.startswith("news_")]
    results.append(run_variant("long_history_stats_calendar_prior_month_model_signals",
                               news, variants[-1][1]+news_cols))
    OUT.write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda x: x["overall"]["wape"])
    print(json.dumps({"best":best["variant"], "variants":[{r["variant"]:r["overall"]} for r in results]}, indent=2))


if __name__ == "__main__": main()
