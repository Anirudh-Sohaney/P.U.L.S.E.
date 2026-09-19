"""Leakage-safe evaluation of the restored news-only signal surface.

The evaluator treats the dated news output as the larger model's external
representation and trains one small binary head per independently observed
monthly target. Targets are public proxies, not pharmacy inventory labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .news_only_adapter import NEWS_ONLY_SIGNAL_IDS, load_news_only_features
from .regression import LogisticRidge
from .config import Config


BASE_TARGET_COLUMNS = (
    "hhs_claim_lines", "hhs_paid", "hhs_provider_activity",
    "fda_active_shortages", "fda_new_shortages", "fda_resolutions",
    "flu_ar_ili", "flu_ar_wili", "flu_national_ili", "flu_national_wili",
    "flu_ar_patients", "flu_ar_providers", "flu_national_patients",
    "flu_national_providers", "flu_national_age_0", "flu_national_age_1",
    "flu_national_age_3", "flu_national_age_4", "flu_national_age_5",
)


def _month(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, errors="coerce").dt.to_period("M").astype(str)


def build_monthly_targets(root: Path) -> pd.DataFrame:
    """Build ten observed monthly proxy targets from local public data."""
    hhs_path = root / "data/targeted_additions/hhs_medicaid_provider_spending_ndc/data/arkansas_pharmacy_ndc_monthly.csv.gz"
    hhs_raw = pd.read_csv(hhs_path, usecols=["month", "drug_key", "demand_claim_lines",
                                             "demand_paid", "pharmacy_provider_count"])
    hhs_raw["month"] = _month(hhs_raw["month"])
    # Choose the consistently observed drug series using the training-era
    # coverage window. The target values themselves remain held out below.
    train_era = hhs_raw[hhs_raw["month"] <= "2021-12"]
    drug_stats = train_era.groupby("drug_key").agg(
        train_months=("month", "nunique"),
        demand_total=("demand_claim_lines", "sum"),
    )
    top_drugs = (drug_stats.sort_values(["train_months", "demand_total"],
                                        ascending=False).head(10).index
                 .astype(str).tolist())
    hhs = hhs_raw
    hhs = hhs.groupby("month", as_index=False).agg(
        hhs_claim_lines=("demand_claim_lines", "sum"),
        hhs_paid=("demand_paid", "sum"),
        hhs_provider_activity=("pharmacy_provider_count", "sum"),
    )
    drug_rows = hhs_raw[hhs_raw["drug_key"].astype(str).isin(top_drugs)]
    drug_targets = drug_rows.pivot_table(index="month", columns="drug_key",
                                         values="demand_claim_lines", aggfunc="sum")
    drug_targets.columns = [f"hhs_drug_{str(c)}_claim_lines" for c in drug_targets.columns]
    hhs = hhs.merge(drug_targets.reset_index(), on="month", how="outer")

    fda_path = root / "data/targeted_additions/fda_shortage_archive/data/fda_shortage_monthly.csv"
    fda_raw = pd.read_csv(fda_path, usecols=["month", "ndc9", "shortage_active", "resolution_observed",
                                         "first_posting_month"])
    fda_raw["month"] = _month(fda_raw["month"])
    fda_raw["first_posting_month"] = _month(fda_raw["first_posting_month"])
    fda_train = fda_raw[fda_raw["month"] <= "2021-12"]
    top_ndcs = (fda_train.assign(active=fda_train["shortage_active"].astype(float))
                .groupby("ndc9")["active"].sum().sort_values(ascending=False)
                .head(10).index.astype(str).tolist())
    fda = fda_raw
    fda = fda.groupby("month", as_index=False).agg(
        fda_active_shortages=("shortage_active", "sum"),
        fda_resolutions=("resolution_observed", "sum"),
        fda_new_shortages=("first_posting_month", lambda x: int(x.notna().sum())),
    )
    ndc_targets = (fda_raw[fda_raw["ndc9"].astype(str).isin(top_ndcs)]
                   .groupby(["month", "ndc9"])["shortage_active"].max()
                   .unstack("ndc9").fillna(0.0))
    ndc_targets.columns = [f"fda_ndc_{str(c)}_active" for c in ndc_targets.columns]
    fda = fda.merge(ndc_targets.reset_index(), on="month", how="outer")

    flu_path = root / "data/targeted_additions/disease_surveillance_current/data/cdc_fluview_ar_national_weekly.csv.gz"
    flu_measures = ["ili", "wili", "num_patients", "num_providers",
                    "num_age_0", "num_age_1", "num_age_3", "num_age_4", "num_age_5"]
    flu = pd.read_csv(flu_path, usecols=["region", "epiweek", *flu_measures])
    issue = pd.to_numeric(flu["epiweek"], errors="coerce")
    flu["issue_date"] = pd.to_datetime(
        issue.floordiv(100).astype("Int64").astype(str) + "-" +
        issue.mod(100).astype("Int64").astype(str) + "-1",
        format="%G-%V-%u", errors="coerce")
    flu["month"] = flu["issue_date"].dt.to_period("M").astype(str)
    flu = flu.dropna(subset=["month", "ili", "wili"])
    flu = flu.pivot_table(index="month", columns="region", values=flu_measures,
                          aggfunc="mean")
    flu.columns = [f"flu_{region}_{measure}" for measure, region in flu.columns]
    flu = flu.reset_index()
    flu = flu.rename(columns={
        **{f"flu_{region}_{measure}": f"flu_{'national' if region == 'nat' else 'ar'}_{measure.removeprefix('num_')}"
           for region in ("ar", "nat") for measure in flu_measures},
    })
    flu_columns = [column for column in flu.columns if column.startswith("flu_")]
    return hhs.merge(fda, on="month", how="outer").merge(
        flu[["month", *flu_columns]],
        on="month", how="outer").sort_values("month").reset_index(drop=True)


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = actual.astype(bool)
    predicted = predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    return {
        "test_rows": int(len(actual)),
        "accuracy": float((actual == predicted).mean()),
        "true_positive_precision": float(tp / (tp + fp)) if tp + fp else None,
        "actual_positive_rows": int(actual.sum()),
        "predicted_positive_rows": int(predicted.sum()),
    }


def evaluate_news_signal_heads(root: Path, *, train_end: str = "2021-12",
                               include_target_lag: bool = True) -> dict:
    """Evaluate binary heads using prior-month news and optional target lags."""
    news = load_news_only_features(Config(root))
    news["month"] = news["date"].dt.to_period("M").astype(str)
    targets = build_monthly_targets(root)
    frame = targets.merge(news.drop(columns="date"), on="month", how="inner").sort_values("month")
    # A signal published in month t can only predict an outcome in month t+1.
    frame[list(NEWS_ONLY_SIGNAL_IDS)] = frame[list(NEWS_ONLY_SIGNAL_IDS)].shift(1)
    frame = frame.dropna(subset=list(NEWS_ONLY_SIGNAL_IDS)).reset_index(drop=True)
    target_columns = [*BASE_TARGET_COLUMNS,
                      *[c for c in frame.columns if c.startswith("hhs_drug_")],
                      *[c for c in frame.columns if c.startswith("fda_ndc_")]]
    lag_columns = []
    if include_target_lag:
        for target in target_columns:
            column = f"lag_{target}"
            frame[column] = frame[target].shift(1)
            lag_columns.append(column)
    train = frame[frame["month"] <= train_end]
    test = frame[frame["month"] > train_end]
    results = []
    for target in target_columns:
        usable_train = train[target].notna()
        usable_test = test[target].notna()
        if include_target_lag:
            usable_train &= train[f"lag_{target}"].notna()
            usable_test &= test[f"lag_{target}"].notna()
        if usable_train.sum() < 20 or usable_test.sum() < 25:
            continue
        # A high-activity event is defined before testing from the training
        # distribution only; the 75th percentile avoids majority-label
        # collapse while retaining a simple, reproducible binary target.
        threshold = float(train.loc[usable_train, target].quantile(0.75))
        if threshold == 0:
            threshold = 1.0
        y_train = (train.loc[usable_train, target].to_numpy(float) >= threshold).astype(float)
        y_test = (test.loc[usable_test, target].to_numpy(float) >= threshold).astype(float)
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        feature_columns = list(NEWS_ONLY_SIGNAL_IDS)
        if include_target_lag:
            feature_columns.append(f"lag_{target}")
        x_train = train.loc[usable_train, feature_columns].to_numpy(float)
        x_test = test.loc[usable_test, feature_columns].to_numpy(float)
        model = LogisticRidge(alpha=10.0, max_iter=80).fit(
            x_train, y_train, feature_columns)
        probability = model.predict_proba(x_test)
        predicted = probability >= 0.5
        majority = np.full(len(y_test), y_train.mean() >= 0.5)
        metrics = _binary_metrics(y_test, predicted)
        metrics["target"] = target
        metrics["threshold_fit_on_train"] = threshold
        metrics["majority_accuracy"] = float((y_test == majority).mean())
        metrics["beats_majority"] = bool(metrics["accuracy"] > metrics["majority_accuracy"])
        metrics["passes_accuracy_threshold"] = bool(metrics["accuracy"] >= 0.70)
        metrics["passes_event_precision_threshold"] = bool(
            metrics["true_positive_precision"] is not None and
            metrics["true_positive_precision"] >= 0.70)
        metrics["promotion_candidate"] = bool(
            metrics["beats_majority"] and
            (metrics["passes_accuracy_threshold"] or
             metrics["passes_event_precision_threshold"]))
        results.append(metrics)
    return {
        "protocol": "monthly_previous_news_to_next_observed_proxy_v1",
        "input_signals": list(NEWS_ONLY_SIGNAL_IDS),
        "target_columns": target_columns,
        "train_end": train_end,
        "feature_boundary": "previous calendar month; target lag is optional structured input",
        "include_target_lag": include_target_lag,
        "source_scope": "public Arkansas/national proxy data; not direct pharmacy inventory",
        "sources": {
            "news": "restored dated GDELT/FLAN-T5-small output; see model/docs/NEWS_ONLY_INTEGRATION.md",
            "hhs": "https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/",
            "fda": "https://www.accessdata.fda.gov/scripts/drugshortages/Drugshortages.cfm",
            "cdc": "https://www.cdc.gov/fluview/",
        },
        "rows_after_join": int(len(frame)),
        "results": results,
        "evaluated_count": len(results),
        "excluded_target_count": len(target_columns) - len(results),
        "passing_count": int(sum(r["promotion_candidate"] for r in results)),
    }
