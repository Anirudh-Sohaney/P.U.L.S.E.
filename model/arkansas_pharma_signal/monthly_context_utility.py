"""Monthly external-context utility test for Arkansas pharmacy demand."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .regression import RidgeLinear


BASE_FEATURES = [
    "current_log", "lag1_log", "lag2_log", "rolling3_log", "calendar_month",
]
CONTEXT_FEATURES = [
    "prior_shortage_active", "prior_shortage_supplier_count",
    "prior_news_count", "prior_ar_wili", "prior_nat_wili",
    "prior_nat_respnet_rsv",
]
APCD_CONTEXT_FEATURE = "prior_apcd_claim_count"


def _month(value: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(value.astype(str), freq="M")


def _monthly_context(
        shortage: pd.DataFrame | None, news: pd.DataFrame | None,
        fluview: pd.DataFrame | None, drug_keys: set[str],
        respnet: pd.DataFrame | None = None,
        apcd_claims: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate public context and shift it before joining to demand rows."""
    context = pd.DataFrame({"month": pd.period_range("2018-01", "2024-12", freq="M")})
    context["shortage_active"] = 0.0
    context["shortage_supplier_count"] = 0.0
    if shortage is not None and not shortage.empty:
        frame = shortage.copy()
        frame["ndc9"] = frame["ndc9"].astype(str).str.zfill(9)
        frame = frame[frame["ndc9"].isin(drug_keys)]
        frame["month"] = _month(frame["month"])
        frame["shortage_active"] = pd.to_numeric(
            frame["shortage_active"], errors="coerce").fillna(0.0)
        monthly = frame.groupby("month", as_index=False).agg(
            shortage_active=("shortage_active", "sum"),
            shortage_supplier_count=("supplier", "nunique"),
        )
        context = context.merge(monthly, on="month", how="left", suffixes=("", "_new"))
        for name in ("shortage_active", "shortage_supplier_count"):
            context[name] = context[f"{name}_new"].fillna(context[name])
            context = context.drop(columns=f"{name}_new")
    context["news_count"] = 0.0
    if news is not None and not news.empty:
        frame = news.copy()
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["month_number"] = pd.to_numeric(frame["month"], errors="coerce")
        frame["article_count"] = pd.to_numeric(frame["article_count"], errors="coerce").fillna(0.0)
        frame = frame.dropna(subset=["year", "month_number"])
        frame["month"] = pd.PeriodIndex(
            frame["year"].astype(int).astype(str) + "-" +
            frame["month_number"].astype(int).astype(str).str.zfill(2), freq="M")
        monthly = frame.groupby("month", as_index=False)["article_count"].sum()
        context = context.merge(monthly.rename(columns={"article_count": "news_count"}),
                                on="month", how="left", suffixes=("", "_new"))
        context["news_count"] = context["news_count_new"].fillna(context["news_count"])
        context = context.drop(columns="news_count_new")
    context["ar_wili"] = 0.0
    context["nat_wili"] = 0.0
    if fluview is not None and not fluview.empty:
        frame = fluview.copy()
        frame["month"] = pd.to_datetime(frame["release_date"], errors="coerce").dt.to_period("M")
        frame["wili"] = pd.to_numeric(frame["wili"], errors="coerce")
        monthly = frame.dropna(subset=["month", "wili"]).pivot_table(
            index="month", columns="region", values="wili", aggfunc="mean").reset_index()
        monthly = monthly.rename(columns={"ar": "ar_wili", "nat": "nat_wili"})
        context = context.merge(monthly, on="month", how="left", suffixes=("", "_new"))
        for name in ("ar_wili", "nat_wili"):
            if f"{name}_new" in context:
                context[name] = context[f"{name}_new"].fillna(context[name])
                context = context.drop(columns=f"{name}_new")
    context["nat_respnet_rsv"] = 0.0
    if respnet is not None and not respnet.empty:
        frame = respnet.copy()
        required = {"date", "estimate"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"RESP-NET data missing columns: {sorted(missing)}")
        frame["month"] = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M")
        frame["estimate"] = pd.to_numeric(frame["estimate"], errors="coerce")
        monthly = (frame.dropna(subset=["month", "estimate"])
                   .groupby("month", as_index=False)["estimate"].mean()
                   .rename(columns={"estimate": "nat_respnet_rsv"}))
        context = context.merge(monthly, on="month", how="left", suffixes=("", "_new"))
        context["nat_respnet_rsv"] = context["nat_respnet_rsv_new"].fillna(
            context["nat_respnet_rsv"])
        context = context.drop(columns="nat_respnet_rsv_new")
    if apcd_claims is not None and not apcd_claims.empty:
        frame = apcd_claims.copy()
        required = {"year", "month", "claim_count"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"APCD data missing columns: {sorted(missing)}")
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["month_number"] = pd.to_numeric(frame["month"], errors="coerce")
        frame["claim_count"] = pd.to_numeric(frame["claim_count"], errors="coerce")
        frame = frame.dropna(subset=["year", "month_number", "claim_count"])
        frame["month"] = pd.PeriodIndex(
            frame["year"].astype(int).astype(str) + "-" +
            frame["month_number"].astype(int).astype(str).str.zfill(2), freq="M")
        monthly = frame.groupby("month", as_index=False)["claim_count"].sum()
        context = context.merge(
            monthly.rename(columns={"claim_count": "apcd_claim_count"}),
            on="month", how="left")
        # Missing APCD periods are unavailable observations, not zero claims.
        # The utility script restricts evaluation to the source coverage window.
    # Current-month context is not available to a next-month forecast until the
    # following month. Shift every context field once before joining demand.
    for name in ("shortage_active", "shortage_supplier_count", "news_count",
                 "ar_wili", "nat_wili", "nat_respnet_rsv"):
        context[f"prior_{name}"] = context[name].shift(1).fillna(0.0)
    columns = ["month", *CONTEXT_FEATURES]
    if "apcd_claim_count" in context:
        context[APCD_CONTEXT_FEATURE] = context["apcd_claim_count"].shift(1).fillna(0.0)
        columns.append(APCD_CONTEXT_FEATURE)
    return context[columns]


def build_monthly_context_view(
        demand: pd.DataFrame, *, shortage: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None, fluview: pd.DataFrame | None = None,
        respnet: pd.DataFrame | None = None,
        apcd_claims: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build consecutive observed HHS NDC transitions with lagged context."""
    required = {"month", "drug_key", "demand_claim_lines"}
    missing = required.difference(demand.columns)
    if missing:
        raise ValueError(f"monthly demand missing columns: {sorted(missing)}")
    frame = demand.copy()
    frame["month"] = _month(frame["month"])
    frame["drug_key"] = frame["drug_key"].astype(str).str.zfill(11)
    frame["value"] = pd.to_numeric(frame["demand_claim_lines"], errors="coerce")
    frame = frame.dropna(subset=["month", "drug_key", "value"])
    frame = frame.groupby(["month", "drug_key"], as_index=False)["value"].sum()
    context = _monthly_context(
        shortage, news, fluview, set(frame["drug_key"].str[:9]), respnet,
        apcd_claims)
    frame = frame.merge(context, on="month", how="left")
    frame = frame.sort_values(["drug_key", "month"])
    group = frame.groupby("drug_key", sort=False)
    frame["next_month"] = group["month"].shift(-1)
    frame["target"] = group["value"].shift(-1)
    frame["current_log"] = np.log1p(frame["value"].clip(lower=0.0))
    frame["lag1_log"] = np.log1p(group["value"].shift(1).fillna(frame["value"]).clip(lower=0.0))
    frame["lag2_log"] = np.log1p(group["value"].shift(2).fillna(frame["value"]).clip(lower=0.0))
    frame["rolling3_log"] = group["value"].transform(
        lambda values: np.log1p(values.shift(1).rolling(3, min_periods=1).mean()
                                .fillna(values)))
    frame["calendar_month"] = frame["month"].dt.month
    frame = frame[frame["next_month"].eq(frame["month"] + 1)]
    return frame.reset_index(drop=True)


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    absolute = np.abs(actual - prediction)
    relative = absolute / np.maximum(np.abs(actual), 1.0)
    return {
        "wape": float(absolute.sum() / max(np.abs(actual).sum(), 1e-12)),
        "within_5_percent": float(np.mean(relative < 0.05)) if actual.size else 0.0,
    }


def _fit_predict(train: pd.DataFrame, target: pd.DataFrame,
                 features: list[str], alpha: float) -> np.ndarray:
    model = RidgeLinear(alpha=alpha).fit(
        train[features].to_numpy(float), np.log1p(train["target"].to_numpy(float)), features)
    cap = float(np.quantile(np.log1p(train["target"].to_numpy(float)), 0.995))
    return np.maximum(np.expm1(np.clip(model.predict(target[features].to_numpy(float)), 0.0, cap)), 0.0)


def evaluate_monthly_context_utility(
        view: pd.DataFrame, *, min_train_months: int = 24) -> dict[str, Any]:
    """Compare history-only and lagged external-context regression by fold."""
    context_features = [name for name in [*CONTEXT_FEATURES, APCD_CONTEXT_FEATURE]
                        if name in view.columns]
    months = sorted(view["month"].unique())
    folds = []
    for index in range(min_train_months, len(months)):
        train = view[view["month"].isin(months[:index - 1])]
        validation = view[view["month"].eq(months[index - 1])]
        test = view[view["month"].eq(months[index])]
        if train.empty or validation.empty or test.empty:
            continue
        selected = {}
        for name, features in (("base", BASE_FEATURES),
                               ("context", [*BASE_FEATURES, *context_features])):
            scores = {}
            for alpha in (0.1, 1.0, 10.0, 100.0):
                scores[alpha] = _metrics(
                    validation["target"], _fit_predict(train, validation, features, alpha)
                )["wape"]
            selected[name] = min(scores, key=scores.get)
        combined = pd.concat([train, validation], ignore_index=True)
        base = _metrics(test["target"], _fit_predict(
            combined, test, BASE_FEATURES, selected["base"]))
        context = _metrics(test["target"], _fit_predict(
            combined, test, [*BASE_FEATURES, *context_features], selected["context"]))
        folds.append({
            "validation_month": str(months[index - 1]), "test_month": str(months[index]),
            "train_rows": int(len(train)), "validation_rows": int(len(validation)),
            "test_rows": int(len(test)), "base": base, "context": context,
            "wape_delta_context_minus_base": context["wape"] - base["wape"],
            "within_5_delta_context_minus_base": context["within_5_percent"] - base["within_5_percent"],
            "selected_base_alpha": float(selected["base"]),
            "selected_context_alpha": float(selected["context"]),
        })
    if not folds:
        raise ValueError("no complete monthly context folds")
    deltas = [fold["wape_delta_context_minus_base"] for fold in folds]
    return {
        "protocol": "rolling_origin_next_month_hhs_pharmacy_external_context_utility",
        "fold_count": len(folds), "test_rows": int(sum(f["test_rows"] for f in folds)),
        "drug_count": int(view["drug_key"].nunique()),
        "context_feature_count": len(context_features),
        "context_features": context_features,
        "mean_base_wape": float(np.mean([f["base"]["wape"] for f in folds])),
        "mean_context_wape": float(np.mean([f["context"]["wape"] for f in folds])),
        "mean_wape_delta_context_minus_base": float(np.mean(deltas)),
        "mean_base_within_5_percent": float(np.mean([f["base"]["within_5_percent"] for f in folds])),
        "mean_context_within_5_percent": float(np.mean([f["context"]["within_5_percent"] for f in folds])),
        "all_folds_context_wape_improves": bool(all(delta < 0 for delta in deltas)),
        "publishable_candidate": False,
        "scope": "HHS Medicaid/CHIP Arkansas pharmacy-NDC utility proxy; not private inventory truth",
        "folds": folds,
    }
