"""Forecast grid writer.

Produces the full output schema from ARCHITECTURE.md with at least 100 rows
per run. Each row is keyed by (forecast date x horizon x geography x drug x
supplier x disease driver x target). Demand predictions come from strict
next-period models (features at t -> demand at t+1); risk scores come from the
calibrated shortage logistic model or an explicitly labeled artifact prior.
Coverage scales with --max-rows: higher budgets iterate over more city-drug
combinations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import io
from .datasets import apply_encoder, numeric_feature_columns
from .input_contract import summarize_dispositions
from .regression import LogisticRidge, RidgeLinear, _fill_nan

OUTPUT_COLUMNS = [
    "forecast_run_id", "forecast_created_at", "forecast_date",
    "horizon_days", "geography_level", "geography_id", "geography_name",
    "drug_key", "drug_name", "ingredient_key", "ingredient_name",
    "supplier_key", "supplier_name", "disease_key", "disease_name",
    "target", "prediction", "prediction_interval_low",
    "prediction_interval_high", "risk_score", "model_family",
    "driver_summary_json", "source_feature_window_start",
    "source_feature_window_end",
]

TARGETS = [
    "demand_claims",
    "demand_cost",
    "demand_shock_index",
    "supply_disruption_risk",
    "arkansas_shortage_impact",
]

HORIZONS_DAYS = [7, 28, 56, 91, 182]

DRIVER_FEATURES: Dict[str, List[str]] = {
    "influenza": ["ar_ili_mean", "ar_wili_mean", "nat_ili_mean", "nat_wili_mean",
                  "ww_flu_ar", "ww_flu_nat"],
    "respiratory_virus": ["ww_covid_ar", "ww_covid_nat", "ww_rsv_ar", "ww_rsv_nat",
                          "news_covid_articles", "news_influenza_articles"],
    "disaster": ["ar_disaster_active_mean", "ar_disaster_severity_max"],
    "supply": ["na_shortage_active_mean", "na_recall_active_mean",
               "shortage_active_total", "shortage_current_total",
               "recall_count_x", "recall_count_y", "recall_firms",
               "shortage_events", "shortage_active",
               "recall_class_1", "recall_class_2", "recall_class_3",
               "shortage_reason_demand", "shortage_reason_discontinuation",
               "shortage_reason_ingredient", "shortage_reason_other",
               "labeler_shortage_n", "labeler_recall_n",
               "event_count", "event_severity_mean", "event_confidence_mean",
               "arcos_distribution_grams", "arcos_distribution_zip3_count",
               "arcos_distribution_growth"],
    "disease_burden": [],  # matches any nndss_* feature
    "economic": ["ar_unemployment_mean", "na_ppi_mean", "us_tariff_rate_mean"],
    "demand_history": ["y_last_log", "demand_claims_lag1_log",
                       "demand_claims_lag2_log", "demand_claims_ma2_log",
                       "delta_log", "delta2_log"],
    "population": ["ar_population_total"],
    "global_trade": ["global_gscpi_mean"],
    # The news-only SLM bridge is intentionally exposed as a separate driver
    # so forecasts retain attribution to the imported signal family.
    "news_only_slm": [],
}

GEOGRAPHY_LEVEL = "prescriber_city"


def _restore_demand_model(spec: Dict):
    return RidgeLinear.from_dict(spec)


def _restore_risk_model(spec: Optional[Dict]):
    if not spec:
        return None
    return LogisticRidge.from_dict(spec)



def _driver_contributions(model: RidgeLinear, x: np.ndarray) -> Dict[str, float]:
    """Signed driver contributions from ridge contributions on one row."""
    if model is None:
        return {}
    contribs = model.contributions(x.reshape(1, -1))[0]
    out: Dict[str, float] = {}
    for driver, feats in DRIVER_FEATURES.items():
        total = 0.0
        for j, name in enumerate(model.feature_names):
            in_driver = name in feats or (driver == "disease_burden" and name.startswith("nndss_"))
            in_driver = in_driver or (driver == "news_only_slm" and name.startswith("news_only_"))
            if in_driver:
                total += float(contribs[j])
        if abs(total) > 1e-12:
            out[driver] = total
    return out



def _feature_frame(rows: pd.DataFrame, spec: List[dict]) -> pd.DataFrame:
    """Build numeric + encoded feature frame keyed by trained column names."""
    num_cols = numeric_feature_columns(rows)
    numeric = rows[num_cols].copy() if num_cols else pd.DataFrame(index=rows.index)
    encoded = apply_encoder(rows, spec)
    frame = pd.concat([numeric.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    # Feature builders may emit a legacy duplicate (notably repeated disease
    # aggregates).  Preserve the first column deterministically before model
    # feature reindexing; duplicate labels make pandas alignment undefined.
    return frame.loc[:, ~frame.columns.duplicated()]


def _feature_matrix(rows: pd.DataFrame, feature_cols: List[str],
                    spec: List[dict]) -> np.ndarray:
    frame = _feature_frame(rows, spec)
    frame = frame.reindex(columns=feature_cols, fill_value=0.0)
    return _fill_nan(frame[feature_cols].to_numpy(dtype=float))


def forecast_input_contract(trained: Dict) -> Dict:
    """Return the saved input contract, or a conservative fallback audit.

    Fallback re-audits ``demand_claims.feature_cols`` so forecast metadata can
    still report operational readiness for artifacts trained before the
    contract was stored.
    """
    saved = trained.get("input_contract")
    if saved:
        return saved
    claims = trained.get("demand_claims", {})
    feature_cols = claims.get("feature_cols") or []
    return summarize_dispositions(feature_cols)


def _risk_prior(risk_spec: Dict) -> float:
    """Use only a serialized prior when a calibrated risk model is absent."""
    value = pd.to_numeric(risk_spec.get("baseline_probability", 0.5),
                          errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        return 0.5
    return float(np.clip(value, 0.0, 1.0))


def build_forecast_grid(
    panel: pd.DataFrame,
    trained: Dict,
    cfg=None,
    forecast_years: int = 2,
    max_rows: int = 10000,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Generate the full forecast schema table from next-period models."""
    run_id = run_id or f"ar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    created_at = datetime.now(timezone.utc).isoformat()

    claims = trained.get("demand_claims", {})
    cost = trained.get("demand_cost", {})
    risk = trained.get("shortage_risk", {})
    demand_model = _restore_demand_model(claims["model"]) if claims.get("model") else None
    cost_model = _restore_demand_model(cost["model"]) if cost.get("model") else None
    risk_model = _restore_risk_model(risk.get("model"))
    risk_prior = _risk_prior(risk)
    spec = trained.get("encoder_spec", [])
    feature_cols = claims.get("feature_cols") or []

    max_year = int(panel["year"].max())
    last_panel = panel.sort_values("year").groupby(
        ["drug_key", "city"], as_index=False).tail(1)
    last_panel = last_panel.sort_values("demand_claims", ascending=False).reset_index(drop=True)
    ar_total_claims = float(last_panel["demand_claims"].sum()) or 1.0

    # Encode every candidate row once; prediction is a pure vector op.
    X_all = _feature_matrix(last_panel, feature_cols, spec)
    ridge_claims_annual = (
        np.expm1(np.clip(demand_model.predict(X_all), -20.0, 20.0))
        if demand_model is not None else np.zeros(len(X_all))
    )
    blend = trained.get("calibrated_blend") or {}
    blend_weight = float(blend.get("blend_weight", 1.0 if demand_model is not None else 0.0))
    last_claims = last_panel["demand_claims"].to_numpy(dtype=float)
    claims_annual = (1.0 - blend_weight) * last_claims + blend_weight * ridge_claims_annual
    cost_annual = (np.expm1(np.clip(cost_model.predict(X_all), -20.0, 20.0))
                   if cost_model is not None else np.zeros(len(X_all)))
    risk_proba = (
        risk_model.predict_proba(X_all)
        if risk_model is not None
        else np.full(len(last_panel), risk_prior, dtype=float)
    )
    contribs = [
        _driver_contributions(demand_model, x) if demand_model is not None else {}
        for x in X_all
    ]
    resid_claims = float(demand_model.residual_std) if demand_model else 0.0

    rows: List[Dict] = []
    forecast_dates = [datetime(max_year + y, 1, 1) for y in range(1, forecast_years + 1)]

    for i in range(len(last_panel)):
        base = last_panel.iloc[i]
        claims_w = float(claims_annual[i])
        cost_w = float(cost_annual[i])
        exposure = float(base["demand_claims"]) / ar_total_claims
        contrib = contribs[i]
        supply_risk = float(risk_proba[i])

        for fdate in forecast_dates:
            for horizon in HORIZONS_DAYS:
                window_claims = claims_w * (horizon / 365.0)
                window_cost = cost_w * (horizon / 365.0)
                ma = float(base.get("demand_claims_ma2", 0.0)) or 0.0
                shock = (window_claims - ma) / max(ma, 1.0) if ma > 0 else 0.0
                shortage_impact = exposure * supply_risk

                # Intervals on the model's log scale for demand targets.
                resid = resid_claims * (horizon / 365.0)
                targets = {
                    "demand_claims": window_claims,
                    "demand_cost": window_cost,
                    "demand_shock_index": shock,
                    "supply_disruption_risk": supply_risk,
                    "arkansas_shortage_impact": shortage_impact,
                }
                for target, prediction in targets.items():
                    low = float(prediction) - 1.96 * resid
                    high = float(prediction) + 1.96 * resid
                    if target == "supply_disruption_risk":
                        low, high = max(0.0, low), min(1.0, high)
                    for driver in sorted(DRIVER_FEATURES):
                        row: Dict = {
                            "forecast_run_id": run_id,
                            "forecast_created_at": created_at,
                            "forecast_date": fdate.date().isoformat(),
                            "horizon_days": horizon,
                            "geography_level": GEOGRAPHY_LEVEL,
                            "geography_id": str(base["city"]),
                            "geography_name": str(base["city"]),
                            "drug_key": str(base["drug_key"]),
                            "drug_name": str(base["drug"]),
                            "ingredient_key": str(base.get("ingredient", "")),
                            "ingredient_name": str(base.get("ingredient", "")),
                            "supplier_key": str(base.get("labeler", "")),
                            "supplier_name": str(base.get("labeler", "")),
                            "disease_key": driver,
                            "disease_name": driver,
                            "target": target,
                            "prediction": float(prediction),
                            "prediction_interval_low": low,
                            "prediction_interval_high": high,
                            "risk_score": float(
                                supply_risk if target in (
                                    "supply_disruption_risk", "arkansas_shortage_impact",
                                    "demand_shock_index") else prediction),
                            "model_family": (
                                ("validated_convex_blend"
                                 if target == "demand_claims" and blend
                                 else claims.get("family", "ridge_linear"))
                                if target in ("demand_claims", "demand_cost")
                                else "risk_exposure_model"),
                            "driver_summary_json": json.dumps(
                                {"drivers": contrib,
                                 "major_driver": max(contrib, key=contrib.get) if contrib else "",
                                 "model_family": claims.get("family", "")}),
                            "source_feature_window_start": datetime(max_year, 1, 1).date().isoformat(),
                            "source_feature_window_end": datetime(max_year, 12, 31).date().isoformat(),
                        }
                        rows.append(row)
                        if len(rows) >= max_rows:
                            break
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
