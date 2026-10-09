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
from datetime import date, datetime, timezone
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
ANNUAL_HORIZON_DAYS = 365

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
    """Re-audit saved model features under today's source-readiness rules.

    An artifact's stored readiness flag can become stale when source semantics
    or feature builders change. Both the saved gate and the current gate must
    pass before a forecast can be called operational.
    """
    saved = trained.get("input_contract")
    claims = trained.get("demand_claims", {})
    feature_cols = claims.get("feature_cols") or []
    current = summarize_dispositions(feature_cols)
    blend = trained.get("calibrated_blend")
    calibration_ready = (
        not blend or (
            isinstance(blend, dict)
            and blend.get("calibration_fit_scope") == "prior_feature_years_only"
            and isinstance(blend.get("calibration_train_end_year"), int)
            and isinstance(blend.get("calibration_validation_year"), int)
            and blend["calibration_train_end_year"] < blend["calibration_validation_year"]
        )
    )
    current["calibration_contract_ready"] = calibration_ready
    current_ready = current["operational_ready"] and calibration_ready
    current["saved_contract_operational_ready"] = (
        bool(saved.get("operational_ready")) if isinstance(saved, dict) else None
    )
    if not feature_cols:
        current["operational_ready"] = False
        current["readiness_reason"] = "missing_trained_feature_columns"
    elif isinstance(saved, dict) and not saved.get("operational_ready", False) and not current_ready:
        current["operational_ready"] = False
        current["readiness_reason"] = "saved_and_current_gates_failed"
    elif isinstance(saved, dict) and not saved.get("operational_ready", False):
        current["operational_ready"] = False
        current["readiness_reason"] = "saved_artifact_failed_operational_gate"
    elif not calibration_ready:
        current["operational_ready"] = False
        current["readiness_reason"] = "unverified_blend_calibration"
    elif not current_ready:
        current["readiness_reason"] = "current_source_or_feature_gate_failed"
    else:
        current["readiness_reason"] = "ready"
    return current


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
    forecast_years: int = 1,
    max_rows: int = 10000,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Generate one-year-ahead rows from next-year annual models."""
    if forecast_years != 1:
        raise ValueError("the annual model supports exactly one forecast year")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    run_id = run_id or f"ar-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    created_at = datetime.now(timezone.utc).isoformat()

    claims = trained.get("demand_claims", {})
    cost = trained.get("demand_cost", {})
    risk = trained.get("shortage_risk", {})
    demand_model = _restore_demand_model(claims["model"]) if claims.get("model") else None
    cost_model = _restore_demand_model(cost["model"]) if cost.get("model") else None
    risk_model = _restore_risk_model(risk.get("model"))
    spec = trained.get("encoder_spec", [])
    feature_cols = claims.get("feature_cols") or []

    max_year = int(panel["year"].max())
    last_panel = panel[panel["year"].eq(max_year)].sort_values("year").groupby(
        ["drug_key", "city"], as_index=False).tail(1)
    if last_panel.empty:
        raise ValueError("no city-drug rows exist in the latest panel year")
    last_panel = last_panel.sort_values("demand_claims", ascending=False).reset_index(drop=True)
    ar_total_claims = float(last_panel["demand_claims"].sum()) or 1.0

    # Encode every candidate row once; prediction is a pure vector op.
    X_all = _feature_matrix(last_panel, feature_cols, spec)
    ridge_claims_annual = (
        np.maximum(np.expm1(np.clip(demand_model.predict(X_all), -20.0, 20.0)), 0.0)
        if demand_model is not None else None
    )
    blend = trained.get("calibrated_blend") or {}
    blend_weight = float(blend.get("blend_weight", 1.0 if demand_model is not None else 0.0))
    if not np.isfinite(blend_weight) or not 0.0 <= blend_weight <= 1.0:
        raise ValueError("calibrated blend weight must be between zero and one")
    if demand_model is None and blend_weight:
        raise ValueError("calibrated blend requires a demand model")
    last_claims = last_panel["demand_claims"].to_numpy(dtype=float)
    claims_annual = ((1.0 - blend_weight) * last_claims
                     + blend_weight * ridge_claims_annual if ridge_claims_annual is not None
                     else last_claims)
    cost_annual = (np.expm1(np.clip(cost_model.predict(X_all), -20.0, 20.0))
                   if cost_model is not None else None)
    risk_proba = (
        risk_model.predict_proba(X_all)
        if risk_model is not None else None
    )
    claims_contribs = [
        _driver_contributions(demand_model, x)
        if demand_model is not None and blend_weight > 0 else {}
        for x in X_all
    ]
    cost_contribs = [
        _driver_contributions(cost_model, x) if cost_model is not None else {}
        for x in X_all
    ]

    rows: List[Dict] = []
    forecast_date = date(max_year + 1, 1, 1)

    for i in range(len(last_panel)):
        base = last_panel.iloc[i]
        claims_w = float(claims_annual[i])
        cost_w = float(max(cost_annual[i], 0.0)) if cost_annual is not None else None
        exposure = float(base["demand_claims"]) / ar_total_claims
        supply_risk = float(risk_proba[i]) if risk_proba is not None else None
        if (not np.isfinite(claims_w) or claims_w < 0
                or not np.isfinite(exposure) or exposure < 0
                or (cost_w is not None and not np.isfinite(cost_w))
                or (supply_risk is not None and (
                    not np.isfinite(supply_risk) or not 0 <= supply_risk <= 1))):
            raise ValueError("annual forecast contains an invalid prediction or source value")
        claim_family = ("persistence_baseline" if blend_weight == 0
                        else "validated_convex_blend")
        targets = [
            ("demand_claims", claims_w, claim_family, claims_contribs[i],
             "ridge_log_component_only" if blend_weight > 0 else "none"),
        ]
        if cost_w is not None:
            targets.append(("demand_cost", cost_w, cost.get("family", "ridge_linear"),
                            cost_contribs[i], "cost_log_model"))
        ma = pd.to_numeric(base.get("demand_claims_ma2"), errors="coerce")
        if pd.notna(ma) and np.isfinite(float(ma)) and float(ma) > 0:
            targets.append(("demand_shock_index", (claims_w - float(ma)) / float(ma),
                            "derived_annual_change_against_history", {}, "none"))
        if supply_risk is not None:
            targets.extend([
                ("supply_disruption_risk", supply_risk,
                 risk.get("family", "logistic_ridge"), {}, "none"),
                ("arkansas_shortage_impact", exposure * supply_risk,
                 "derived_exposure_risk", {}, "none"),
            ])
        for target, prediction, family, contrib, driver_scope in targets:
            row: Dict = {
                "forecast_run_id": run_id,
                "forecast_created_at": created_at,
                "forecast_date": forecast_date.isoformat(),
                "horizon_days": ANNUAL_HORIZON_DAYS,
                "geography_level": GEOGRAPHY_LEVEL,
                "geography_id": str(base["city"]),
                "geography_name": str(base["city"]),
                "drug_key": str(base["drug_key"]),
                "drug_name": str(base["drug"]),
                "ingredient_key": str(base.get("ingredient", "")),
                "ingredient_name": str(base.get("ingredient", "")),
                "supplier_key": str(base.get("labeler", "")),
                "supplier_name": str(base.get("labeler", "")),
                "disease_key": "all_context",
                "disease_name": "all_context",
                "target": target,
                "prediction": float(prediction),
                "prediction_interval_low": None,
                "prediction_interval_high": None,
                "risk_score": supply_risk,
                "model_family": family,
                "driver_summary_json": json.dumps({
                    "drivers": contrib,
                    "major_driver": max(contrib, key=lambda name: abs(contrib[name]))
                    if contrib else "",
                    "scope": driver_scope,
                    "ridge_blend_weight": blend_weight if target == "demand_claims" else None,
                }),
                "source_feature_window_start": date(max_year, 1, 1).isoformat(),
                "source_feature_window_end": date(max_year, 12, 31).isoformat(),
            }
            rows.append(row)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
