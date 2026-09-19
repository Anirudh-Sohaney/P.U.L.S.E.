"""Live-style forecast adapters for the currently qualified metrics.

These adapters intentionally use the validation-approved persistence baseline
until a separately trained production checkpoint is supplied. Persistence is
not disguised as a learned model: the output provenance records the method,
feature period, and the fact that the target is an external proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external_metric_output import build_metric_rows
from .regional_demand_state import build_regional_demand_view, build_state_demand_view
from .recall_pressure import (build_recall_panel, build_supplier_recall_panel,
                              _source_edge_recall_predictions)
from .shortage_pressure import build_pressure_panel
from .nadac_target import _ndc9
from .weekly_health_proxy import load_fluview_weekly_proxy
from .hospital_respiratory import TARGET_COLUMNS
from .arcos_evaluation import build_latest_scoring_view
from .overdose_pressure import build_overdose_observations, build_overdose_view
from .nssp_respiratory import build_nssp_weekly_view
from .regional_wastewater import build_regional_wastewater_weekly_view
from .respnet import load_respnet_rsv_weekly
from .apcd_claim_counts import load_apcd_claim_counts, SOURCE_URL as APCD_SOURCE_URL
from .therapeutic_class_demand import build_latest_therapeutic_class_state


def _state_row(target: str, period: str, prediction: int, *, geography_level: str,
              geography_id: str, drug_key: str = "", ingredient: str = "",
              therapeutic_class: str = "", pathogen: str = "", supplier: str = "", county_fips: str = "", horizon: int,
              feature_period: str, source: str,
              attribution: dict) -> dict:
    """Create a state prediction with explicit proxy provenance."""
    return {
        "target": target, "forecast_period": period, "prediction": int(prediction),
        "horizon": int(horizon), "geography_level": geography_level,
        "geography_id": geography_id, "county_fips": county_fips, "county_name": "",
        "arkansas_region": geography_id if geography_level == "arkansas_region" else "",
        "drug_key": drug_key, "ingredient": ingredient, "therapeutic_class": therapeutic_class, "pathogen": pathogen,
        "labeler": "", "supplier": supplier,
        "api_source": source, "interval_low": None, "interval_high": None,
        "risk_score": None, "confidence": None, "source_freshness": feature_period,
        "uncertainty_status": "not_estimated", "calibration_status": "not_calibrated",
        "driver_attribution": json.dumps({
            "forecast_method": "validation_approved_persistence_baseline",
            "feature_period": feature_period, "target_proxy": True,
            **attribution,
        }), "evidence_type": "qualified_external_proxy_forecast",
        "feature_window_start": feature_period, "feature_window_end": feature_period,
    }


def _numeric_row(target: str, period: str, prediction: float, *, geography_level: str,
                 geography_id: str, drug_key: str = "", ingredient: str = "",
                 supplier: str = "", county_fips: str = "", horizon: int,
                 feature_period: str, source: str, attribution: dict) -> dict:
    """Create a non-promoted numeric candidate with explicit provenance."""
    return {
        "target": target, "forecast_period": period, "prediction": float(prediction),
        "horizon": int(horizon), "geography_level": geography_level,
        "geography_id": geography_id, "county_fips": county_fips, "county_name": "",
        "arkansas_region": geography_id if geography_level == "arkansas_region" else "",
        "drug_key": drug_key, "ingredient": ingredient, "labeler": "", "supplier": supplier,
        "api_source": source, "interval_low": None, "interval_high": None,
        "risk_score": None, "confidence": None, "source_freshness": feature_period,
        "uncertainty_status": "not_estimated", "calibration_status": "not_calibrated",
        "driver_attribution": json.dumps({
            "forecast_method": "validation_approved_persistence_baseline",
            "feature_period": feature_period, "target_proxy": True,
            **attribution,
        }), "evidence_type": "numeric_external_proxy_candidate",
        "feature_window_start": feature_period, "feature_window_end": feature_period,
    }


def forecast_latest_shortage_pressure(source: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-month FDA supplier-count states for every latest NDC."""
    panel = build_pressure_panel(source)
    # A pair whose archive history ended years ago is not a live forecast
    # candidate. Keep only NDCs present in the most recent source month.
    latest_month = panel["month"].max()
    latest = panel[panel["month"].eq(latest_month)].copy().reset_index(drop=True)
    rows = []
    for _, row in latest.iterrows():
        feature_period = str(row["month"])
        target_period = str(row["month"] + 1)
        rows.append(_state_row(
            "ndc_monthly_shortage_pressure_state", target_period,
            int(row["pressure_state"]), geography_level="national_ndc",
            geography_id="AR-exposed", drug_key=str(row["ndc9"]),
            horizon=30,
            feature_period=feature_period, source="FDA shortage archive",
            attribution={
                "active_supplier_count": int(row["active_supplier_count"]),
                "observed_supplier_count": int(row["observed_supplier_count"]),
                "state_definition": {
                    "0": "none_observed", "1": "single_supplier",
                    "2": "two_suppliers", "3": "three_suppliers",
                    "4": "four_or_more_suppliers",
                },
            }))
    return pd.DataFrame(rows)


def forecast_latest_numeric_shortage_count(source: pd.DataFrame) -> pd.DataFrame:
    """Emit next-month FDA shortage supplier counts as a numeric candidate."""
    panel = build_pressure_panel(source)
    latest_month = panel["month"].max()
    latest = panel[panel["month"].eq(latest_month)]
    rows = [_numeric_row(
        "ndc_monthly_shortage_supplier_count", str(row["month"] + 1),
        float(row["active_supplier_count"]), geography_level="national_ndc",
        geography_id="AR-exposed", drug_key=str(row["ndc9"]), horizon=30,
        feature_period=str(row["month"]), source="FDA shortage archive",
        attribution={"active_supplier_count": int(row["active_supplier_count"]),
                     "observed_supplier_count": int(row["observed_supplier_count"])})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def forecast_latest_nadac_price(source: pd.DataFrame) -> pd.DataFrame:
    """Emit one-week persistence forecasts for the latest NADAC NDC rows."""
    required = {"ndc", "as_of_date", "nadac_per_unit"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"NADAC source missing columns: {sorted(missing)}")
    frame = source.copy()
    frame["ndc"] = frame["ndc"].map(_ndc9)
    frame["date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["nadac_per_unit"], errors="coerce")
    frame = frame[frame["ndc"].ne("") & frame["date"].notna() & frame["price"].gt(0)]
    frame = frame.groupby(["ndc", "date"], as_index=False)["price"].mean()
    latest_date = frame["date"].max()
    latest = frame[frame["date"].eq(latest_date)]
    rows = [_numeric_row(
        "nadac_next_observed_price", (row["date"] + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        float(row["price"]), geography_level="national_ndc", geography_id="US",
        drug_key=str(row["ndc"]), horizon=7, feature_period=str(row["date"].date()),
        source="CMS NADAC", attribution={"nadac_current": float(row["price"]),
                                          "forecast_method": "persistence_selected_by_validation"})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def forecast_latest_fluview(panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-week Arkansas respiratory state from latest WILI state."""
    view = load_fluview_weekly_proxy(panel) if "target" not in panel.columns else panel.copy()
    latest = view[view["region"].eq("ar")].sort_values("week_start").tail(1)
    if latest.empty:
        return pd.DataFrame()
    row = latest.iloc[0]
    feature_period = pd.Timestamp(row["week_start"]).strftime("%G-W%V")
    target_period = (pd.Timestamp(row["week_start"]) + pd.Timedelta(days=7)).strftime("%G-W%V")
    # The production adapter uses the validated persistence method. Training
    # and validation artifacts can replace this row with a learned head later.
    history = view[view["week_start"] < row["week_start"]]["current_value"]
    if len(history) >= 3:
        lower, upper = history.quantile([1 / 3, 2 / 3]).to_numpy()
        prediction = int(pd.cut([row["current_value"]],
                                bins=[-float("inf"), lower, upper, float("inf")],
                                labels=False)[0])
    else:
        prediction = 1
    return pd.DataFrame([_state_row(
        "arkansas_weekly_fluview_respiratory_pressure_state", target_period,
        prediction, geography_level="state", geography_id="AR",
        horizon=7,
        feature_period=feature_period, source="CDC FluView",
        attribution={"wili_current": float(row["current_value"]),
                     "state_definition": {"0": "low", "1": "mid", "2": "high"}})])


def forecast_latest_national_fluview_five_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-week national FluView WILI using five fit-history states."""
    view = load_fluview_weekly_proxy(panel) if "target" not in panel.columns else panel.copy()
    latest = view[view["region"].eq("nat")].sort_values("week_start").tail(1)
    if latest.empty:
        return pd.DataFrame()
    row = latest.iloc[0]
    history = view[(view["region"].eq("nat"))
                   & (view["week_start"] < row["week_start"])]
    if len(history) < 5:
        return pd.DataFrame()
    thresholds = history["current_value"].quantile(np.arange(1, 5) / 5).to_numpy()
    prediction = int(np.digitize(float(row["current_value"]), thresholds))
    feature_period = pd.Timestamp(row["week_start"]).strftime("%G-W%V")
    target_period = (pd.Timestamp(row["week_start"]) + pd.Timedelta(days=7)).strftime("%G-W%V")
    return pd.DataFrame([_state_row(
        "national_weekly_fluview_respiratory_pressure_state", target_period,
        prediction, geography_level="national", geography_id="US", horizon=7,
        feature_period=feature_period, source="CDC FluView",
        attribution={
            "wili_current": float(row["current_value"]),
            "thresholds_fit_history": [float(value) for value in thresholds],
            "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
        })])


def forecast_latest_respnet_rsv_five_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Emit the next-week national RESP-NET RSV five-state proxy."""
    view = load_respnet_rsv_weekly(panel) if "target" not in panel.columns else panel.copy()
    if view.empty:
        return pd.DataFrame()
    latest = view.sort_values("week_end").tail(1).iloc[0]
    history = view[view["week_end"] < latest["week_end"]]
    if len(history) < 5:
        return pd.DataFrame()
    thresholds = history["target"].quantile(np.arange(1, 5) / 5).to_numpy()
    prediction = int(np.digitize(float(latest["current_value"]), thresholds))
    feature_period = pd.Timestamp(latest["week_end"]).strftime("%Y-%m-%d")
    target_period = (pd.Timestamp(latest["week_end"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return pd.DataFrame([_state_row(
        "national_weekly_respnet_rsv_hospitalization_pressure_state", target_period,
        prediction, geography_level="national", geography_id="US", pathogen="rsv",
        horizon=7, feature_period=feature_period, source="CDC RESP-NET",
        attribution={
            "rsv_rate_per_100k_current": float(latest["current_value"]),
            "thresholds_fit_history": [float(value) for value in thresholds],
            "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
        })])


def forecast_latest_apcd_claim_activity(panel: pd.DataFrame) -> pd.DataFrame:
    """Emit next-month APCD reporting-entity claim-activity states."""
    view = load_apcd_claim_counts(panel) if "next_period" not in panel.columns else panel.copy()
    if view.empty:
        return pd.DataFrame()
    latest_period = view["period"].max()
    latest = view[view["period"].eq(latest_period)]
    history = view[view["period"].lt(latest_period)]
    if latest.empty or len(history) < 5:
        return pd.DataFrame()
    thresholds = history["target"].quantile(np.arange(1, 5) / 5).to_numpy()
    rows = []
    for _, row in latest.iterrows():
        rows.append(_state_row(
            "arkansas_monthly_apcd_pharmacy_claim_activity_state",
            str(latest_period + 1),
            int(np.digitize(float(row["current_value"]), thresholds)),
            geography_level="apcd_submitter", geography_id=str(row["submitter_id"]),
            horizon=30, feature_period=str(latest_period), source=APCD_SOURCE_URL,
            attribution={
                "submitter_name": str(row["submitter_name"]),
                "claim_count_current": float(row["current_value"]),
                "thresholds_fit_history": [float(value) for value in thresholds],
                "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
                "scope": "reporting-entity claim activity, not NDC demand or inventory",
            }))
    return pd.DataFrame(rows)


def forecast_latest_therapeutic_class_demand(
    demand: pd.DataFrame, mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Emit next-month ATC therapeutic-group demand states."""
    view = build_latest_therapeutic_class_state(demand, mapping)
    if view.empty:
        return pd.DataFrame()
    rows = []
    for _, row in view.iterrows():
        rows.append(_state_row(
            "arkansas_monthly_atc_therapeutic_demand_state",
            str(row["forecast_period"]), int(row["prediction"]),
            geography_level="state", geography_id="AR",
            therapeutic_class=str(row["therapeutic_class"]), horizon=30,
            feature_period=str(row["feature_period"]), source="HHS + NLM RxNorm/RxClass",
            attribution={
                "therapeutic_class": str(row["therapeutic_class"]),
                "class_name": str(row["class_name"]),
                "current_claim_lines": float(row["current_claim_lines"]),
                "thresholds_fit_history": row["thresholds"],
                "state_definition": {str(i): f"fit_quantile_{i + 1}_of_5" for i in range(5)},
            }))
    return pd.DataFrame(rows)


def forecast_latest_nssp_influenza_five_state(nssp_panel: pd.DataFrame) -> pd.DataFrame:
    """Emit the qualified next-week Arkansas NSSP influenza state."""
    view = build_nssp_weekly_view(nssp_panel, pathogen="influenza")
    if view.empty:
        return pd.DataFrame()
    latest_week = view["week"].max()
    latest = view[view["week"].eq(latest_week)]
    history = view[view["week"].lt(latest_week)]
    if latest.empty or history.empty:
        return pd.DataFrame()
    thresholds = np.quantile(history["target"], np.arange(1, 5) / 5)
    row = latest.iloc[-1]
    return pd.DataFrame([_state_row(
        "arkansas_weekly_nssp_influenza_ed_pressure_state", str(latest_week + 1),
        int(np.digitize(float(row["current_value"]), thresholds)),
        geography_level="arkansas_state", geography_id="AR", pathogen="influenza",
        horizon=7, feature_period=str(latest_week), source="CDC NSSP ED trajectories",
        attribution={
            "ed_visit_percentage_current": float(row["current_value"]),
            "thresholds_fit_history": [float(value) for value in thresholds],
            "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
        })])


def forecast_latest_regional_wastewater_five_state(
    panel: pd.DataFrame, *, pathogen: str,
) -> pd.DataFrame:
    """Emit current, filterable regional wastewater candidates.

    Only regions with an observation in the latest global source week are
    emitted; stale regional series are not silently forecast forward.
    """
    view = build_regional_wastewater_weekly_view(panel, pathogen=pathogen)
    if view.empty:
        return pd.DataFrame()
    latest_week = view["week"].max()
    latest = view[view["week"].eq(latest_week)].copy()
    target_suffix = {"Influenza A virus": "influenza", "SARS-CoV-2": "sars_cov_2",
                     "RSV": "rsv"}[pathogen]
    rows = []
    for _, row in latest.iterrows():
        history = view[(view["region"].eq(row["region"]))
                       & view["week"].lt(latest_week)]
        if len(history) < 5:
            continue
        thresholds = np.quantile(history["current_value"], np.arange(1, 5) / 5)
        rows.append(_state_row(
            f"arkansas_region_weekly_wastewater_{target_suffix}_pressure_state",
            str(latest_week + 1), int(np.digitize(float(row["current_value"]), thresholds)),
            geography_level="arkansas_region", geography_id=str(row["region"]),
            pathogen=target_suffix, horizon=7, feature_period=str(latest_week),
            source="CDC wastewater WVAL + Census county naming + Arkansas region crosswalk",
            attribution={"wastewater_current": float(row["current_value"]),
                         "site_count": int(row["site_count"]),
                         "thresholds_fit_history": [float(value) for value in thresholds],
                         "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)}}))
    return pd.DataFrame(rows)


def forecast_latest_numeric_fluview(panel: pd.DataFrame) -> pd.DataFrame:
    """Emit next-week Arkansas WILI as a numeric candidate."""
    view = load_fluview_weekly_proxy(panel) if "target" not in panel.columns else panel.copy()
    latest = view[view["region"].eq("ar")].sort_values("week_start").tail(1)
    if latest.empty:
        return pd.DataFrame()
    row = latest.iloc[0]
    feature_period = pd.Timestamp(row["week_start"]).strftime("%G-W%V")
    target_period = (pd.Timestamp(row["week_start"]) + pd.Timedelta(days=7)).strftime("%G-W%V")
    return pd.DataFrame([_numeric_row(
        "arkansas_weekly_fluview_wili", target_period, float(row["current_value"]),
        geography_level="state", geography_id="AR", horizon=7,
        feature_period=feature_period, source="CDC FluView",
        attribution={"wili_current": float(row["current_value"])})])


def forecast_latest_hospital_respiratory(
    panel: pd.DataFrame, *, pathogen: str = "influenza"
) -> pd.DataFrame:
    """Forecast next-week Arkansas hospital admission pressure."""
    if pathogen not in TARGET_COLUMNS:
        raise ValueError(f"unsupported hospital pathogen: {pathogen}")
    frame = panel[panel["jurisdiction"].astype(str).str.upper().eq("AR")].copy()
    frame["week_end"] = pd.to_datetime(frame["weekendingdate"], errors="coerce")
    frame["value"] = pd.to_numeric(frame[TARGET_COLUMNS[pathogen]], errors="coerce")
    frame = frame.dropna(subset=["week_end", "value"]).sort_values("week_end")
    if frame.empty:
        return pd.DataFrame()
    row = frame.iloc[-1]
    history = frame.iloc[:-1]["value"]
    lower, upper = history.quantile([1 / 3, 2 / 3]).to_numpy() if len(history) >= 3 else (0.0, 1.0)
    prediction = int(pd.cut(
        [row["value"]], bins=[-float("inf"), lower, upper, float("inf")],
        labels=False,
    )[0])
    feature_period = pd.Timestamp(row["week_end"]).strftime("%Y-%m-%d")
    target_period = (pd.Timestamp(row["week_end"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return pd.DataFrame([_state_row(
        f"arkansas_weekly_hospital_{pathogen}_admission_pressure_state", target_period,
        prediction, geography_level="state", geography_id="AR",
        horizon=7,
        feature_period=feature_period, source="CDC NHSN Hospital Respiratory Data",
        attribution={"pathogen": pathogen, "current_admissions": float(row["value"]),
                     "state_definition": {"0": "low", "1": "mid", "2": "high"}})])


def forecast_latest_numeric_hospital_respiratory(
    panel: pd.DataFrame, *, pathogen: str = "influenza"
) -> pd.DataFrame:
    """Emit next-week Arkansas hospital admissions as a numeric candidate."""
    if pathogen not in TARGET_COLUMNS:
        raise ValueError(f"unsupported hospital pathogen: {pathogen}")
    frame = panel[panel["jurisdiction"].astype(str).str.upper().eq("AR")].copy()
    frame["week_end"] = pd.to_datetime(frame["weekendingdate"], errors="coerce")
    frame["value"] = pd.to_numeric(frame[TARGET_COLUMNS[pathogen]], errors="coerce")
    frame = frame.dropna(subset=["week_end", "value"]).sort_values("week_end")
    if frame.empty:
        return pd.DataFrame()
    row = frame.iloc[-1]
    feature_period = pd.Timestamp(row["week_end"]).strftime("%Y-%m-%d")
    target_period = (pd.Timestamp(row["week_end"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return pd.DataFrame([_numeric_row(
        "arkansas_weekly_hospital_influenza_admissions", target_period,
        float(row["value"]), geography_level="state", geography_id="AR", horizon=7,
        feature_period=feature_period, source="CDC NHSN Hospital Respiratory Data",
        attribution={"pathogen": pathogen, "current_admissions": float(row["value"])})])


def forecast_latest_overdose_pressure(source: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-month overdose-pressure state for latest observed counties."""
    view = build_overdose_view(source, enforce_vintage=False)
    observed = build_overdose_observations(source)
    if view.empty or observed.empty:
        return pd.DataFrame()
    latest_period = observed["period"].max()
    latest = observed[observed["period"].eq(latest_period)].copy()
    history = observed[observed["period"].lt(latest_period)]
    if latest.empty or history.empty:
        return pd.DataFrame()
    lower, upper = history["provisional_drug_overdose"].quantile([1 / 3, 2 / 3]).to_numpy()
    rows = []
    for _, row in latest.iterrows():
        prediction = int(pd.cut(
            [float(row["provisional_drug_overdose"])],
            bins=[-float("inf"), lower, upper, float("inf")], labels=False,
        )[0])
        rows.append(_state_row(
            "arkansas_county_monthly_overdose_pressure_state",
            str(latest_period + 1), prediction,
            geography_level="county", geography_id=str(row["county_fips"]),
            county_fips=str(row["county_fips"]),
            horizon=30,
            feature_period=str(latest_period), source="CDC VSRR county overdose deaths",
            attribution={
                "county_fips": str(row["county_fips"]),
                "current_rolling_12_month_count": float(row["provisional_drug_overdose"]),
                "thresholds": [float(lower), float(upper)],
                "state_definition": {"0": "low", "1": "mid", "2": "high"},
            }))
    return pd.DataFrame(rows)


def forecast_latest_regional_demand(county_demand: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-year low/mid/high demand states for each region and drug."""
    view = build_regional_demand_view(county_demand)
    # Do not emit stale region/drug pairs whose latest annual observation is
    # older than the newest complete CMS year in the source.
    latest_year = view["year"].max()
    latest = view[view["year"].eq(latest_year)].copy()
    rows = []
    thresholds = view["demand_claims"].quantile([1 / 3, 2 / 3]).to_numpy()
    for _, row in latest.iterrows():
        region = str(row["arkansas_region"])
        feature_year = int(row["year"])
        rows.append(_state_row(
            "arkansas_region_annual_demand_state", str(feature_year + 1),
            int(pd.cut([row["demand_claims"]], bins=[-float("inf"), thresholds[0],
                                                        thresholds[1], float("inf")],
                       labels=False)[0]), geography_level="arkansas_region",
            geography_id=region, drug_key=str(row["drug_key"]),
            horizon=365,
            feature_period=str(feature_year), source="CMS Part D county demand",
            attribution={"demand_claims_current": float(row["demand_claims"]),
                         "state_definition": {"0": "low", "1": "mid", "2": "high"}}))
    return pd.DataFrame(rows)


def forecast_latest_regional_demand_five_state(county_demand: pd.DataFrame) -> pd.DataFrame:
    """Emit the validated five-quantile annual regional demand proxy."""
    view = build_regional_demand_view(county_demand)
    latest_year = view["year"].max()
    latest = view[view["year"].eq(latest_year)].copy()
    thresholds = tuple(np.quantile(view["demand_claims"], np.arange(1, 5) / 5))
    rows = []
    for _, row in latest.iterrows():
        prediction = int(np.digitize(float(row["demand_claims"]), thresholds))
        rows.append(_state_row(
            "arkansas_region_annual_demand_five_state", str(int(row["year"]) + 1),
            prediction, geography_level="arkansas_region",
            geography_id=str(row["arkansas_region"]), drug_key=str(row["drug_key"]),
            horizon=365, feature_period=str(row["year"]),
            source="CMS Part D county demand",
            attribution={
                "demand_claims_current": float(row["demand_claims"]),
                "thresholds_fit_history": [float(value) for value in thresholds],
                "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
            }))
    return pd.DataFrame(rows)


def forecast_latest_state_partd_demand_five_state(state_demand: pd.DataFrame) -> pd.DataFrame:
    """Emit next-year Arkansas statewide Part D demand states by generic drug."""
    view = build_state_demand_view(state_demand)
    latest_year = int(view["year"].max())
    latest = view[view["year"].eq(latest_year)].copy()
    history = view[view["year"].lt(latest_year)]
    if latest.empty or history.empty:
        return pd.DataFrame()
    thresholds = np.quantile(history["demand_claims"], np.arange(1, 5) / 5)
    rows = []
    for _, row in latest.iterrows():
        rows.append(_state_row(
            "arkansas_state_annual_partd_demand_five_state", str(latest_year + 1),
            int(np.digitize(float(row["demand_claims"]), thresholds)),
            geography_level="state", geography_id="AR", drug_key=str(row["drug_key"]),
            horizon=365, feature_period=str(latest_year),
            source="CMS Part D geography and drug",
            attribution={
                "demand_claims_current": float(row["demand_claims"]),
                "thresholds_fit_history": [float(value) for value in thresholds],
                "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
            }))
    return pd.DataFrame(rows)


def forecast_latest_county_demand_five_state(county_demand: pd.DataFrame) -> pd.DataFrame:
    """Emit next-year five-quantile demand states by Arkansas county and drug."""
    required = {"year", "county_fips", "drug_key", "demand_claims"}
    if not required.issubset(county_demand.columns):
        return pd.DataFrame()
    frame = county_demand.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["demand_claims"] = pd.to_numeric(frame["demand_claims"], errors="coerce")
    frame["county_fips"] = frame["county_fips"].astype(str)
    frame["drug_key"] = frame["drug_key"].astype(str)
    frame = frame.dropna(subset=["year", "demand_claims"])
    latest_year = int(frame["year"].max())
    latest = frame[frame["year"].eq(latest_year)]
    history = frame[frame["year"].lt(latest_year)]
    if latest.empty or history.empty:
        return pd.DataFrame()
    thresholds = np.quantile(history["demand_claims"], np.arange(1, 5) / 5)
    rows = []
    for _, row in latest.iterrows():
        rows.append(_state_row(
            "arkansas_county_annual_demand_five_state", str(latest_year + 1),
            int(np.digitize(float(row["demand_claims"]), thresholds)),
            geography_level="county", geography_id=str(row["county_fips"]),
            county_fips=str(row["county_fips"]), drug_key=str(row["drug_key"]),
            horizon=365, feature_period=str(latest_year),
            source="CMS Part D county demand",
            attribution={
                "demand_claims_current": float(row["demand_claims"]),
                "thresholds_fit_history": [float(value) for value in thresholds],
                "state_definition": {str(i): f"quantile_{i + 1}_of_5" for i in range(5)},
            }))
    return pd.DataFrame(rows)


def forecast_latest_numeric_regional_demand(county_demand: pd.DataFrame) -> pd.DataFrame:
    """Emit next-year regional CMS claims as a numeric candidate."""
    view = build_regional_demand_view(county_demand)
    latest_year = view["year"].max()
    latest = view[view["year"].eq(latest_year)]
    rows = [_numeric_row(
        "arkansas_region_annual_demand_claims", str(int(row["year"]) + 1),
        float(row["demand_claims"]), geography_level="arkansas_region",
        geography_id=str(row["arkansas_region"]), drug_key=str(row["drug_key"]),
        horizon=365, feature_period=str(row["year"]), source="CMS Part D county demand",
        attribution={"demand_claims_current": float(row["demand_claims"])})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def forecast_latest_recall_pressure(events: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-month recall severity with validation-selected fallback."""
    panel = build_recall_panel(events, include_source_edge=True)
    latest, selection = _source_edge_recall_predictions(
        panel, group_columns=("ndc",))
    latest_month = panel["month"].max()
    rows = []
    for _, row in latest.iterrows():
        rows.append(_state_row(
            "ndc_monthly_recall_pressure_state", str(latest_month + 1),
            int(row["forecast_state"]), geography_level="national_ndc",
            geography_id="US", drug_key=str(row["ndc"]),
            horizon=30,
            feature_period=str(latest_month), source="FDA drug enforcement recalls",
            attribution={**selection, "state_definition": {
                "0": "no_active_recall", "1": "active_class_II_or_III",
                "2": "active_class_I"}}))
    return pd.DataFrame(rows)


def forecast_latest_supplier_recall_pressure(events: pd.DataFrame) -> pd.DataFrame:
    """Forecast supplier-by-NDC recall severity with validation-selected fallback."""
    panel = build_supplier_recall_panel(events, include_source_edge=True)
    latest, selection = _source_edge_recall_predictions(
        panel, group_columns=("supplier", "ndc"))
    latest_month = panel["month"].max()
    rows = []
    for _, row in latest.iterrows():
        rows.append(_state_row(
            "supplier_ndc_monthly_recall_pressure_state", str(latest_month + 1),
            int(row["forecast_state"]), geography_level="supplier_drug",
            geography_id=str(row["supplier"]), drug_key=str(row["ndc"]),
            supplier=str(row["supplier"]), feature_period=str(latest_month),
            horizon=30,
            source="FDA drug enforcement recalls",
            attribution={**selection, "state_definition": {
                "0": "no_active_recall", "1": "active_class_II_or_III",
                "2": "active_class_I"}}))
    return pd.DataFrame(rows)


def forecast_latest_numeric_recall_pressure(events: pd.DataFrame) -> pd.DataFrame:
    """Emit next-month NDC recall severity from the selected state model."""
    panel = build_recall_panel(events, include_source_edge=True)
    latest, selection = _source_edge_recall_predictions(
        panel, group_columns=("ndc",))
    latest_month = panel["month"].max()
    rows = [_numeric_row(
        "ndc_monthly_recall_severity", str(latest_month + 1),
        float(row["forecast_state"]), geography_level="national_ndc", geography_id="US",
        drug_key=str(row["ndc"]), horizon=30, feature_period=str(latest_month),
        source="FDA drug enforcement recalls",
        attribution={**selection, "recall_severity_current": int(row["recall_state"]),
                     "numeric_scale": {"0": "none", "1": "Class II/III", "2": "Class I"}})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def forecast_latest_numeric_supplier_recall_pressure(events: pd.DataFrame) -> pd.DataFrame:
    """Emit next-month supplier-by-NDC recall severity from the selected model."""
    panel = build_supplier_recall_panel(events, include_source_edge=True)
    latest, selection = _source_edge_recall_predictions(
        panel, group_columns=("supplier", "ndc"))
    latest_month = panel["month"].max()
    rows = [_numeric_row(
        "supplier_ndc_monthly_recall_severity", str(latest_month + 1),
        float(row["forecast_state"]), geography_level="supplier_drug",
        geography_id=str(row["supplier"]), drug_key=str(row["ndc"]),
        supplier=str(row["supplier"]), horizon=30, feature_period=str(latest_month),
        source="FDA drug enforcement recalls",
        attribution={**selection, "recall_severity_current": int(row["recall_state"]),
                     "numeric_scale": {"0": "none", "1": "Class II/III", "2": "Class I"}})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def forecast_latest_arcos_distribution_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast next-quarter ARCOS five-quantile state by ZIP3 and drug."""
    if panel.empty:
        return pd.DataFrame()
    latest = build_latest_scoring_view(panel)
    latest_period = int(panel["period_index"].max())
    latest = latest[latest["period_index"].eq(latest_period)].copy()
    history = panel[panel["period_index"].lt(latest_period)]
    if latest.empty or history.empty:
        return pd.DataFrame()
    thresholds = history["log_grams"].quantile(np.arange(1, 5) / 5).to_numpy()
    if len(set(thresholds)) < 2:
        return pd.DataFrame()
    rows = []
    for _, row in latest.iterrows():
        state = int(np.digitize(float(row["log_grams"]), thresholds))
        next_index = int(row["period_index"]) + 1
        rows.append(_state_row(
            "arcos_zip3_drug_next_quarter_distribution_state",
            f"{next_index // 4:04d}-Q{next_index % 4 + 1}", state,
            geography_level="zip3", geography_id=str(row["zip3"]),
            drug_key=str(row["drug_code"]),
            horizon=91,
            feature_period=str(row["period"]), source="DEA ARCOS Report 01",
            attribution={
                "current_distribution_grams": float(row["grams"]),
                "thresholds_log1p_fit_history": [float(value) for value in thresholds],
                "state_definition": {str(i): f"quantile_{i + 1}_of_5"
                                      for i in range(5)},
            }))
    return pd.DataFrame(rows)


def forecast_latest_numeric_arcos_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    """Emit next-quarter ARCOS grams as a numeric candidate."""
    if panel.empty:
        return pd.DataFrame()
    latest_period = int(panel["period_index"].max())
    latest = panel[panel["period_index"].eq(latest_period)]
    next_period = latest_period + 1
    rows = [_numeric_row(
        "arcos_zip3_drug_next_quarter_distribution_grams",
        f"{next_period // 4:04d}-Q{next_period % 4 + 1}",
        float(row["grams"]), geography_level="zip3", geography_id=str(row["zip3"]),
        drug_key=str(row["drug_code"]), horizon=91, feature_period=str(row["period"]),
        source="DEA ARCOS Report 01",
        attribution={"distribution_grams_current": float(row["grams"])})
            for _, row in latest.iterrows()]
    return pd.DataFrame(rows)


def build_qualified_metric_forecasts(
    *, shortage_source: pd.DataFrame, flu_panel: pd.DataFrame,
    county_demand: pd.DataFrame, recall_events: pd.DataFrame | None = None,
    hospital_panel: pd.DataFrame | None = None,
    nssp_panel: pd.DataFrame | None = None,
    respnet_panel: pd.DataFrame | None = None,
    apcd_panel: pd.DataFrame | None = None,
    therapeutic_demand: pd.DataFrame | None = None,
    therapeutic_mapping: pd.DataFrame | None = None,
    arcos_panel: pd.DataFrame | None = None,
    nadac_panel: pd.DataFrame | None = None,
    state_demand: pd.DataFrame | None = None,
    model_version: str = "qualified-proxies-v1",
    forecast_timestamp: str | None = None,
) -> pd.DataFrame:
    """Build the combined filterable forecast surface for qualified proxies."""
    raw = pd.concat([
        forecast_latest_shortage_pressure(shortage_source),
        forecast_latest_numeric_shortage_count(shortage_source),
        forecast_latest_nadac_price(nadac_panel) if nadac_panel is not None else pd.DataFrame(),
        forecast_latest_regional_demand_five_state(county_demand),
        forecast_latest_state_partd_demand_five_state(state_demand)
        if state_demand is not None else pd.DataFrame(),
        forecast_latest_county_demand_five_state(county_demand),
        forecast_latest_national_fluview_five_state(flu_panel),
        forecast_latest_respnet_rsv_five_state(respnet_panel)
        if respnet_panel is not None else pd.DataFrame(),
        forecast_latest_apcd_claim_activity(apcd_panel)
        if apcd_panel is not None else pd.DataFrame(),
        forecast_latest_therapeutic_class_demand(therapeutic_demand, therapeutic_mapping)
        if therapeutic_demand is not None and therapeutic_mapping is not None
        else pd.DataFrame(),
        forecast_latest_nssp_influenza_five_state(nssp_panel)
        if nssp_panel is not None else pd.DataFrame(),
        forecast_latest_arcos_distribution_state(arcos_panel)
        if arcos_panel is not None else pd.DataFrame(),
        forecast_latest_numeric_recall_pressure(recall_events)
        if recall_events is not None else pd.DataFrame(),
        forecast_latest_numeric_supplier_recall_pressure(recall_events)
        if recall_events is not None else pd.DataFrame(),
    ], ignore_index=True, sort=False)
    return build_metric_rows(raw, model_version=model_version,
                             forecast_timestamp=forecast_timestamp)


def build_numeric_metric_candidates(
    *, shortage_source: pd.DataFrame, flu_panel: pd.DataFrame,
    county_demand: pd.DataFrame, recall_events: pd.DataFrame | None = None,
    hospital_panel: pd.DataFrame | None = None,
    arcos_panel: pd.DataFrame | None = None,
    nadac_panel: pd.DataFrame | None = None,
    model_version: str = "numeric-candidates-v1",
    forecast_timestamp: str | None = None,
) -> pd.DataFrame:
    """Build numeric candidates without treating them as promoted metrics."""
    raw = pd.concat([
        forecast_latest_numeric_shortage_count(shortage_source),
        forecast_latest_nadac_price(nadac_panel) if nadac_panel is not None else pd.DataFrame(),
        forecast_latest_numeric_fluview(flu_panel),
        forecast_latest_numeric_hospital_respiratory(hospital_panel)
        if hospital_panel is not None else pd.DataFrame(),
        forecast_latest_numeric_regional_demand(county_demand),
        forecast_latest_numeric_recall_pressure(recall_events)
        if recall_events is not None else pd.DataFrame(),
        forecast_latest_numeric_supplier_recall_pressure(recall_events)
        if recall_events is not None else pd.DataFrame(),
        forecast_latest_numeric_arcos_distribution(arcos_panel)
        if arcos_panel is not None else pd.DataFrame(),
    ], ignore_index=True, sort=False)
    return build_metric_rows(raw, model_version=model_version,
                             forecast_timestamp=forecast_timestamp, validate=False)
