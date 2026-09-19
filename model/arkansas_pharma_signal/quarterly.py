"""Arkansas Medicaid SDUD quarterly evaluation path.

Loads real Arkansas Medicaid SDUD prescription_count records from
``data/S_D/data/combined/*.json.gz`` (only records where
``source.source_id == "medicaid_sdud"``, ``geography.admin1 == "AR"``,
``observation.metric == "prescription_count"``, and ``observation.value`` is
present) and aggregates them by year, quarter, canonical drug.

A strict next-quarter view aligns features at quarter t with the real
prescription_count at quarter t+1 for the same drug, so no same-quarter
target leakage is possible. Baselines and a history ridge (with a
validation-selected calibrated blend) are scored with the task-standard split
train (feature years <= 2020), validation (2021), test (>= 2022). Writes
model/artifacts/evaluation/quarterly_leaderboard.csv and
quarterly_metrics.json.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .entities import normalize_name
from .cms_partd_quarterly import DEFAULT_PATH as CMS_PARTD_DEFAULT_PATH, add_latest_context_layer
from .evaluate import _fit_ridge_log, _ridge_log_predict, residual_to_raw
from .forecast import OUTPUT_COLUMNS
from .regression import RidgeLinear, _fill_nan

SOURCE_ID = "medicaid_sdud"
METRIC = "prescription_count"
FIRST_YEAR = 2012

FEATURE_COLS = ["prev_q_log", "lag2_log", "ma2_log", "q1", "q2", "q3", "q4"]
ANNUAL_LAYER_PREFIXES = (
    "ar_", "nat_", "ww_", "nndss_", "recall_", "shortage_", "labeler_",
    "event_", "news_", "na_", "global_", "us_",
)
ANNUAL_DRUG_COLS = (
    "demand_claims", "demand_fills", "demand_cost", "demand_benes",
    "n_provider_types", "cost_per_fill", "product_count",
    "ingredient_products", "labeler_products",
)

NAIVE_BASELINES = ["previous_quarter", "ma2", "same_target_quarter_last_year",
                   "croston_sba", "drug_mean", "global_mean"]
MODEL_NAMES = ["ridge_history", "calibrated_ridge_blend"]

PUBLISHABLE_THRESHOLD = 0.10
REQUESTED_ACCURACY_THRESHOLD = 0.75
QUARTER_DAYS = 91
# A categorical diagnostic must describe a material change, not turn ordinary
# count noise into a false signal.  This band is fixed before evaluation and
# is not selected from test labels.
DEMAND_STATE_BAND = 0.20

DISEASE_SURVEILLANCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data" / "targeted_additions" / "disease_surveillance_current" / "data"
)
FLUVIEW_DEFAULT = DISEASE_SURVEILLANCE_DIR / "cdc_fluview_ar_national_weekly.csv.gz"
WASTEWATER_DEFAULT = DISEASE_SURVEILLANCE_DIR / "cdc_wastewater_ar_site_weekly.csv.gz"
NADAC_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "data" / "demand_price" / "data" / "nadac_ndc_weekly.csv.gz"
)

FLUVIEW_NUMERIC = (
    "num_ili", "num_patients", "num_providers", "num_age_0", "num_age_1",
    "num_age_2", "num_age_3", "num_age_4", "num_age_5", "wili", "ili",
)
WASTEWATER_PATHOGENS = {
    "SARS-CoV-2": "sars2", "Influenza A virus": "flu_a", "RSV": "rsv",
}


def iter_medicaid_records(combined_dir: Path) -> Iterable[Dict]:
    """Yield every medicaid_sdud AR prescription_count record across files."""
    paths = sorted(Path(combined_dir).glob("20*.json.gz"))
    for p in paths:
        with gzip.open(p, "rt") as fh:
            recs = json.load(fh)
        for r in recs:
            if r["source"]["source_id"] != SOURCE_ID:
                continue
            if r["geography"].get("admin1") != "AR":
                continue
            if r["observation"]["metric"] != METRIC:
                continue
            yield r


def _mode_text(values: pd.Series) -> str:
    vals = values.dropna().astype(str)
    vals = vals[vals != ""]
    if vals.empty:
        return ""
    return vals.value_counts().index[0]


def _normalize_ndc_identifier(value: object) -> str:
    """Preserve an SDUD package NDC as an 11-digit identifier.

    SDUD records sometimes encode the numeric NDC without leading zeroes;
    normalizing before CSV serialization is required for exact FDA joins.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(11) if digits else ""


def load_medicaid_sdud_panel(combined_dir: Path) -> pd.DataFrame:
    """Aggregate records to (year, quarter, canonical drug, value).

    The quarterly target remains drug-level Medicaid prescriptions. Public
    identifiers found in the raw records (NDC, RxCUI, ingredient, manufacturer)
    are retained as modal descriptors per drug-quarter for downstream real-data
    joins and auditability; they are never used as target labels.
    """
    rows = []
    for r in iter_medicaid_records(combined_dir):
        value = r["observation"].get("value")
        if value is None:
            continue
        drug = r["drug"].get("canonical_name")
        if not drug:
            continue
        period_start = r.get("period_start")
        year = int(period_start[:4])
        quarter = (int(period_start[5:7]) - 1) // 3 + 1
        d = r.get("drug", {})
        rows.append({
            "year": year,
            "quarter": quarter,
            "drug": drug,
            "value": float(value),
            "ingredient": d.get("ingredient") or "",
            "ndc": _normalize_ndc_identifier(d.get("ndc")),
            "rxnorm_rxcui": d.get("rxnorm_rxcui") or "",
            "manufacturer": d.get("manufacturer") or "",
            "mapping_confidence": d.get("mapping_confidence") or "",
        })
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError(
            f"no Arkansas medicaid_sdud prescription_count records under "
            f"{combined_dir}")
    agg = {
        "value": "sum",
        "ingredient": _mode_text,
        "ndc": _mode_text,
        "rxnorm_rxcui": _mode_text,
        "manufacturer": _mode_text,
        "mapping_confidence": _mode_text,
    }
    panel = panel.groupby(["year", "quarter", "drug"], as_index=False).agg(agg)
    return panel.sort_values(["drug", "year", "quarter"]).reset_index(drop=True)


def build_quarterly_view(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature quarter t -> target quarter t+1 for the same drug.

    Each row carries the real prescription_count at t (``value``, the
    previous-quarter baseline) and at t-1 (``value_last``, the lag-2 feature),
    the two-quarter moving average, log1p transforms, and fixed quarter
    seasonality one-hot columns. The target is the real value at the
    immediately following calendar quarter. Drugs without an observed
    consecutive t+1 quarter are dropped; missing quarters also invalidate the
    lag feature. No feature references the target period, so the view is
    leak-free by construction.
    """
    panel = panel.copy()
    panel["q_id"] = (panel["year"] - FIRST_YEAR) * 4 + panel["quarter"] - 1
    panel = panel.sort_values(["drug", "q_id"]).reset_index(drop=True)
    grp = panel.groupby("drug", group_keys=False)
    previous_q_id = grp["q_id"].shift(1)
    next_q_id = grp["q_id"].shift(-1)
    # A missing quarter is not a valid one-quarter lag or target.  Treating
    # the next observed record as t+1 would silently turn this into a
    # multi-quarter forecast while labeling it next-quarter evidence.
    consecutive_previous = panel["q_id"].sub(previous_q_id).eq(1)
    consecutive_next = next_q_id.sub(panel["q_id"]).eq(1)
    panel["value_last"] = grp["value"].shift(1).where(consecutive_previous)
    panel["ma2"] = 0.5 * (panel["value"]
                          + panel["value_last"].fillna(panel["value"]))
    panel["prev_q_log"] = np.log1p(np.clip(panel["value"], 0, None))
    panel["lag2_log"] = np.log1p(
        np.clip(panel["value_last"].fillna(panel["value"]), 0, None))
    panel["ma2_log"] = np.log1p(np.clip(panel["ma2"], 0, None))
    for q in range(1, 5):
        panel[f"q{q}"] = (panel["quarter"] == q).astype(float)
    panel["target"] = grp["value"].shift(-1).where(consecutive_next)
    view = panel.dropna(subset=["value", "target"])
    return view.drop(columns=["q_id"]).reset_index(drop=True)


def build_quarterly_scoring_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Latest real feature quarter per drug for forward scoring."""
    panel = panel.copy()
    panel["q_id"] = (panel["year"] - FIRST_YEAR) * 4 + panel["quarter"] - 1
    panel = panel.sort_values(["drug", "q_id"]).reset_index(drop=True)
    grp = panel.groupby("drug", group_keys=False)
    previous_q_id = grp["q_id"].shift(1)
    panel["value_last"] = grp["value"].shift(1).where(
        panel["q_id"].sub(previous_q_id).eq(1))
    panel["ma2"] = 0.5 * (panel["value"]
                          + panel["value_last"].fillna(panel["value"]))
    panel["prev_q_log"] = np.log1p(np.clip(panel["value"], 0, None))
    panel["lag2_log"] = np.log1p(
        np.clip(panel["value_last"].fillna(panel["value"]), 0, None))
    panel["ma2_log"] = np.log1p(np.clip(panel["ma2"], 0, None))
    for q in range(1, 5):
        panel[f"q{q}"] = (panel["quarter"] == q).astype(float)
    latest = panel.groupby("drug", as_index=False).tail(1)
    return latest.drop(columns=["q_id"]).reset_index(drop=True)


def _safe_numeric_cols(df: pd.DataFrame, required: set) -> List[str]:
    cols = []
    for c in df.columns:
        if c in required:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _annual_feature_columns(cols: Iterable[str]) -> List[str]:
    out = []
    for c in cols:
        if c in ANNUAL_DRUG_COLS or c.startswith(ANNUAL_LAYER_PREFIXES):
            out.append(c)
    return out


def add_annual_panel_layers(view: pd.DataFrame, panel_path: Path) -> tuple:
    """Join previous-completed-year Part D/drug/supply/news layers.

    For feature quarter t in year Y, the join uses annual panel year Y-1. This
    intentionally gives up same-year information to avoid leaking annual demand
    or external summaries that would not be fully known at the forecast date.
    """
    path = Path(panel_path)
    if not path.exists():
        return view, []
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keep = ["year", "drug_key"] + _annual_feature_columns(header)
    keep = [c for c in keep if c in header]
    if len(keep) <= 2:
        return view, []
    annual = pd.read_csv(path, usecols=keep, low_memory=False)
    annual["drug_norm"] = annual["drug_key"].map(normalize_name)
    numeric = _safe_numeric_cols(annual, {"year"})
    drug_numeric = [c for c in numeric if c != "year"]
    drug = (annual.groupby(["year", "drug_norm"], as_index=False)[drug_numeric]
            .mean())
    drug = drug.rename(columns={c: f"exo_annual_drug_{c}" for c in drug_numeric})

    # State/national annual context is available for all Medicaid drugs,
    # including brand names that cannot be mapped to the annual Part D key.
    global_cols = [c for c in drug_numeric if c.startswith(ANNUAL_LAYER_PREFIXES)]
    global_year = (annual.groupby("year", as_index=False)[global_cols].mean()
                   if global_cols else pd.DataFrame({"year": []}))
    global_year = global_year.rename(
        columns={c: f"exo_annual_state_{c}" for c in global_cols})

    out = view.copy()
    out["drug_norm"] = out.get("ingredient", out["drug"]).fillna(out["drug"])
    out["drug_norm"] = out["drug_norm"].map(normalize_name)
    out["annual_feature_year"] = out["year"] - 1
    out = out.merge(
        drug,
        left_on=["annual_feature_year", "drug_norm"],
        right_on=["year", "drug_norm"],
        how="left",
        suffixes=("", "_annual_drug_join"),
    )
    out = out.drop(columns=[c for c in ["year_annual_drug_join"] if c in out.columns])
    if not global_year.empty:
        out = out.merge(
            global_year,
            left_on="annual_feature_year",
            right_on="year",
            how="left",
            suffixes=("", "_annual_state_join"),
        )
        out = out.drop(columns=[c for c in ["year_annual_state_join"] if c in out.columns])
    feature_cols = [c for c in out.columns if c.startswith("exo_annual_")]
    return out, feature_cols


def add_event_layers(view: pd.DataFrame, events_path: Path) -> tuple:
    """Join real event counts/severity by feature quarter only."""
    path = Path(events_path)
    if not path.exists():
        return view, []
    cols = ["event_type", "start_time", "severity", "confidence"]
    events = pd.read_csv(path, usecols=cols)
    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events = events.dropna(subset=["start_time"])
    if events.empty:
        return view, []
    events["year"] = events["start_time"].dt.year.astype(int)
    events["quarter"] = events["start_time"].dt.quarter.astype(int)
    events["is_shortage"] = (events["event_type"] == "drug_shortage").astype(float)
    events["is_recall"] = events["event_type"].astype(str).str.contains(
        "recall", case=False, na=False).astype(float)
    q = (events.groupby(["year", "quarter"], as_index=False)
         .agg(exo_event_count=("event_type", "size"),
              exo_event_severity_mean=("severity", "mean"),
              exo_event_confidence_mean=("confidence", "mean"),
              exo_event_shortage_count=("is_shortage", "sum"),
              exo_event_recall_count=("is_recall", "sum")))
    out = view.merge(q, on=["year", "quarter"], how="left")
    feature_cols = [c for c in q.columns if c.startswith("exo_")]
    return out, feature_cols


def add_news_event_layers(view: pd.DataFrame, news_events_path: Path) -> tuple:
    """Join structured article-event states to the feature quarter.

    These rows come from the real historical Arkansas article corpus and are
    intentionally kept separate from FDA/FEMA event counts.  Publication
    time is the availability time; the next-quarter target is never used to
    construct the join.  Missing or metadata-only articles contribute no
    fabricated values.
    """
    path = Path(news_events_path)
    if not path.exists():
        return view, []
    required = {"event_type", "source_timestamp", "confidence", "severity"}
    header = pd.read_csv(path, nrows=0).columns
    if not required <= set(header):
        return view, []
    news = pd.read_csv(path, usecols=list(required))
    news["source_timestamp"] = pd.to_datetime(news["source_timestamp"], errors="coerce", utc=True)
    news = news.dropna(subset=["source_timestamp"])
    if news.empty:
        return view, []
    news["year"] = news["source_timestamp"].dt.year.astype(int)
    news["quarter"] = news["source_timestamp"].dt.quarter.astype(int)
    news["event_type"] = news["event_type"].astype(str)
    news["is_shortage"] = news["event_type"].eq("shortage").astype(float)
    news["is_recall"] = news["event_type"].eq("recall").astype(float)
    news["is_outbreak"] = news["event_type"].eq("disease_outbreak").astype(float)
    news["is_disaster"] = news["event_type"].eq("disaster").astype(float)
    news["is_policy_trade"] = news["event_type"].eq("policy_trade").astype(float)
    q = (news.groupby(["year", "quarter"], as_index=False)
         .agg(exo_news_article_event_count=("event_type", "size"),
              exo_news_severity_mean=("severity", "mean"),
              exo_news_confidence_mean=("confidence", "mean"),
              exo_news_shortage_count=("is_shortage", "sum"),
              exo_news_recall_count=("is_recall", "sum"),
              exo_news_outbreak_count=("is_outbreak", "sum"),
              exo_news_disaster_count=("is_disaster", "sum"),
              exo_news_policy_trade_count=("is_policy_trade", "sum")))
    out = view.merge(q, on=["year", "quarter"], how="left")
    return out, [c for c in q.columns if c.startswith("exo_news_")]


def add_nadac_price_layer(
    view: pd.DataFrame,
    nadac_path: Optional[Path] = None,
) -> tuple:
    """Join observed CMS NADAC price pressure by NDC and feature quarter.

    NADAC is a public weekly acquisition-cost series.  A feature quarter uses
    only observations dated within that completed quarter, and price change is
    computed within NDC before joining.  Missing NDC matches remain missing;
    no drug-name allocation is invented.
    """
    path = NADAC_DEFAULT if nadac_path is None else Path(nadac_path)
    if "ndc" not in view.columns or not path.exists():
        return view, []
    required = {"ndc", "nadac_per_unit", "as_of_date"}
    header = set(pd.read_csv(path, nrows=0).columns)
    if not required <= header:
        return view, []
    nadac = pd.read_csv(
        path, usecols=sorted(required), dtype={"ndc": "string"},
        low_memory=False)
    nadac["ndc_key"] = nadac["ndc"].map(_normalize_ndc_identifier)
    nadac["price"] = pd.to_numeric(nadac["nadac_per_unit"], errors="coerce")
    nadac["date"] = pd.to_datetime(nadac["as_of_date"], errors="coerce")
    nadac = nadac[(nadac["ndc_key"] != "") & nadac["price"].gt(0)
                  & nadac["date"].notna()].copy()
    if nadac.empty:
        return view, []
    nadac["year"] = nadac["date"].dt.year.astype(int)
    nadac["quarter"] = nadac["date"].dt.quarter.astype(int)
    q = (nadac.groupby(["ndc_key", "year", "quarter"], as_index=False)
         .agg(exo_nadac_price_mean=("price", "mean"),
              exo_nadac_price_obs=("price", "size")))
    q = q.sort_values(["ndc_key", "year", "quarter"])
    q["exo_nadac_price_log"] = np.log1p(q["exo_nadac_price_mean"])
    prior = q.groupby("ndc_key")["exo_nadac_price_mean"].shift(1)
    prior_key = q.groupby("ndc_key")["quarter"].shift(1)
    prior_year = q.groupby("ndc_key")["year"].shift(1)
    prior_q_id = (prior_year * 4 + prior_key)
    q_id = q["year"] * 4 + q["quarter"]
    q["exo_nadac_price_qoq"] = (
        (q["exo_nadac_price_mean"] - prior)
        / prior.replace(0, np.nan)
    ).where(q_id.sub(prior_q_id).eq(1))
    out = view.copy()
    out["ndc_key"] = out.get("ndc", "").map(_normalize_ndc_identifier)
    out = out.merge(q, on=["ndc_key", "year", "quarter"], how="left")
    out = out.drop(columns=["ndc_key"])
    return out, [c for c in q.columns if c.startswith("exo_nadac_")]


def add_real_exogenous_layers(
    view: pd.DataFrame,
    annual_panel_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    news_events_path: Optional[Path] = None,
    fluview_path: Optional[Path] = None,
    wastewater_path: Optional[Path] = None,
    nadac_path: Optional[Path] = None,
    cms_partd_path: Optional[Path] = None,
) -> tuple:
    """Add leak-free real-data layers and return feature column names."""
    out = view.copy()
    feature_cols: List[str] = []
    if annual_panel_path is not None:
        out, cols = add_annual_panel_layers(out, annual_panel_path)
        feature_cols.extend(cols)
    if events_path is not None:
        out, cols = add_event_layers(out, events_path)
        feature_cols.extend(cols)
    if news_events_path is not None:
        out, cols = add_news_event_layers(out, news_events_path)
        feature_cols.extend(cols)
    if fluview_path is not None or wastewater_path is not None:
        out, cols = add_weekly_surveillance_layers(
            out, fluview_path=fluview_path, wastewater_path=wastewater_path)
        feature_cols.extend(cols)
    out, cols = add_nadac_price_layer(out, nadac_path=nadac_path)
    feature_cols.extend(cols)
    cms_path = (CMS_PARTD_DEFAULT_PATH if cms_partd_path is None
                else Path(cms_partd_path))
    out, cols = add_latest_context_layer(out, path=cms_path)
    feature_cols.extend(cols)
    feature_cols = sorted(set(feature_cols))
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, feature_cols


def add_arcos_distribution_layer(
    view: pd.DataFrame,
    arcos_path: Optional[Path] = None,
) -> tuple:
    """Join real DEA ARCOS quarterly retail distribution by normalized drug.

    ARCOS is an observed supply/distribution proxy, not a demand label. The
    join uses the feature quarter only; its lagged change is computed within
    the source before joining, so no target-quarter information is introduced.
    Unmatched drugs remain missing rather than receiving synthetic values.
    """
    if arcos_path is None or not Path(arcos_path).exists():
        return view, []
    raw = pd.read_csv(arcos_path, low_memory=False)
    required = {"year", "drug_name", "quarter1_grams", "quarter2_grams",
                "quarter3_grams", "quarter4_grams"}
    if not required <= set(raw.columns):
        return view, []
    rows = []
    for quarter in range(1, 5):
        grams = pd.to_numeric(
            raw[f"quarter{quarter}_grams"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        rows.append(pd.DataFrame({
            "year": pd.to_numeric(raw["year"], errors="coerce"),
            "quarter": quarter,
            "drug": raw["drug_name"].map(normalize_name),
            "arcos_grams": grams,
        }))
    arcos = pd.concat(rows, ignore_index=True).dropna(subset=["year", "arcos_grams"])
    arcos = arcos[arcos["drug"] != ""].copy()
    arcos["year"] = arcos["year"].astype(int)
    arcos = arcos.groupby(["year", "quarter", "drug"], as_index=False)["arcos_grams"].sum()
    arcos = arcos.sort_values(["drug", "year", "quarter"])
    arcos["arcos_grams_log"] = np.log1p(arcos["arcos_grams"].clip(lower=0))
    arcos["arcos_grams_lag"] = arcos.groupby("drug")["arcos_grams"].shift(1)
    arcos["arcos_grams_qoq"] = (
        (arcos["arcos_grams"] - arcos["arcos_grams_lag"])
        / arcos["arcos_grams_lag"].replace(0, np.nan)
    )
    keep = ["year", "quarter", "drug", "arcos_grams_log", "arcos_grams_qoq"]
    out = view.merge(arcos[keep], on=["year", "quarter", "drug"], how="left")
    return out, keep[3:]


def _epiweek_to_date(epiweek: pd.Series) -> pd.Series:
    """ISO epiweek (YYYYWW) -> Monday date of that week."""
    return pd.to_datetime(epiweek.astype(str) + "1", format="%G%V%u",
                          errors="coerce")


def _join_prior_quarter(view: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Left-join quarterly obs summaries one quarter ahead of their own.

    An observation in quarter q is joined to feature rows in quarter q+1, so
    feature rows only ever see completed weeks strictly before their own
    quarter. Unmatched feature quarters stay NaN; nothing is imputed.
    """
    obs = obs.dropna(subset=["year", "quarter"])
    if obs.empty:
        return view
    obs["year"] = obs["year"].astype(int)
    obs["quarter"] = obs["quarter"].astype(int)
    obs["q_id"] = (obs["year"] - FIRST_YEAR) * 4 + obs["quarter"] - 1 + 1
    out = view.copy()
    out["q_id"] = (out["year"] - FIRST_YEAR) * 4 + out["quarter"] - 1
    out = out.merge(obs.drop(columns=["year", "quarter"]), on="q_id", how="left")
    return out.drop(columns=["q_id"])


def _latest_issue_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Retain the latest published revision per (region, epiweek).

    FluView republishes the same epiweek under later ``issue`` (YYYYWW) or
    ``release_date`` values as reporting-lag corrections arrive. Only the most
    recent revision is kept so quarterly aggregates reflect final published
    values, never early snapshots. Rows without a revision key are kept as-is.
    """
    if raw.empty:
        return raw
    if "issue" in raw.columns:
        raw = raw.copy()
        raw["issue"] = pd.to_numeric(raw["issue"], errors="coerce")
        key = "issue"
    elif "release_date" in raw.columns:
        raw = raw.copy()
        raw["release_date"] = pd.to_datetime(raw["release_date"],
                                             errors="coerce")
        key = "release_date"
    else:
        return raw
    raw = raw.sort_values(["region", "epiweek", key], na_position="first")
    return raw.groupby(["region", "epiweek"], as_index=False).tail(1)


def _add_fluview_layer(view: pd.DataFrame, path: Path) -> tuple:
    """Prior-quarter FluView ILI summaries by region (ar, nat).

    Revisions of the same (region, epiweek) are collapsed to the latest
    published issue/release_date. Each observation week is assigned to the
    quarter of its ISO week end (Monday + 6 days), so a week is joined only to
    feature quarters that begin strictly after the week completes.
    """
    header = pd.read_csv(path, nrows=0).columns
    if not {"epiweek", "region"} <= set(header):
        return view, []
    usecols = (["epiweek", "region"]
               + [c for c in FLUVIEW_NUMERIC if c in header]
               + [c for c in ("issue", "release_date") if c in header])
    raw = pd.read_csv(path, usecols=usecols)
    raw = _latest_issue_rows(raw)
    raw["date"] = _epiweek_to_date(raw["epiweek"])
    raw = raw.dropna(subset=["date"])
    if raw.empty:
        return view, []
    raw["week_end"] = raw["date"] + pd.Timedelta(days=6)
    raw["year"] = raw["week_end"].dt.year
    raw["quarter"] = raw["week_end"].dt.quarter
    for c in FLUVIEW_NUMERIC:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    num = [c for c in FLUVIEW_NUMERIC if c in raw.columns]
    agg = {c: (c, "mean") for c in num}
    agg["weeks"] = ("epiweek", "size")
    q = raw.groupby(["year", "quarter", "region"], as_index=False).agg(**agg)
    out, cols = view, []
    for region in sorted(q["region"].dropna().unique()):
        sub = q[q["region"] == region].drop(columns=["region"])
        if "wili" in raw.columns:
            region_raw = raw[raw["region"] == region].sort_values("date")
            trajectory = (region_raw.groupby(["year", "quarter"], as_index=False)
                          .agg(
                              wili_last=("wili", "last"),
                              wili_max=("wili", "max"),
                              wili_first=("wili", "first")))
            trajectory["wili_delta"] = (
                trajectory["wili_last"] - trajectory["wili_first"])
            sub = sub.merge(trajectory, on=["year", "quarter"], how="left")
        sub = sub.rename(columns={
            c: f"exo_weekly_flu_{region}_{c}"
            for c in sub.columns if c not in ("year", "quarter")})
        out = _join_prior_quarter(out, sub)
        cols.extend(c for c in sub.columns if c.startswith("exo_weekly_"))
    return out, cols


def _add_wastewater_layer(view: pd.DataFrame, path: Path) -> tuple:
    """Prior-quarter wastewater WVAL summaries with site coverage."""
    header = pd.read_csv(path, nrows=0).columns
    required = {"week_end", "site_wval", "pathogen_target", "site"}
    if not required <= set(header):
        return view, []
    raw = pd.read_csv(path, usecols=list(required))
    raw["week_end"] = pd.to_datetime(raw["week_end"], errors="coerce")
    raw = raw.dropna(subset=["week_end"])
    if raw.empty:
        return view, []
    raw["year"] = raw["week_end"].dt.year
    raw["quarter"] = raw["week_end"].dt.quarter
    raw["site_wval"] = pd.to_numeric(raw["site_wval"], errors="coerce")
    overall = raw.groupby(["year", "quarter"], as_index=False).agg(
        exo_weekly_ww_wval_mean=("site_wval", "mean"),
        exo_weekly_ww_sites=("site", "nunique"))
    raw["pathogen"] = (raw["pathogen_target"].astype(str)
                       .map(WASTEWATER_PATHOGENS).fillna("other"))
    per = (raw.groupby(["year", "quarter", "pathogen"], as_index=False)
           ["site_wval"].mean())
    per = per.pivot(index=["year", "quarter"], columns="pathogen",
                    values="site_wval").reset_index()
    per.columns = ["year", "quarter"] + [
        f"exo_weekly_ww_{c}_wval_mean" for c in per.columns[2:]]
    out = _join_prior_quarter(view, overall)
    out = _join_prior_quarter(out, per)
    cols = ([c for c in overall.columns if c.startswith("exo_weekly_")]
            + [c for c in per.columns if c.startswith("exo_weekly_")])
    return out, cols


def add_weekly_surveillance_layers(
    view: pd.DataFrame,
    fluview_path: Optional[Path] = None,
    wastewater_path: Optional[Path] = None,
) -> tuple:
    """Join prior-completed-quarter CDC weekly surveillance summaries.

    FluView ILI and wastewater WVAL observations are weekly. Each observation
    is assigned to the quarter of its completed week end (Monday + 6 days),
    aggregated to a quarterly summary, and joined to feature rows one quarter
    later. A feature row at quarter t therefore sees only completed weeks
    strictly before t; a week crossing a quarter boundary lands in the quarter
    that begins after it completes, never one it overlaps. Missing weeks,
    sites, or pathogens contribute NaN rather than synthetic values. Absent or
    unusable files return the view unchanged with an empty feature list.
    """
    out = view.copy()
    feature_cols: List[str] = []
    if fluview_path is not None and Path(fluview_path).exists():
        out, cols = _add_fluview_layer(out, fluview_path)
        feature_cols.extend(cols)
    if wastewater_path is not None and Path(wastewater_path).exists():
        out, cols = _add_wastewater_layer(out, wastewater_path)
        feature_cols.extend(cols)
    return out, sorted(set(feature_cols))


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Forecast metrics on the raw prescription-count scale.

    Count targets use a one-prescription absolute tolerance for zero and
    near-zero observations; percentage error is only evaluated for counts
    greater than one. This avoids undefined or unstable relative errors while
    keeping the tolerance tied to the integer measurement unit.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {k: float("nan") for k in
                ("wape", "mae", "rmse", "smape", "r2", "within_5pct",
                 "within_10pct", "within_20pct", "within_5pct_eligible",
                 "within_tolerance")}
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    wape = float(np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1e-9))
    smape = float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-9)))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-9))
    abs_error = np.abs(yt - yp)
    eligible = np.abs(yt) > 1.0
    relative = np.full(len(yt), np.nan)
    relative[eligible] = abs_error[eligible] / np.abs(yt[eligible])
    within = {
        f"within_{int(rate * 100)}pct": float(np.mean(relative[eligible] <= rate))
        if eligible.any() else float("nan")
        for rate in (0.05, 0.10, 0.20)
    }
    # Prescription counts are integer observations, so one count is the
    # smallest meaningful absolute tolerance for zero/near-zero outcomes.
    tolerance = np.where(eligible, np.abs(yt) * 0.05, 1.0)
    within_tolerance = float(np.mean(abs_error <= tolerance))
    return {"wape": wape, "mae": mae, "rmse": rmse, "smape": smape, "r2": r2,
            **within, "within_5pct_eligible": int(eligible.sum()),
            "within_tolerance": within_tolerance}


def demand_state_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current: np.ndarray,
    band: float = DEMAND_STATE_BAND,
) -> Dict[str, float]:
    """Score a three-state next-quarter demand-change diagnostic.

    States are ``decrease``, ``stable`` and ``increase`` relative to the
    observed feature-quarter count.  The stable band is deliberately a fixed
    material-change threshold; it is not tuned on validation or test labels.
    This diagnostic supplements, and never replaces, numeric count metrics.
    """
    if not 0 < float(band) < 1:
        raise ValueError("band must be between zero and one")
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    cur = np.asarray(current, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(cur) & (cur >= 0)
    if not mask.any():
        return {
            "demand_state_accuracy": float("nan"),
            "demand_state_balanced_accuracy": float("nan"),
            "demand_state_macro_f1": float("nan"),
            "demand_state_rows": 0,
            "demand_state_decrease_rows": 0,
            "demand_state_stable_rows": 0,
            "demand_state_increase_rows": 0,
        }

    def _states(values: np.ndarray) -> np.ndarray:
        ratio = values / np.maximum(cur[mask], 1.0)
        return np.where(ratio < 1.0 - band, -1,
                        np.where(ratio > 1.0 + band, 1, 0))

    actual = _states(yt[mask])
    predicted = _states(yp[mask])
    accuracy = float(np.mean(actual == predicted))
    recalls = []
    f1s = []
    for state in (-1, 0, 1):
        actual_state = actual == state
        predicted_state = predicted == state
        support = int(actual_state.sum())
        if support == 0:
            continue
        recalls.append(float(np.sum(actual_state & predicted_state) / support))
        precision_denominator = int(predicted_state.sum())
        precision = (float(np.sum(actual_state & predicted_state) /
                           precision_denominator)
                     if precision_denominator else 0.0)
        recall = recalls[-1]
        f1s.append((2.0 * precision * recall / (precision + recall))
                   if precision + recall else 0.0)
    return {
        "demand_state_accuracy": accuracy,
        "demand_state_balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "demand_state_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "demand_state_rows": int(mask.sum()),
        "demand_state_decrease_rows": int(np.sum(actual == -1)),
        "demand_state_stable_rows": int(np.sum(actual == 0)),
        "demand_state_increase_rows": int(np.sum(actual == 1)),
    }


def _split_years(view: pd.DataFrame, train_cutoff: int,
                 test_end_year: Optional[int] = None) -> tuple:
    train_mask = view["year"] <= train_cutoff - 1
    val_mask = view["year"] == train_cutoff
    test_mask = view["year"] > train_cutoff
    if test_end_year is not None:
        test_mask = test_mask & (view["year"] <= test_end_year)
    return train_mask, val_mask, test_mask


def _map_group_stat(frame: pd.DataFrame, group_cols: List[str],
                    stat: pd.Series) -> np.ndarray:
    if len(group_cols) == 1:
        return frame[group_cols[0]].map(stat).to_numpy(dtype=float)
    return frame.set_index(group_cols).index.map(stat).to_numpy(dtype=float)


def _drug_quarter_transition_predictions(
    train: pd.DataFrame,
    apply: pd.DataFrame,
) -> np.ndarray:
    """Historical median target/current ratio by drug and feature quarter.

    This learns a real transition layer: for each Medicaid drug and feature
    quarter, how much next-quarter demand historically differs from current
    quarter demand. Fallbacks are quarter-level and then global medians fitted
    on the training split only.
    """
    tr = train.copy()
    tr["ratio"] = np.clip(
        tr["target"].to_numpy(dtype=float)
        / np.maximum(tr["value"].to_numpy(dtype=float), 1e-9),
        0.0,
        10.0,
    )
    drug_q = tr.groupby(["drug", "quarter"])["ratio"].median()
    q = tr.groupby("quarter")["ratio"].median()
    global_ratio = float(np.nanmedian(tr["ratio"]))
    ratio = _map_group_stat(apply, ["drug", "quarter"], drug_q)
    q_ratio = apply["quarter"].map(q).to_numpy(dtype=float)
    ratio = np.where(np.isfinite(ratio), ratio, q_ratio)
    ratio = np.where(np.isfinite(ratio), ratio, global_ratio)
    return np.maximum(apply["value"].to_numpy(dtype=float) * ratio, 0.0)


def _same_target_quarter_last_year_predictions(
    train: pd.DataFrame,
    apply: pd.DataFrame,
) -> np.ndarray:
    """Seasonal-naive forecast from the same drug/target-quarter last year.

    The lookup is fit only from observed training targets. It is a standard
    seasonal comparator for quarterly demand and is kept separate from the
    learned transition layer so the latter cannot win against an incomplete
    baseline set.
    """
    tr = train.copy()
    tr["target_year"] = tr["year"].astype(int)
    tr["target_quarter"] = tr["quarter"].astype(int) + 1
    tr.loc[tr["target_quarter"] == 5, "target_quarter"] = 1
    tr.loc[tr["quarter"] == 4, "target_year"] += 1
    lookup = tr.set_index(["drug", "target_year", "target_quarter"])["target"]
    target_year = apply["year"].astype(int).to_numpy()
    target_quarter = apply["quarter"].astype(int).to_numpy() + 1
    target_quarter[target_quarter == 5] = 1
    target_year = target_year + (apply["quarter"].astype(int).to_numpy() == 4)
    keys = pd.MultiIndex.from_arrays(
        [apply["drug"].to_numpy(), target_year - 1, target_quarter],
        names=["drug", "target_year", "target_quarter"],
    )
    seasonal = lookup.reindex(keys).to_numpy(dtype=float)
    return np.where(np.isfinite(seasonal), seasonal,
                    apply["value"].to_numpy(dtype=float))


def _croston_sba_predictions(train: pd.DataFrame, apply: pd.DataFrame,
                             alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan approximation fitted per drug on training history."""
    forecasts = {}
    intervals, sizes = [], []
    for drug, group in train.sort_values(["drug", "year", "quarter"]).groupby("drug"):
        values = group["target"].to_numpy(dtype=float)
        positive = np.flatnonzero(values > 0)
        if len(positive) == 0:
            continue
        z = float(values[positive[0]])
        p = float(positive[0] + 1)
        last = int(positive[0])
        for idx in positive[1:]:
            interval = float(idx - last)
            z = alpha * float(values[idx]) + (1.0 - alpha) * z
            p = alpha * interval + (1.0 - alpha) * p
            last = int(idx)
        forecasts[drug] = max(0.0, (1.0 - alpha / 2.0) * z / max(p, 1e-9))
        sizes.append(float(values[positive].mean()))
        intervals.append(float(np.mean(np.diff(positive)) if len(positive) > 1 else len(values)))
    fallback = (float(np.mean(sizes)) / max(float(np.mean(intervals)), 1e-9)
                if sizes and intervals else float(np.nanmean(train["target"])))
    return np.asarray([forecasts.get(drug, fallback) for drug in apply["drug"]], dtype=float)


def _drug_quarter_delta_predictions(
    train: pd.DataFrame,
    apply: pd.DataFrame,
) -> np.ndarray:
    """Historical median next-quarter minus current-quarter delta."""
    tr = train.copy()
    tr["delta"] = (tr["target"].to_numpy(dtype=float)
                   - tr["value"].to_numpy(dtype=float))
    drug_q = tr.groupby(["drug", "quarter"])["delta"].median()
    q = tr.groupby("quarter")["delta"].median()
    global_delta = float(np.nanmedian(tr["delta"]))
    delta = _map_group_stat(apply, ["drug", "quarter"], drug_q)
    q_delta = apply["quarter"].map(q).to_numpy(dtype=float)
    delta = np.where(np.isfinite(delta), delta, q_delta)
    delta = np.where(np.isfinite(delta), delta, global_delta)
    return np.maximum(apply["value"].to_numpy(dtype=float) + delta, 0.0)


def _recent_drug_quarter_transition_predictions(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    years: int = 3,
) -> np.ndarray:
    """Drug-quarter ratio using only the most recent training years."""
    max_year = int(train["year"].max())
    recent = train[train["year"] >= max_year - years + 1]
    if recent.empty:
        recent = train
    return _drug_quarter_transition_predictions(recent, apply)


def _expanding_transition_predictions(train: pd.DataFrame,
                                      apply: pd.DataFrame) -> np.ndarray:
    """One-step transition forecasts with an expanding, time-safe history."""
    by_drug_q = {}
    by_q = {}
    all_ratios = []
    for row in train[["drug", "quarter", "target", "value"]].itertuples(index=False):
        ratio = float(np.clip(row.target / max(row.value, 1e-9), 0.0, 10.0))
        by_drug_q.setdefault((row.drug, int(row.quarter)), []).append(ratio)
        by_q.setdefault(int(row.quarter), []).append(ratio)
        all_ratios.append(ratio)
    ordered = apply.sort_values(["year", "quarter", "drug"]).copy()
    predictions = {}
    for idx, row in ordered.iterrows():
        ratios = by_drug_q.get((row["drug"], int(row["quarter"])), [])
        if not ratios:
            ratios = by_q.get(int(row["quarter"]), [])
        ratio = float(np.median(ratios if ratios else all_ratios))
        predictions[idx] = float(max(row["value"] * ratio, 0.0))
        # Append the row only after forecasting it; its target is unavailable
        # at its own forecast timestamp but becomes available for later rows.
        observed_ratio = float(np.clip(row["target"] / max(row["value"], 1e-9), 0.0, 10.0))
        by_drug_q.setdefault((row["drug"], int(row["quarter"])), []).append(observed_ratio)
        by_q.setdefault(int(row["quarter"]), []).append(observed_ratio)
        all_ratios.append(observed_ratio)
    return np.asarray([predictions[i] for i in apply.index], dtype=float)


def _select_blend(y_true: np.ndarray, base: np.ndarray,
                  candidate: np.ndarray) -> tuple:
    best_w, best_wape = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, 101):
        pred = (1.0 - w) * base + w * candidate
        m = metrics_dict(y_true, pred)["wape"]
        if m < best_wape:
            best_wape, best_w = float(m), float(w)
    return best_w, best_wape


def _select_within5_blend(y_true: np.ndarray, base: np.ndarray,
                          candidate: np.ndarray) -> tuple:
    """Select a convex blend on validation within-5% accuracy only."""
    best_w, best_score, best_wape = 0.0, -1.0, float("inf")
    for w in np.linspace(0.0, 1.0, 101):
        pred = (1.0 - w) * base + w * candidate
        m = metrics_dict(y_true, pred)
        score = float(m["within_5pct"])
        if score > best_score or (score == best_score and m["wape"] < best_wape):
            best_score, best_wape, best_w = score, float(m["wape"]), float(w)
    return best_w, best_score


def _select_transition_simplex(
    y_true: np.ndarray,
    candidates: List[np.ndarray],
    step: float = 0.05,
) -> tuple:
    """Validation-selected convex weights over transition candidates."""
    n_steps = int(round(1.0 / step))
    best_w = np.zeros(len(candidates))
    best_w[0] = 1.0
    best_wape = float("inf")
    if len(candidates) != 4:
        raise ValueError("transition simplex expects four candidates")
    for ia in range(n_steps + 1):
        a = ia * step
        for ib in range(n_steps - ia + 1):
            b = ib * step
            for ic in range(n_steps - ia - ib + 1):
                c = ic * step
                d = 1.0 - a - b - c
                weights = np.asarray([a, b, c, d], dtype=float)
                pred = sum(weights[i] * candidates[i]
                           for i in range(len(candidates)))
                m = metrics_dict(y_true, pred)["wape"]
                if m < best_wape:
                    best_wape = float(m)
                    best_w = weights
    return best_w, best_wape


def _select_accuracy_simplex(y_true: np.ndarray,
                             candidates: List[np.ndarray],
                             step: float = 0.05) -> tuple:
    """Validation-selected simplex maximizing within-5% point accuracy."""
    n_steps = int(round(1.0 / step))
    best_w = np.zeros(len(candidates)); best_w[0] = 1.0
    best_score, best_wape = -1.0, float("inf")
    if len(candidates) != 4:
        raise ValueError("accuracy simplex expects four candidates")
    for ia in range(n_steps + 1):
        a = ia * step
        for ib in range(n_steps - ia + 1):
            b = ib * step
            for ic in range(n_steps - ia - ib + 1):
                c = ic * step
                d = 1.0 - a - b - c
                weights = np.asarray([a, b, c, d], dtype=float)
                pred = sum(weights[i] * candidates[i] for i in range(4))
                m = metrics_dict(y_true, pred)
                if (m["within_5pct"] > best_score
                        or (m["within_5pct"] == best_score and m["wape"] < best_wape)):
                    best_score, best_wape, best_w = float(m["within_5pct"]), float(m["wape"]), weights
    return best_w, best_score


def _quarter_start(year: int, quarter: int) -> str:
    month = (int(quarter) - 1) * 3 + 1
    return f"{int(year):04d}-{month:02d}-01"


def _next_quarter(year: int, quarter: int) -> tuple:
    if int(quarter) >= 4:
        return int(year) + 1, 1
    return int(year), int(quarter) + 1


def _select_quarterly_transition_model(train: pd.DataFrame,
                                       val: pd.DataFrame,
                                       score: pd.DataFrame) -> Dict:
    """Select a deployable transition model by validation WAPE and score rows."""
    y_val = val["target"].to_numpy(dtype=float)
    prev_val = val["value"].to_numpy(dtype=float)
    ma2_val = val["ma2"].to_numpy(dtype=float)
    dq_val = _drug_quarter_transition_predictions(train, val)
    dq_score = _drug_quarter_transition_predictions(train, score)
    delta_val = _drug_quarter_delta_predictions(train, val)
    delta_score = _drug_quarter_delta_predictions(train, score)
    recent_val = _recent_drug_quarter_transition_predictions(train, val)
    recent_score = _recent_drug_quarter_transition_predictions(train, score)

    candidates: List[Dict] = [
        {
            "model": "previous_quarter",
            "validation_wape": metrics_dict(y_val, prev_val)["wape"],
            "prediction": score["value"].to_numpy(dtype=float),
            "weights": [1.0, 0.0, 0.0, 0.0],
            "members": ["previous_quarter"],
        },
        {
            "model": "ma2",
            "validation_wape": metrics_dict(y_val, ma2_val)["wape"],
            "prediction": score["ma2"].to_numpy(dtype=float),
            "weights": [0.0, 0.0, 0.0, 0.0],
            "members": ["ma2"],
        },
    ]
    candidates.append({
        "model": "drug_quarter_transition",
        "validation_wape": metrics_dict(y_val, dq_val)["wape"],
        "prediction": dq_score,
        "weights": [0.0, 1.0, 0.0, 0.0],
        "members": ["previous_quarter", "drug_quarter_transition",
                    "drug_quarter_delta", "recent_drug_quarter_transition"],
    })

    w_dq, _ = _select_blend(y_val, prev_val, dq_val)
    candidates.append({
        "model": "drug_quarter_transition_blend",
        "validation_wape": metrics_dict(
            y_val, (1.0 - w_dq) * prev_val + w_dq * dq_val)["wape"],
        "prediction": (1.0 - w_dq) * score["value"].to_numpy(dtype=float)
                      + w_dq * dq_score,
        "weights": [1.0 - w_dq, w_dq, 0.0, 0.0],
        "members": ["previous_quarter", "drug_quarter_transition"],
    })

    weights, _ = _select_transition_simplex(
        y_val, [prev_val, dq_val, delta_val, recent_val])
    candidates.append({
        "model": "multi_transition_blend",
        "validation_wape": metrics_dict(
            y_val,
            weights[0] * prev_val + weights[1] * dq_val
            + weights[2] * delta_val + weights[3] * recent_val)["wape"],
        "prediction": (weights[0] * score["value"].to_numpy(dtype=float)
                       + weights[1] * dq_score
                       + weights[2] * delta_score
                       + weights[3] * recent_score),
        "weights": weights.tolist(),
        "members": ["previous_quarter", "drug_quarter_transition",
                    "drug_quarter_delta", "recent_drug_quarter_transition"],
    })

    w_ma2, _ = _select_blend(y_val, ma2_val, dq_val)
    candidates.append({
        "model": "ma2_transition_blend",
        "validation_wape": metrics_dict(
            y_val, (1.0 - w_ma2) * ma2_val + w_ma2 * dq_val)["wape"],
        "prediction": ((1.0 - w_ma2) * score["ma2"].to_numpy(dtype=float)
                       + w_ma2 * dq_score),
        "weights": [1.0 - w_ma2, w_ma2],
        "members": ["ma2", "drug_quarter_transition"],
    })

    ma2_weights, _ = _select_transition_simplex(
        y_val, [ma2_val, dq_val, delta_val, recent_val])
    candidates.append({
        "model": "ma2_multi_transition_blend",
        "validation_wape": metrics_dict(
            y_val,
            ma2_weights[0] * ma2_val + ma2_weights[1] * dq_val
            + ma2_weights[2] * delta_val + ma2_weights[3] * recent_val)["wape"],
        "prediction": (ma2_weights[0] * score["ma2"].to_numpy(dtype=float)
                       + ma2_weights[1] * dq_score
                       + ma2_weights[2] * delta_score
                       + ma2_weights[3] * recent_score),
        "weights": ma2_weights.tolist(),
        "members": ["ma2", "drug_quarter_transition",
                    "drug_quarter_delta", "recent_drug_quarter_transition"],
    })
    best = min(candidates, key=lambda c: c["validation_wape"])
    best["prediction"] = np.maximum(np.asarray(best["prediction"], dtype=float), 0.0)
    # Keep validation-only candidate evidence available to the rolling selector
    # without serializing prediction arrays into artifacts.
    best["candidate_validation_wapes"] = {
        str(candidate["model"]): float(candidate["validation_wape"])
        for candidate in candidates
    }
    best["candidate_predictions"] = {
        str(candidate["model"]): np.maximum(
            np.asarray(candidate["prediction"], dtype=float), 0.0)
        for candidate in candidates
    }
    best["selection_method"] = "single_validation_year"
    return best


def _select_quarterly_transition_model_rolling(
    view: pd.DataFrame,
    score: pd.DataFrame,
    validation_years: int = 3,
) -> Dict:
    """Select a forecast family using multiple chronological validation years.

    Each fold fits only on feature years before its validation year. Candidate
    family scores are averaged across eligible folds, then the winning family
    is refit using the final fold's train/validation arrangement to produce
    live scores. This prevents the latest sparse validation year from
    unilaterally selecting a deployment family. If fewer than two folds meet
    the minimum row contract, the original single-year selector is used.
    """
    if int(validation_years) < 1:
        raise ValueError("validation_years must be positive")
    years = sorted(int(year) for year in view["year"].dropna().unique())
    if not years:
        raise ValueError("quarterly view has no validation years")
    eligible_years = years[-int(validation_years):]
    folds = []
    for year in eligible_years:
        train = view[view["year"] < year]
        val = view[view["year"] == year]
        if len(train) < 30 or len(val) < 5:
            continue
        fold = _select_quarterly_transition_model(train, val, val)
        folds.append({
            "validation_year": year,
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "candidate_validation_wapes": fold["candidate_validation_wapes"],
        })

    final_year = years[-1]
    final_train = view[view["year"] < final_year]
    final_val = view[view["year"] == final_year]
    final = _select_quarterly_transition_model(final_train, final_val, score)
    if len(folds) < 2:
        final["selection_method"] = "single_validation_year_fallback"
        final["validation_years"] = [final_year]
        final["rolling_validation_folds"] = folds
        return final

    all_models = sorted({
        model for fold in folds
        for model in fold["candidate_validation_wapes"]
    })
    mean_wapes = {
        model: float(np.mean([
            fold["candidate_validation_wapes"][model]
            for fold in folds if model in fold["candidate_validation_wapes"]
        ]))
        for model in all_models
    }
    selected_model = min(mean_wapes, key=mean_wapes.get)
    candidate_predictions = final["candidate_predictions"]
    if selected_model not in candidate_predictions:
        # This should only be possible if a future candidate is unavailable in
        # the final fold; fail closed to the final fold's selected candidate.
        selected_model = str(final["model"])
    final["model"] = selected_model
    final["prediction"] = candidate_predictions[selected_model]
    final["validation_wape"] = float(mean_wapes[selected_model])
    final["candidate_validation_wapes"] = mean_wapes
    final["selection_method"] = "rolling_validation_mean_wape"
    final["validation_years"] = [int(fold["validation_year"]) for fold in folds]
    final["rolling_validation_folds"] = folds
    return final


def build_quarterly_forecast_grid(
    panel: pd.DataFrame,
    max_rows: int = 10000,
    horizons_days: Optional[List[int]] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Forecast next-quarter Arkansas Medicaid prescription counts by drug.

    The model is selected on rolling validation years, then scored only for
    drugs observed in the global latest feature quarter. Stale drug histories
    are excluded rather than assigned an old forecast date. Outputs use the
    same forecast contract as the annual grid but are state-level Medicaid
    demand signals, not city/supplier inventory predictions.
    """
    horizons_days = horizons_days or [91, 182, 365]
    view = build_quarterly_view(panel)
    if view.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    score = build_quarterly_scoring_rows(panel)
    score = score.dropna(subset=["value"])
    # A live application must not publish a forecast whose source history is
    # years behind the current panel merely because that drug has a valid
    # historical row.  Use exact global-quarter freshness; missing observations
    # remain missing and are not backfilled.
    panel_q_id = ((pd.to_numeric(panel["year"], errors="coerce") - FIRST_YEAR) * 4
                  + pd.to_numeric(panel["quarter"], errors="coerce") - 1)
    latest_q_id = int(panel_q_id.max())
    score_q_id = ((pd.to_numeric(score["year"], errors="coerce") - FIRST_YEAR) * 4
                  + pd.to_numeric(score["quarter"], errors="coerce") - 1)
    score = score.loc[score_q_id.eq(latest_q_id)].reset_index(drop=True)
    if score.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    selected = _select_quarterly_transition_model_rolling(view, score)
    next_preds = selected["prediction"]
    validation_years = selected.get("validation_years", [int(view["year"].max())])
    validation_year = int(validation_years[-1])
    latest_year = score["year"].astype(int).to_numpy()
    latest_quarter = score["quarter"].astype(int).to_numpy()

    order = np.argsort(-score["value"].to_numpy(dtype=float))
    run_id = run_id or f"arq-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    created_at = datetime.now(timezone.utc).isoformat()
    rows: List[Dict] = []
    for idx in order:
        base = score.iloc[int(idx)]
        f_year, f_quarter = _next_quarter(latest_year[int(idx)],
                                          latest_quarter[int(idx)])
        q_pred = float(next_preds[int(idx)])
        val_wape = float(selected["validation_wape"])
        for horizon in horizons_days:
            scale = float(horizon) / QUARTER_DAYS
            prediction = q_pred * scale
            low = max(0.0, prediction * max(0.0, 1.0 - val_wape))
            high = prediction * (1.0 + val_wape)
            row = {
                "forecast_run_id": run_id,
                "forecast_created_at": created_at,
                "forecast_date": _quarter_start(f_year, f_quarter),
                "horizon_days": int(horizon),
                "geography_level": "state",
                "geography_id": "AR",
                "geography_name": "Arkansas",
                "drug_key": str(base["drug"]),
                "drug_name": str(base["drug"]),
                "ingredient_key": str(base.get("ingredient", base["drug"])),
                "ingredient_name": str(base.get("ingredient", base["drug"])),
                "supplier_key": str(base.get("manufacturer", "")),
                "supplier_name": str(base.get("manufacturer", "")),
                "disease_key": "demand_history",
                "disease_name": "demand_history",
                "target": "medicaid_prescription_count",
                "prediction": prediction,
                "prediction_interval_low": low,
                "prediction_interval_high": high,
                "risk_score": prediction,
                "model_family": "quarterly_medicaid_selected_transition",
                    "driver_summary_json": json.dumps({
                        "selected_model": selected["model"],
                        "validation_year": validation_year,
                        "validation_years": validation_years,
                        "selection_method": selected.get("selection_method", "unknown"),
                        "validation_wape": val_wape,
                    "members": selected["members"],
                    "weights": selected["weights"],
                    "source": "Arkansas Medicaid SDUD prescription_count",
                }),
                "source_feature_window_start": _quarter_start(
                    int(base["year"]), int(base["quarter"])),
                "source_feature_window_end": _quarter_start(
                    int(base["year"]), int(base["quarter"])),
            }
            rows.append(row)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def evaluate_quarterly(
    panel: pd.DataFrame,
    train_cutoff: int = 2021,
    annual_panel_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    news_events_path: Optional[Path] = None,
    arcos_path: Optional[Path] = None,
    test_end_year: Optional[int] = None,
    fluview_path: Optional[Path] = None,
    wastewater_path: Optional[Path] = None,
    nadac_path: Optional[Path] = None,
) -> Dict:
    """Full strict next-quarter leaderboard and publishability check."""
    fluview_path = (fluview_path if fluview_path is not None
                    else (FLUVIEW_DEFAULT if FLUVIEW_DEFAULT.exists() else None))
    wastewater_path = (wastewater_path if wastewater_path is not None
                       else (WASTEWATER_DEFAULT if WASTEWATER_DEFAULT.exists()
                             else None))
    view = build_quarterly_view(panel)
    view, exogenous_cols = add_real_exogenous_layers(
        view, annual_panel_path=annual_panel_path, events_path=events_path,
        news_events_path=news_events_path, fluview_path=fluview_path,
        wastewater_path=wastewater_path, nadac_path=nadac_path)
    view, arcos_cols = add_arcos_distribution_layer(view, arcos_path=arcos_path)
    exogenous_cols.extend(arcos_cols)
    train_mask, val_mask, test_mask = _split_years(
        view, train_cutoff, test_end_year=test_end_year)
    train, val, test = view[train_mask], view[val_mask], view[test_mask]

    result: Dict = {
        "split": "strict_next_quarter_time",
        "feature_years": {
            "train": f"<= {train_cutoff - 1}",
            "validation": str(train_cutoff),
            "test": (f"> {train_cutoff}" if test_end_year is None
                     else f"{train_cutoff + 1}-{test_end_year}"),
        },
        "n_rows": {"train": int(len(train)), "validation": int(len(val)),
                   "test": int(len(test))},
        "exogenous_feature_count": int(len(exogenous_cols)),
    }
    if len(train) < 30 or len(test) < 5:
        result["error"] = f"insufficient rows train={len(train)} test={len(test)}"
        return result

    y_train = train["target"].to_numpy(dtype=float)
    y_val = val["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)
    prev_val = val["value"].to_numpy(dtype=float)
    ma2_val = val["ma2"].to_numpy(dtype=float)

    def _row(name: str, family: str, pred: np.ndarray) -> Dict:
        m = metrics_dict(y_test, pred)
        m.update(demand_state_metrics(
            y_test, pred, test["value"].to_numpy(dtype=float)))
        prediction_store[name] = np.asarray(pred, dtype=float)
        return {"model": name, "family": family, "params": 0, **m}

    def _model_row(name: str, family: str, pred: np.ndarray,
                   val_pred: np.ndarray, params: int = 0) -> Dict:
        row = _row(name, family, pred)
        row["params"] = int(params)
        validation_metrics = metrics_dict(y_val, val_pred)
        validation_states = demand_state_metrics(
            y_val, val_pred, val["value"].to_numpy(dtype=float))
        row["validation_wape"] = validation_metrics["wape"]
        row["validation_within_5pct"] = validation_metrics["within_5pct"]
        row["validation_within_10pct"] = validation_metrics["within_10pct"]
        row["validation_within_20pct"] = validation_metrics["within_20pct"]
        row["validation_demand_state_accuracy"] = validation_states["demand_state_accuracy"]
        row["validation_demand_state_balanced_accuracy"] = (
            validation_states["demand_state_balanced_accuracy"])
        row["deployable_candidate"] = True
        return row

    leaderboard: List[Dict] = []
    baseline_preds: Dict[str, np.ndarray] = {}
    prediction_store: Dict[str, np.ndarray] = {}

    # ---- naive baselines ---------------------------------------------------
    baseline_preds["previous_quarter"] = test["value"].to_numpy(dtype=float)
    baseline_preds["ma2"] = test["ma2"].to_numpy(dtype=float)
    baseline_preds["same_target_quarter_last_year"] = (
        _same_target_quarter_last_year_predictions(train, test))
    baseline_preds["croston_sba"] = _croston_sba_predictions(train, test)
    global_mean = float(np.nanmean(y_train))
    drug_means = train.groupby("drug")["target"].mean()
    baseline_preds["drug_mean"] = (
        test["drug"].map(drug_means).fillna(global_mean).to_numpy(dtype=float))
    baseline_preds["global_mean"] = np.full(len(y_test), global_mean)
    # Baseline selection is part of model selection and must use validation
    # labels only. Test labels are reserved for the final score.
    val_baselines = {
        "previous_quarter": prev_val,
        "ma2": val["ma2"].to_numpy(dtype=float),
        "same_target_quarter_last_year": (
            _same_target_quarter_last_year_predictions(train, val)),
        "croston_sba": _croston_sba_predictions(train, val),
        "drug_mean": val["drug"].map(drug_means).fillna(global_mean).to_numpy(dtype=float),
        "global_mean": np.full(len(y_val), global_mean),
    }
    for name, pred in baseline_preds.items():
        row = _row(name, "naive", pred)
        validation_metrics = metrics_dict(y_val, val_baselines[name])
        validation_states = demand_state_metrics(
            y_val, val_baselines[name], val["value"].to_numpy(dtype=float))
        row["validation_wape"] = validation_metrics["wape"]
        row["validation_within_5pct"] = validation_metrics["within_5pct"]
        row["validation_within_10pct"] = validation_metrics["within_10pct"]
        row["validation_within_20pct"] = validation_metrics["within_20pct"]
        row["validation_demand_state_accuracy"] = validation_states["demand_state_accuracy"]
        row["validation_demand_state_balanced_accuracy"] = (
            validation_states["demand_state_balanced_accuracy"])
        row["deployable_candidate"] = True
        leaderboard.append(row)

    best_naive = min(
        val_baselines,
        key=lambda n: metrics_dict(y_val, val_baselines[n])["wape"],
    )
    naive_wape = metrics_dict(y_test, baseline_preds[best_naive])["wape"]
    validation_naive_wape = metrics_dict(
        y_val, val_baselines[best_naive])["wape"]

    # ---- real transition layer: drug x quarter historical ratio ------------
    dq_val = _drug_quarter_transition_predictions(train, val)
    dq_test = _drug_quarter_transition_predictions(train, test)
    dq_raw = _model_row(
        "drug_quarter_transition",
        "historical_drug_quarter_transition",
        dq_test,
        dq_val,
        params=int(train.groupby(["drug", "quarter"]).ngroups + 1),
    )
    leaderboard.append(dq_raw)
    best_dq_w, best_dq_val = _select_blend(y_val, prev_val, dq_val)
    dq_blend_test = ((1.0 - best_dq_w) * test["value"].to_numpy(dtype=float)
                     + best_dq_w * dq_test)
    dq_blend_val = ((1.0 - best_dq_w) * prev_val + best_dq_w * dq_val)
    dq_blend = _model_row(
        "drug_quarter_transition_blend",
        "validated_seasonal_transition",
        dq_blend_test,
        dq_blend_val,
        params=dq_raw["params"],
    )
    leaderboard.append(dq_blend)
    result["drug_quarter_transition"] = {
        "members": ["previous_quarter", "drug_quarter_transition"],
        "blend_weight": best_dq_w,
        "validation_wape": best_dq_val,
        "parameter_count": dq_raw["params"],
        "fitted_on": "train split only",
    }

    # A separate candidate optimizes the requested pointwise accuracy gate on
    # validation. It is intentionally not substituted for WAPE selection
    # unless validation proves that it generalizes.
    acc_w, acc_val = _select_within5_blend(y_val, prev_val, dq_val)
    acc_test = ((1.0 - acc_w) * test["value"].to_numpy(dtype=float)
                + acc_w * dq_test)
    acc_val_pred = (1.0 - acc_w) * prev_val + acc_w * dq_val
    acc_row = _model_row(
        "within5_transition_blend",
        "validation_within5_transition",
        acc_test,
        acc_val_pred,
        params=dq_raw["params"],
    )
    acc_row["selection_metric"] = "validation_within_5pct"
    acc_row["selection_score"] = float(acc_val)
    leaderboard.append(acc_row)
    result["within5_transition_blend"] = {
        "blend_weight": acc_w,
        "validation_within_5pct": acc_val,
        "fitted_on": "train split only",
    }

    expanding_val = _expanding_transition_predictions(train, val)
    expanding_test = _expanding_transition_predictions(
        pd.concat([train, val], ignore_index=True), test)
    expanding_w, expanding_val_wape = _select_blend(
        y_val, prev_val, expanding_val)
    expanding_test_pred = ((1.0 - expanding_w)
                           * test["value"].to_numpy(dtype=float)
                           + expanding_w * expanding_test)
    expanding_val_pred = (1.0 - expanding_w) * prev_val + expanding_w * expanding_val
    expanding_row = _model_row(
        "expanding_transition_blend",
        "time_safe_expanding_transition",
        expanding_test_pred,
        expanding_val_pred,
        params=dq_raw["params"],
    )
    leaderboard.append(expanding_row)
    result["expanding_transition_blend"] = {
        "blend_weight": expanding_w,
        "validation_wape": expanding_val_wape,
        "history": "train plus observations strictly before each forecast timestamp",
    }

    accuracy_weights, accuracy_val = _select_accuracy_simplex(
        y_val, [prev_val, ma2_val, dq_val, expanding_val])
    accuracy_test_pred = (
        accuracy_weights[0] * test["value"].to_numpy(dtype=float)
        + accuracy_weights[1] * test["ma2"].to_numpy(dtype=float)
        + accuracy_weights[2] * dq_test
        + accuracy_weights[3] * expanding_test)
    accuracy_val_pred = (
        accuracy_weights[0] * prev_val + accuracy_weights[1] * ma2_val
        + accuracy_weights[2] * dq_val + accuracy_weights[3] * expanding_val)
    accuracy_row = _model_row(
        "accuracy_transition_simplex",
        "validation_within5_simplex",
        accuracy_test_pred,
        accuracy_val_pred,
        params=dq_raw["params"] * 2,
    )
    accuracy_row["selection_metric"] = "validation_within_5pct"
    accuracy_row["selection_score"] = float(accuracy_val)
    leaderboard.append(accuracy_row)
    result["accuracy_transition_simplex"] = {
        "weights": accuracy_weights.tolist(),
        "validation_within_5pct": accuracy_val,
        "members": ["previous_quarter", "ma2", "drug_quarter_transition",
                     "expanding_transition"],
    }

    # Validation-only multiplicative bias correction, motivated by forecast
    # combination literature. It cannot use test totals or future labels.
    base_val = dq_blend_val
    base_test = dq_blend_test
    best_scale, best_scale_wape = 1.0, float("inf")
    for scale in np.linspace(0.50, 1.50, 101):
        score = metrics_dict(y_val, base_val * scale)["wape"]
        if score < best_scale_wape:
            best_scale, best_scale_wape = float(scale), float(score)
    bias_row = _model_row(
        "bias_corrected_transition",
        "validation_multiplicative_bias_correction",
        base_test * best_scale,
        base_val * best_scale,
        params=dq_raw["params"],
    )
    bias_row["bias_scale"] = best_scale
    leaderboard.append(bias_row)
    result["bias_corrected_transition"] = {
        "scale": best_scale,
        "validation_wape": best_scale_wape,
        "base": "drug_quarter_transition_blend",
    }

    delta_val = _drug_quarter_delta_predictions(train, val)
    delta_test = _drug_quarter_delta_predictions(train, test)
    recent_val = _recent_drug_quarter_transition_predictions(train, val)
    recent_test = _recent_drug_quarter_transition_predictions(train, test)
    multi_weights, multi_val_wape = _select_transition_simplex(
        y_val, [prev_val, dq_val, delta_val, recent_val])
    multi_test = (
        multi_weights[0] * test["value"].to_numpy(dtype=float)
        + multi_weights[1] * dq_test
        + multi_weights[2] * delta_test
        + multi_weights[3] * recent_test
    )
    multi_val = (
        multi_weights[0] * prev_val
        + multi_weights[1] * dq_val
        + multi_weights[2] * delta_val
        + multi_weights[3] * recent_val
    )
    multi_row = _model_row(
        "multi_transition_blend",
        "validated_multi_transition",
        multi_test,
        multi_val,
        params=int(dq_raw["params"] * 3),
    )
    leaderboard.append(multi_row)
    result["multi_transition_blend"] = {
        "members": [
            "previous_quarter",
            "drug_quarter_transition",
            "drug_quarter_delta",
            "recent_drug_quarter_transition",
        ],
        "weights": multi_weights.tolist(),
        "validation_wape": multi_val_wape,
        "fitted_on": "train split only",
    }

    best_ma2_dq_w, best_ma2_dq_val = _select_blend(y_val, ma2_val, dq_val)
    ma2_dq_test = ((1.0 - best_ma2_dq_w) * test["ma2"].to_numpy(dtype=float)
                   + best_ma2_dq_w * dq_test)
    ma2_dq_val = ((1.0 - best_ma2_dq_w) * ma2_val + best_ma2_dq_w * dq_val)
    ma2_dq_row = _model_row(
        "ma2_transition_blend",
        "validated_ma2_transition",
        ma2_dq_test,
        ma2_dq_val,
        params=dq_raw["params"],
    )
    leaderboard.append(ma2_dq_row)
    ma2_multi_weights, ma2_multi_val_wape = _select_transition_simplex(
        y_val, [ma2_val, dq_val, delta_val, recent_val])
    ma2_multi_test = (
        ma2_multi_weights[0] * test["ma2"].to_numpy(dtype=float)
        + ma2_multi_weights[1] * dq_test
        + ma2_multi_weights[2] * delta_test
        + ma2_multi_weights[3] * recent_test
    )
    ma2_multi_val = (
        ma2_multi_weights[0] * ma2_val
        + ma2_multi_weights[1] * dq_val
        + ma2_multi_weights[2] * delta_val
        + ma2_multi_weights[3] * recent_val
    )
    ma2_multi_row = _model_row(
        "ma2_multi_transition_blend",
        "validated_ma2_multi_transition",
        ma2_multi_test,
        ma2_multi_val,
        params=int(dq_raw["params"] * 3),
    )
    leaderboard.append(ma2_multi_row)
    result["ma2_multi_transition_blend"] = {
        "members": [
            "ma2",
            "drug_quarter_transition",
            "drug_quarter_delta",
            "recent_drug_quarter_transition",
        ],
        "weights": ma2_multi_weights.tolist(),
        "ma2_transition_weight": best_ma2_dq_w,
        "ma2_transition_validation_wape": best_ma2_dq_val,
        "validation_wape": ma2_multi_val_wape,
        "fitted_on": "train split only",
    }

    # ---- history ridge and calibrated blend --------------------------------
    X_train = _fill_nan(train[FEATURE_COLS].to_numpy(dtype=float))
    X_val = _fill_nan(val[FEATURE_COLS].to_numpy(dtype=float))
    X_test = _fill_nan(test[FEATURE_COLS].to_numpy(dtype=float))
    w_train = train["value"].to_numpy(dtype=float)

    ridge = _fit_ridge_log(X_train, y_train, FEATURE_COLS, alpha=10.0,
                             sample_weight=w_train)
    ridge_val = _ridge_log_predict(ridge, X_val)
    ridge_test = _ridge_log_predict(ridge, X_test)
    leaderboard.append(_model_row("ridge_history", "ridge_linear", ridge_test,
                                  ridge_val, params=len(FEATURE_COLS) + 1))

    # Validation-selected convex blend with previous_quarter. Validation
    # labels are used only to pick the weight; test labels never enter it.
    best_w, best_val_wape = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        m = metrics_dict(y_val, (1.0 - w) * prev_val + w * ridge_val)["wape"]
        if m < best_val_wape:
            best_val_wape, best_w = float(m), float(w)
    calib_test = ((1.0 - best_w) * test["value"].to_numpy(dtype=float)
                  + best_w * ridge_test)
    calib_val = ((1.0 - best_w) * prev_val + best_w * ridge_val)
    leaderboard.append(_model_row(
        "calibrated_ridge_blend", "validated_convex_blend",
        calib_test, calib_val, params=len(FEATURE_COLS) + 1))
    result["calibrated_blend"] = {
        "members": ["previous_quarter", "ridge_history"],
        "blend_weight": best_w,
        "ridge_alpha": 10.0,
        "validation_wape": best_val_wape,
    }

    if exogenous_cols:
        full_cols = FEATURE_COLS + exogenous_cols
        Xf_train = _fill_nan(train[full_cols].to_numpy(dtype=float))
        Xf_val = _fill_nan(val[full_cols].to_numpy(dtype=float))
        Xf_test = _fill_nan(test[full_cols].to_numpy(dtype=float))
        ridge_full = _fit_ridge_log(Xf_train, y_train, full_cols, alpha=25.0,
                                    sample_weight=w_train)
        full_val = _ridge_log_predict(ridge_full, Xf_val)
        full_test = _ridge_log_predict(ridge_full, Xf_test)
        row = _model_row("ridge_history_exogenous", "ridge_linear_exogenous",
                         full_test, full_val, params=len(full_cols) + 1)
        leaderboard.append(row)

        best_wx, best_val_wapex = 0.0, float("inf")
        for w in np.linspace(0.0, 1.0, 21):
            m = metrics_dict(y_val, (1.0 - w) * prev_val + w * full_val)["wape"]
            if m < best_val_wapex:
                best_val_wapex, best_wx = float(m), float(w)
        full_blend_test = ((1.0 - best_wx) * test["value"].to_numpy(dtype=float)
                           + best_wx * full_test)
        full_blend_val = ((1.0 - best_wx) * prev_val + best_wx * full_val)
        row = _model_row(
            "calibrated_exogenous_blend",
            "validated_convex_blend_exogenous",
            full_blend_test,
            full_blend_val,
            params=len(full_cols) + 1,
        )
        leaderboard.append(row)
        result["calibrated_exogenous_blend"] = {
            "members": ["previous_quarter", "ridge_history_exogenous"],
            "blend_weight": best_wx,
            "ridge_alpha": 25.0,
            "validation_wape": best_val_wapex,
            "feature_count": len(full_cols),
        }

        resid_train = (np.log1p(np.clip(y_train, 0, None))
                       - np.log1p(np.clip(train["value"], 0, None)))
        resid_model = RidgeLinear(alpha=50.0).fit(
            Xf_train, resid_train, full_cols, sample_weight=w_train)
        resid_pred = resid_model.predict(Xf_test)
        lo, hi = np.nanpercentile(resid_train, [2.5, 97.5])
        shock_test = residual_to_raw(test["value"].to_numpy(dtype=float),
                                     resid_pred, float(lo), float(hi))
        resid_val_pred = resid_model.predict(Xf_val)
        shock_val = residual_to_raw(val["value"].to_numpy(dtype=float),
                                    resid_val_pred, float(lo), float(hi))
        row = _model_row("residual_shock_exogenous",
                         "ridge_residual_exogenous",
                         shock_test, shock_val,
                         params=len(full_cols) + 1)
        leaderboard.append(row)

    # ---- leaderboard with improvements -------------------------------------
    ldf = pd.DataFrame(leaderboard)
    prev_wape = metrics_dict(y_test, baseline_preds["previous_quarter"])["wape"]
    ldf["improvement_vs_previous_quarter"] = (
        (prev_wape - ldf["wape"]) / max(prev_wape, 1e-9))
    ldf = ldf.sort_values("wape").reset_index(drop=True)
    result["leaderboard"] = ldf

    best_naive_wape = metrics_dict(y_test, baseline_preds[best_naive])["wape"]
    model_rows = ldf[~ldf["model"].isin(NAIVE_BASELINES)]
    best_model_row = model_rows.loc[model_rows["wape"].idxmin()]
    best_model_wape = float(best_model_row["wape"])
    # A learned model must earn deployment against the same validation-only
    # baselines used to measure it. Do not force a transition model when
    # persistence or a seasonal baseline is the safer validated choice.
    deployable_rows = ldf[ldf.get("deployable_candidate", False) == True]
    if deployable_rows.empty:
        selected_row = best_model_row
    else:
        selected_row = deployable_rows.loc[deployable_rows["validation_wape"].idxmin()]
    selected_wape = float(selected_row["wape"])
    selected_prediction = prediction_store[str(selected_row["model"])]
    subgroup_rows = []
    for group_name, group_values in (("drug", test["drug"]),
                                     ("year", test["year"]),
                                     ("quarter", test["quarter"])):
        grouped = test.assign(_group=group_values.to_numpy(),
                              _prediction=selected_prediction)
        for key, frame in grouped.groupby("_group", dropna=False):
            gm = metrics_dict(frame["target"].to_numpy(dtype=float),
                              frame["_prediction"].to_numpy(dtype=float))
            subgroup_rows.append({"group_type": group_name, "group": str(key),
                                  "rows": int(len(frame)), **gm})
    subgroup_frame = pd.DataFrame(subgroup_rows)
    result["selected_subgroup_accuracy"] = {
        "group_count": int(len(subgroup_frame)),
        "worst_drugs_by_within_5pct": (
            subgroup_frame[subgroup_frame["group_type"] == "drug"]
            .sort_values(["within_5pct", "rows"])
            .head(10).to_dict("records")
            if not subgroup_frame.empty else []),
        "by_year": (subgroup_frame[subgroup_frame["group_type"] == "year"]
                     .sort_values("group").to_dict("records")
                     if not subgroup_frame.empty else []),
        "by_quarter": (subgroup_frame[subgroup_frame["group_type"] == "quarter"]
                        .sort_values("group").to_dict("records")
                        if not subgroup_frame.empty else []),
    }
    aggregate_rows = []
    for group_name, group_cols in (("quarter", ["year", "quarter"]),
                                   ("year", ["year"]),
                                   ("arkansas_total", [])):
        if group_cols:
            aggregate = (test.assign(_prediction=selected_prediction)
                         .groupby(group_cols, as_index=False)
                         .agg(target=("target", "sum"), prediction=("_prediction", "sum")))
        else:
            aggregate = pd.DataFrame({"target": [test["target"].sum()],
                                      "prediction": [selected_prediction.sum()]})
        am = metrics_dict(aggregate["target"].to_numpy(dtype=float),
                          aggregate["prediction"].to_numpy(dtype=float))
        aggregate_rows.append({"group_type": group_name, "rows": int(len(aggregate)), **am})
    result["selected_aggregate_accuracy"] = aggregate_rows
    selected_gap = (best_naive_wape - selected_wape) / max(best_naive_wape, 1e-9)
    gap = (best_naive_wape - best_model_wape) / max(best_naive_wape, 1e-9)
    publishable = bool(selected_gap >= PUBLISHABLE_THRESHOLD)
    result["best_naive"] = {"model": best_naive, "wape": best_naive_wape}
    result["best_naive_validation_wape"] = float(validation_naive_wape)
    result["best_model"] = {"model": best_model_row["model"],
                            "wape": best_model_wape}
    result["best_model_is_diagnostic_test_best"] = True
    result["selected_model"] = {
        "model": selected_row["model"],
        "family": selected_row["family"],
        "is_baseline": bool(selected_row["model"] in NAIVE_BASELINES),
        "wape": selected_wape,
        "validation_wape": float(selected_row.get("validation_wape", float("nan"))),
        "improvement_vs_best_naive": float(selected_gap),
    }
    result["improvement_vs_previous_quarter"] = float(
        (prev_wape - best_model_wape) / max(prev_wape, 1e-9))
    result["publishable_candidate"] = publishable
    result["publishability_threshold"] = PUBLISHABLE_THRESHOLD
    if publishable:
        result["publishability_reason"] = (
            f"validation-selected {selected_row['model']} beats {best_naive} "
            f"naive by {selected_gap:.1%} WAPE on strict next-quarter test "
            f"(>=10% threshold)")
    else:
        result["publishability_reason"] = (
            f"validation-selected strict next-quarter model "
            f"{selected_row['model']} wape {selected_wape:.4f} vs best naive "
            f"{best_naive} {best_naive_wape:.4f}; relative gap "
            f"{selected_gap:.1%} below the 10% publishable threshold "
            f"(diagnostic test-best {best_model_row['model']} gap {gap:.1%})")
    return result


def quarterly_fold_cutoffs(panel: pd.DataFrame,
                           min_train_years: int = 4) -> List[int]:
    """Return validation-year cutoffs with train, validation, and test years."""
    view = build_quarterly_view(panel)
    years = sorted(int(y) for y in view["year"].dropna().unique())
    out = []
    for cutoff in years:
        train_years = [y for y in years if y <= cutoff - 1]
        val_years = [y for y in years if y == cutoff]
        test_years = [y for y in years if y > cutoff]
        if (len(train_years) >= min_train_years
                and len(val_years) == 1 and len(test_years) >= 1):
            out.append(cutoff)
    return out


def evaluate_quarterly_rolling(
    panel: pd.DataFrame,
    annual_panel_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    news_events_path: Optional[Path] = None,
    arcos_path: Optional[Path] = None,
    min_train_years: int = 4,
    test_window_years: Optional[int] = None,
    fluview_path: Optional[Path] = None,
    wastewater_path: Optional[Path] = None,
    nadac_path: Optional[Path] = None,
) -> Dict:
    """Run rolling-origin strict quarterly evaluations.

    Each fold uses feature years <= cutoff-1 for training, feature year
    ``cutoff`` for validation/blend selection, and all later feature years for
    testing. The per-fold evaluator is the same strict evaluator used for the
    single publishability check.
    """
    cutoffs = quarterly_fold_cutoffs(panel, min_train_years=min_train_years)
    rows: List[Dict] = []
    for cutoff in cutoffs:
        test_end_year = None
        if test_window_years is not None:
            test_end_year = cutoff + int(test_window_years)
        res = evaluate_quarterly(
            panel,
            train_cutoff=cutoff,
            annual_panel_path=annual_panel_path,
            events_path=events_path,
            news_events_path=news_events_path,
            arcos_path=arcos_path,
            test_end_year=test_end_year,
            fluview_path=fluview_path,
            wastewater_path=wastewater_path,
            nadac_path=nadac_path,
        )
        if "leaderboard" not in res:
            continue
        train_years = f"<= {cutoff - 1}"
        val_year = str(cutoff)
        test_years = (f"> {cutoff}" if test_end_year is None
                      else f"{cutoff + 1}-{test_end_year}")
        best_naive = res.get("best_naive", {})
        best_model = res.get("best_model", {})
        selected_model = res.get("selected_model", best_model)
        naive_wape = float(best_naive.get("wape", float("nan")))
        model_wape = float(best_model.get("wape", float("nan")))
        selected_wape = float(selected_model.get("wape", float("nan")))
        selected_lb = res["leaderboard"]
        selected_accuracy_row = selected_lb[
            selected_lb["model"].eq(selected_model.get("model"))].iloc[0]
        persistence_row = selected_lb[
            selected_lb["model"].eq("previous_quarter")].iloc[0]
        diagnostic_improvement = (
            (naive_wape - model_wape) / max(naive_wape, 1e-9)
            if np.isfinite(naive_wape) and np.isfinite(model_wape)
            else float("nan"))
        selected_improvement = (
            (naive_wape - selected_wape) / max(naive_wape, 1e-9)
            if np.isfinite(naive_wape) and np.isfinite(selected_wape)
            else float("nan"))
        rows.append({
            "train_years": train_years,
            "validation_year": val_year,
            "test_years": test_years,
            "train_rows": res.get("n_rows", {}).get("train", 0),
            "validation_rows": res.get("n_rows", {}).get("validation", 0),
            "test_rows": res.get("n_rows", {}).get("test", 0),
            "best_naive": best_naive.get("model"),
            "best_naive_wape": naive_wape,
            "best_model_by_test": best_model.get("model"),
            "best_model_by_test_wape": model_wape,
            "diagnostic_improvement_vs_best_naive": float(diagnostic_improvement),
            "selected_model": selected_model.get("model"),
            "selected_model_wape": selected_wape,
            "selected_model_validation_wape": selected_model.get("validation_wape"),
            "selected_within_5pct": float(selected_accuracy_row.get("within_5pct", float("nan"))),
            "selected_within_10pct": float(selected_accuracy_row.get("within_10pct", float("nan"))),
            "selected_within_20pct": float(selected_accuracy_row.get("within_20pct", float("nan"))),
            "selected_demand_state_accuracy": float(
                selected_accuracy_row.get("demand_state_accuracy", float("nan"))),
            "selected_demand_state_balanced_accuracy": float(
                selected_accuracy_row.get("demand_state_balanced_accuracy", float("nan"))),
            "selected_demand_state_macro_f1": float(
                selected_accuracy_row.get("demand_state_macro_f1", float("nan"))),
            "selected_demand_state_decrease_rows": int(
                selected_accuracy_row.get("demand_state_decrease_rows", 0)),
            "selected_demand_state_stable_rows": int(
                selected_accuracy_row.get("demand_state_stable_rows", 0)),
            "selected_demand_state_increase_rows": int(
                selected_accuracy_row.get("demand_state_increase_rows", 0)),
            "persistence_demand_state_accuracy": float(
                persistence_row.get("demand_state_accuracy", float("nan"))),
            "persistence_demand_state_balanced_accuracy": float(
                persistence_row.get("demand_state_balanced_accuracy", float("nan"))),
            "persistence_demand_state_macro_f1": float(
                persistence_row.get("demand_state_macro_f1", float("nan"))),
            "selected_improvement_vs_best_naive": float(selected_improvement),
            "clears_10pct_gate": bool(selected_improvement >= PUBLISHABLE_THRESHOLD),
            "beats_naive": bool(selected_improvement > 0),
            "exogenous_feature_count": res.get("exogenous_feature_count", 0),
        })

    folds = pd.DataFrame(rows)
    summary: Dict = {
        "split": "rolling_strict_next_quarter_time",
        "min_train_years": int(min_train_years),
        "test_window_years": (None if test_window_years is None
                              else int(test_window_years)),
        "publishability_threshold": PUBLISHABLE_THRESHOLD,
        "fold_count": int(len(folds)),
        "folds": folds,
    }
    if folds.empty:
        summary.update({
            "publishable_rolling_candidate": False,
            "publishability_reason": "no eligible rolling folds",
        })
        return summary

    imp = folds["selected_improvement_vs_best_naive"].to_numpy(dtype=float)
    finite = imp[np.isfinite(imp)]
    mean_imp = float(np.mean(finite)) if finite.size else float("nan")
    median_imp = float(np.median(finite)) if finite.size else float("nan")
    min_imp = float(np.min(finite)) if finite.size else float("nan")
    n_clear = int(np.sum(finite >= PUBLISHABLE_THRESHOLD))
    n_beat = int(np.sum(finite > 0))
    all_beat = bool(finite.size == len(folds) and n_beat == len(folds))
    rolling_publishable = bool(
        all_beat and np.isfinite(mean_imp) and mean_imp >= PUBLISHABLE_THRESHOLD)
    mean_within_5pct = float(folds["selected_within_5pct"].mean())
    mean_state_accuracy = float(folds["selected_demand_state_accuracy"].mean())
    mean_state_balanced = float(
        folds["selected_demand_state_balanced_accuracy"].mean())
    mean_persistence_state_accuracy = float(
        folds["persistence_demand_state_accuracy"].mean())
    mean_persistence_state_balanced = float(
        folds["persistence_demand_state_balanced_accuracy"].mean())
    numeric_goal = bool(
        np.isfinite(mean_within_5pct)
        and mean_within_5pct >= REQUESTED_ACCURACY_THRESHOLD)
    boolean_goal = bool(
        np.isfinite(mean_state_accuracy)
        and np.isfinite(mean_state_balanced)
        and mean_state_accuracy >= REQUESTED_ACCURACY_THRESHOLD
        and mean_state_balanced >= REQUESTED_ACCURACY_THRESHOLD)
    requested_accuracy_goal_met = bool(numeric_goal or boolean_goal)
    summary.update({
        "mean_improvement_vs_best_naive": mean_imp,
        "median_improvement_vs_best_naive": median_imp,
        "min_improvement_vs_best_naive": min_imp,
        "folds_clearing_10pct_gate": n_clear,
        "fraction_folds_clearing_10pct_gate": float(n_clear / len(folds)),
        "folds_beating_naive": n_beat,
        "fraction_folds_beating_naive": float(n_beat / len(folds)),
        "mean_selected_within_5pct": mean_within_5pct,
        "mean_selected_within_10pct": float(folds["selected_within_10pct"].mean()),
        "mean_selected_within_20pct": float(folds["selected_within_20pct"].mean()),
        "mean_selected_demand_state_accuracy": mean_state_accuracy,
        "mean_selected_demand_state_balanced_accuracy": mean_state_balanced,
        "mean_selected_demand_state_macro_f1": float(
            folds["selected_demand_state_macro_f1"].mean()),
        "mean_persistence_demand_state_accuracy": mean_persistence_state_accuracy,
        "mean_persistence_demand_state_balanced_accuracy": mean_persistence_state_balanced,
        "mean_state_balanced_improvement_vs_persistence": float(
            mean_state_balanced - mean_persistence_state_balanced),
        "publishable_rolling_candidate": rolling_publishable,
        "requested_accuracy_threshold": REQUESTED_ACCURACY_THRESHOLD,
        "numeric_within_5pct_goal_met": numeric_goal,
        "boolean_state_goal_met": boolean_goal,
        "requested_accuracy_goal_met": requested_accuracy_goal_met,
        "promotion_candidate": bool(rolling_publishable and requested_accuracy_goal_met),
    })
    if rolling_publishable:
        summary["publishability_reason"] = (
            f"all {len(folds)} rolling folds beat their strongest naive "
            f"baseline and mean improvement {mean_imp:.1%} is >=10%")
    else:
        summary["publishability_reason"] = (
            f"rolling evidence below gate: {n_beat}/{len(folds)} folds beat "
            f"naive, {n_clear}/{len(folds)} clear 10%, mean improvement "
            f"{mean_imp:.1%}")
    summary["promotion_reason"] = (
        "rolling WAPE gate and requested 75% accuracy goal both pass"
        if summary["promotion_candidate"] else
        "research-only: WAPE gate may pass, but requested 75% numeric/state "
        "accuracy goal is not met")
    return summary


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    panel = load_medicaid_sdud_panel(root / "data/S_D/data/combined")
    res = evaluate_quarterly(panel)
    print(f"rows: {res.get('n_rows')}")
    lb = res["leaderboard"]
    print(lb[["model", "wape", "improvement_vs_previous_quarter"]]
          .to_string(index=False))
    print(f"publishable_candidate: {res['publishable_candidate']}")
    print(res.get("publishability_reason", ""))
