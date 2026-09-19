"""Annual Arkansas drug-demand panel builder.

Builds the panel from CMS Part D provider-drug data aggregated by
(year, prescriber city, generic drug), maps generic names to FDA NDC drug
identity attributes, and joins annual aggregates of the external-state feature
store plus disease, news, supply, and provider layers. All values derive from
real local files; no synthetic data.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import io
from .entities import (
    build_drug_identity,
    build_therapeutic_mapping,
    normalize_name,
    project_supplier_exposure,
)
from .layers import (
    build_disease_annual,
    build_labeler_annual,
    build_supply_annual,
    build_supply_drug,
)
from .text_signals import build_text_features, pivot_annual, pivot_drug
from .news_signals import ALL_NEWS_SIGNAL_COLUMNS, build_news_state_features
from .news_only_adapter import build_news_only_annual_features

PANEL_COLUMNS = [
    "year", "city", "drug", "drug_key",
    "demand_claims", "demand_fills", "demand_cost", "demand_benes",
    "n_provider_types", "cost_per_fill",
    "demand_claims_lag1", "demand_claims_lag2", "demand_claims_lag3",
    "demand_claims_ma2", "demand_claims_ma3",
    "demand_claims_delta", "demand_claims_delta2",
]

EXTERNAL_FEATURE_COLUMNS = [
    "na_shortage_active_mean", "na_recall_active_mean",
    "ar_ili_mean", "ar_wili_mean", "nat_ili_mean", "nat_wili_mean",
    "ar_unemployment_mean", "na_ppi_mean", "global_gscpi_mean",
    "ar_disaster_active_mean", "ar_disaster_severity_max",
    "us_tariff_rate_mean", "ar_population_total",
    "pharmacy_facility_count", "pharmacy_hospital_count",
    "pharmacy_retail_count", "arcos_distribution_grams",
    "arcos_distribution_zip3_count", "arcos_distribution_growth",
    "ar_humidity_mean", "ar_temperature_max_mean", "ar_temperature_min_mean",
    "ar_population_growth_abs", "ar_births", "ar_deaths", "ar_net_migration",
    "global_container_throughput_mean",
]

DRUG_IDENTITY_COLUMNS = [
    "ingredient", "labeler", "dosage_form", "route", "market_cat",
    "dea_schedule", "pharm_class", "therapeutic_cat",
    "product_count", "ingredient_products", "labeler_products",
]

SUPPLY_COLUMNS = [
    "shortage_active", "shortage_events", "recall_count_x", "recall_count_y",
    "shortage_active_total", "shortage_current_total", "recall_firms",
    "shortage_reason_demand", "shortage_reason_discontinuation",
    "shortage_reason_ingredient", "shortage_reason_other",
    "recall_class_1", "recall_class_2", "recall_class_3",
    "recall_kw_sterility", "recall_kw_cgmp", "recall_kw_potency",
    "recall_kw_mislabel", "recall_kw_contamination",
    "recall_class1", "recall_class2", "recall_class3",
    "labeler_shortage_n", "labeler_recall_n",
]

MEDICAID_QUARTERLY_COLUMNS = [
    "medicaid_rx_annual", "medicaid_rx_q4", "medicaid_rx_recent_qoq",
    "medicaid_rx_q4_share", "medicaid_rx_log", "medicaid_rx_q4_log",
    "medicaid_rx_quarters_observed", "medicaid_rx_bridge_records",
    "medicaid_rx_bridge_families",
]

MEDICAID_EXACT_COLUMNS = [
    "medicaid_exact_rx_annual", "medicaid_exact_rx_q4",
    "medicaid_exact_rx_recent_qoq", "medicaid_exact_rx_q4_share",
    "medicaid_exact_rx_log", "medicaid_exact_rx_q4_log",
    "medicaid_exact_rx_quarters_observed", "medicaid_exact_rx_bridge_records",
    "medicaid_exact_rx_bridge_families",
]


def build_layer1_news_annual(events: pd.DataFrame,
                              relevance_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate the canonical Layer-1 news states for next-period features."""
    if events.empty:
        return pd.DataFrame(columns=["year", *ALL_NEWS_SIGNAL_COLUMNS])
    layer1 = build_news_state_features(events, relevance_scores=relevance_scores)
    if layer1.empty:
        return pd.DataFrame(columns=["year", *ALL_NEWS_SIGNAL_COLUMNS])
    layer1 = layer1.copy()
    layer1["year"] = pd.to_datetime(layer1["observation_date"], errors="coerce").dt.year
    layer1 = layer1.dropna(subset=["year"])
    agg = {
        col: ("mean" if col.endswith(("_outbreak_risk", "_spread_rate")) else "sum")
        for col in ALL_NEWS_SIGNAL_COLUMNS
    }
    layer1 = layer1.groupby("year", as_index=False).agg(agg).copy()
    layer1["year"] = layer1["year"].astype(int)
    return layer1[["year", *ALL_NEWS_SIGNAL_COLUMNS]]

MEDICAID_ALL_COLUMNS = MEDICAID_EXACT_COLUMNS + MEDICAID_QUARTERLY_COLUMNS


def _digits(value: object) -> str:
    import re
    return re.sub(r"[^0-9]+", "", str(value or ""))


def _ndc_product_candidates(ndc: object) -> List[str]:
    """Possible FDA product NDC forms for an 11-digit package NDC."""
    d = _digits(ndc)
    if len(d) != 11:
        return []
    candidates = {
        f"{d[:5]}-{d[5:9]}",
        f"{d[:4]}-{d[4:8]}",
        f"{d[:5]}-{d[5:8]}",
        f"{int(d[:5])}-{d[5:9]}",
        f"{int(d[:4])}-{d[4:8]}",
        f"{int(d[:5])}-{d[5:8]}",
    }
    return sorted(c for c in candidates if c and not c.startswith("-"))


def build_external_annual_features(cfg) -> pd.DataFrame:
    """Annual external feature aggregates from external_state_features.csv.gz."""
    df = io.load_csv(
        cfg.data_path(cfg.external_features),
        usecols=[2, 3, 4, 6, 7, 8, 10],  # domain,variable_id,value,geo_level,geo_id,geo_name,period_end
    )
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["year"] = df["period_end"].dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    rows: List[dict] = []

    def _annual(name: str, sub: pd.DataFrame, agg: str) -> None:
        sub = sub.drop_duplicates(subset=["variable_id", "period_end"])
        grouped = sub.groupby("year")["value"]
        values = getattr(grouped, agg)()
        for year, v in values.items():
            rows.append({"year": int(year), "variable_id": name, "value": float(v)})

    _annual("na_shortage_active_mean",
            df[(df["variable_id"] == "shortage_active") & (df["geography_level"] == "national")],
            "mean")
    _annual("na_recall_active_mean",
            df[(df["variable_id"] == "recall_active") & (df["geography_level"] == "national")],
            "mean")
    _annual("ar_ili_mean",
            df[(df["variable_id"] == "ili") & (df["geography_id"] == "ar")], "mean")
    _annual("ar_wili_mean",
            df[(df["variable_id"] == "wili") & (df["geography_id"] == "ar")], "mean")
    _annual("ar_unemployment_mean",
            df[(df["variable_id"] == "arkansas_unemployment_rate") & (df["geography_id"] == "AR")],
            "mean")
    _annual("na_ppi_mean",
            df[(df["variable_id"] == "pharma_producer_price_index") & (df["geography_level"] == "national")],
            "mean")
    _annual("global_gscpi_mean",
            df[(df["variable_id"] == "gscpi") & (df["geography_level"] == "global")], "mean")
    _annual("us_tariff_rate_mean",
            df[(df["variable_id"] == "pharmaceutical_tariff_rate") & (df["geography_level"] == "country")],
            "mean")
    _annual("ar_disaster_active_mean",
            df[(df["variable_id"] == "disaster_active") & (df["geography_level"] == "county_or_area")],
            "mean")

    # NOAA station rows are retained only when the public station name
    # explicitly identifies Arkansas.  This avoids silently treating national
    # station observations as Arkansas weather exposure.
    ar_weather = df[
        df["geography_level"].eq("station")
        & df["geography_name"].fillna("").astype(str).str.contains(", AR US", regex=False)
        & df["variable_id"].isin({"temperature_mean", "precipitation", "wind"})
    ].copy()
    if not ar_weather.empty:
        ar_weather = ar_weather.drop_duplicates(
            subset=["variable_id", "geography_id", "period_end"])
        for variable, output in {
            "temperature_mean": "ar_temperature_mean",
            "precipitation": "ar_precipitation_mean",
            "wind": "ar_wind_mean",
        }.items():
            values = ar_weather[ar_weather["variable_id"].eq(variable)]
            for year, value in values.groupby("year")["value"].mean().items():
                rows.append({"year": int(year), "variable_id": output,
                             "value": float(value)})

        for variable, output in {
            "humidity": "ar_humidity_mean",
            "temperature_max": "ar_temperature_max_mean",
            "temperature_min": "ar_temperature_min_mean",
        }.items():
            values = df[(df["geography_level"] == "station")
                        & df["geography_name"].fillna("").astype(str).str.contains(
                            ", AR US", regex=False)
                        & df["variable_id"].eq(variable)]
            values = values.drop_duplicates(subset=["variable_id", "geography_id", "period_end"])
            for year, value in values.groupby("year")["value"].mean().items():
                rows.append({"year": int(year), "variable_id": output,
                             "value": float(value)})

    dis_sev = df[(df["variable_id"] == "disaster_severity") & (df["geography_level"] == "county_or_area")]
    dis_sev = dis_sev.drop_duplicates(subset=["variable_id", "period_end"])
    for year, v in dis_sev.groupby("year")["value"].max().items():
        rows.append({"year": int(year), "variable_id": "ar_disaster_severity_max", "value": float(v)})

    pop = df[(df["variable_id"] == "population_total")
             & (df["geography_level"] == "county")
             & (df["geography_id"].astype(str).str.startswith("05"))]
    pop = pop.drop_duplicates(subset=["geography_id", "period_end"])
    for year, v in pop.groupby("year")["value"].sum().items():
        rows.append({"year": int(year), "variable_id": "ar_population_total", "value": float(v)})

    for source, output in {
        "population_growth_abs": "ar_population_growth_abs",
        "births": "ar_births", "deaths": "ar_deaths",
        "net_migration": "ar_net_migration",
    }.items():
        county = df[(df["variable_id"] == source)
                    & df["geography_level"].isin(["county", "region"])
                    & df["geography_id"].astype(str).str.startswith("05")]
        county = county.drop_duplicates(subset=["variable_id", "geography_id", "period_end"])
        for year, value in county.groupby("year")["value"].sum().items():
            rows.append({"year": int(year), "variable_id": output, "value": float(value)})

    for source in ("north_range_original", "north_range_seasonally_adjusted",
                   "north_range_trend_cycle"):
        container = df[(df["variable_id"] == source)
                       & (df["geography_level"] == "global")]
        for year, value in container.groupby("year")["value"].mean().items():
            rows.append({"year": int(year), "variable_id": "global_container_throughput_mean",
                         "value": float(value)})

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(columns=["year"])
    wide = long.pivot_table(index="year", columns="variable_id", values="value").reset_index()
    wide.columns.name = None
    wide["year"] = wide["year"].astype(int)
    return wide


def build_arcos_annual_features(cfg) -> pd.DataFrame:
    """Build drug-level Arkansas ARCOS distribution context by year.

    ARCOS records controlled-substance distribution into Arkansas ZIP3s. It is
    an upstream distribution proxy, not pharmacy dispensing or on-hand
    inventory. Growth is computed only from prior observed years.
    """
    path = cfg.data_path(cfg.arcos_retail_summary)
    if not path.exists():
        return pd.DataFrame(columns=["year", "drug_key"])
    frame = io.load_csv(path)
    required = {"year", "drug_name", "zip3", "total_grams"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "drug_key"])
    frame = frame.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["total_grams"] = pd.to_numeric(
        frame["total_grams"].astype(str).str.replace(",", "", regex=False),
        errors="coerce")
    frame["drug_key"] = frame["drug_name"].map(normalize_name)
    frame = frame.dropna(subset=["year", "total_grams"])
    frame = frame[frame["drug_key"].ne("")]
    if frame.empty:
        return pd.DataFrame(columns=["year", "drug_key"])
    out = (frame.groupby(["year", "drug_key"], as_index=False)
           .agg(arcos_distribution_grams=("total_grams", "sum"),
                arcos_distribution_zip3_count=("zip3", "nunique")))
    out = out.sort_values(["drug_key", "year"])
    previous = out.groupby("drug_key")["arcos_distribution_grams"].shift(1)
    out["arcos_distribution_growth"] = (
        out["arcos_distribution_grams"] / previous.replace(0, np.nan) - 1.0)
    return out.reset_index(drop=True)


def build_pharmacy_access_features(cfg) -> pd.DataFrame:
    """Aggregate real Arkansas pharmacy-directory facilities by city/year."""
    path = (cfg.data_dir / "targeted_additions" / "arkansas_pharmacy_roster"
            / "data" / "arkansas_pharmacy_facilities.csv.gz")
    if not path.exists():
        return pd.DataFrame(columns=["year", "city_key",
                                     "pharmacy_facility_count",
                                     "pharmacy_hospital_count",
                                     "pharmacy_retail_count"])
    roster = io.load_csv(path, usecols=["year", "city", "section", "license_number"])
    roster["year"] = pd.to_numeric(roster["year"], errors="coerce")
    roster = roster.dropna(subset=["year", "city"])
    roster["year"] = roster["year"].astype(int)
    roster["city_key"] = roster["city"].map(normalize_name)
    roster = roster[roster["city_key"] != ""]
    roster["section"] = roster["section"].fillna("").astype(str).str.lower()
    return roster.groupby(["year", "city_key"], as_index=False).agg(
        pharmacy_facility_count=("license_number", "nunique"),
        pharmacy_hospital_count=("section", lambda s: int(s.str.contains("hospital").sum())),
        pharmacy_retail_count=("section", lambda s: int(s.str.contains("retail|community", regex=True).sum())),
    )


def _load_medicaid_quarterly_source(cfg) -> pd.DataFrame:
    """Load real Arkansas Medicaid quarterly prescription counts."""
    import gzip
    import json

    rows: List[dict] = []
    combined = cfg.data_dir / "S_D" / "data" / "combined"
    for path in sorted(combined.glob("20*.json.gz")):
        with gzip.open(path, "rt") as fh:
            recs = json.load(fh)
        for r in recs:
            if r.get("source", {}).get("source_id") != "medicaid_sdud":
                continue
            if r.get("geography", {}).get("admin1") != "AR":
                continue
            obs = r.get("observation", {})
            if obs.get("metric") != "prescription_count":
                continue
            value = obs.get("value")
            if value is None:
                continue
            start = str(r.get("period_start") or "")
            if len(start) < 7:
                continue
            drug = r.get("drug", {}).get("ingredient") \
                or r.get("drug", {}).get("canonical_name") \
                or r.get("drug", {}).get("original_name")
            if not drug:
                continue
            rows.append({
                "record_id": r.get("record_id") or r.get("source", {}).get("source_record_id"),
                "year": int(start[:4]),
                "quarter": (int(start[5:7]) - 1) // 3 + 1,
                "drug": drug,
                "canonical_name": r.get("drug", {}).get("canonical_name") or "",
                "ingredient": r.get("drug", {}).get("ingredient") or "",
                "original_name": r.get("drug", {}).get("original_name") or "",
                "rxnorm_rxcui": r.get("drug", {}).get("rxnorm_rxcui") or "",
                "ndc": r.get("drug", {}).get("ndc") or "",
                "value": float(value),
            })
    if rows:
        return pd.DataFrame(rows)

    cache = cfg.panel_dir / "medicaid_sdud_quarterly_panel.csv"
    if cache.exists():
        out = io.load_csv(cache)
        out["record_id"] = (
            "cache:" + out["year"].astype(str) + ":" + out["quarter"].astype(str)
            + ":" + out["drug"].astype(str)
        )
        return out
    return pd.DataFrame(rows)


def _relationship_drug_maps(cfg) -> tuple:
    """NDC/drug relationships from final_data entity graph."""
    path = cfg.data_path(cfg.relationships)
    if not path.exists():
        return {}, {}
    rel = io.load_csv(
        path,
        usecols=["source_entity_id", "relationship_type", "target_entity_id"],
    )
    ndc_to_drug = {}
    contains = {}
    for _, row in rel.iterrows():
        src = str(row["source_entity_id"])
        tgt = str(row["target_entity_id"])
        typ = str(row["relationship_type"])
        if typ == "represents" and src.startswith("ndc:") and tgt.startswith("drug:"):
            ndc_to_drug[_digits(src[4:])] = normalize_name(tgt[5:])
        elif typ == "contains" and src.startswith("drug:") and tgt.startswith("ingredient:"):
            drug = normalize_name(src[5:])
            ing = normalize_name(tgt[11:])
            if drug and ing:
                contains.setdefault(drug, set()).add(ing)
    drug_to_ingredient = {k: sorted(v)[0] for k, v in contains.items() if v}
    return ndc_to_drug, drug_to_ingredient


def _fda_product_ndc_map(cfg) -> dict:
    """FDA product NDC -> normalized nonproprietary/substance keys."""
    path = cfg.data_path(cfg.fda_ndc_products)
    if not path.exists():
        return {}
    fda = io.load_csv(
        path,
        usecols=["PRODUCTNDC", "NONPROPRIETARYNAME", "SUBSTANCENAME"],
    )
    out = {}
    for _, row in fda.iterrows():
        keys = set()
        for col in ("NONPROPRIETARYNAME", "SUBSTANCENAME"):
            raw = row.get(col)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            for part in str(raw).split(";"):
                key = normalize_name(part)
                if key:
                    keys.add(key)
        if keys:
            out[str(row["PRODUCTNDC"]).strip()] = sorted(keys)
    return out


def _dictionary_maps(cfg) -> tuple:
    """Local drug dictionary maps keyed by NDC and RxCUI."""
    path = cfg.data_path(cfg.drug_dictionary)
    if not path.exists():
        return {}, {}
    import json
    rows = json.load(open(path))
    by_ndc = {}
    by_rxcui = {}
    for row in rows:
        keys = set()
        for col in ("ingredient", "canonical_name", "original_name"):
            key = normalize_name(row.get(col))
            if key:
                keys.add(key)
        if not keys:
            continue
        ndc = _digits(row.get("ndc"))
        if ndc:
            by_ndc.setdefault(ndc, set()).update(keys)
        rxcui = str(row.get("rxnorm_rxcui") or "").strip()
        if rxcui:
            by_rxcui.setdefault(rxcui, set()).update(keys)
    return ({k: sorted(v) for k, v in by_ndc.items()},
            {k: sorted(v) for k, v in by_rxcui.items()})


def _annualize_medicaid_bridge(bridge: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate deduplicated Medicaid record-key rows into annual features."""
    if bridge.empty:
        cols = ["year", "drug_key"] + [
            f"{prefix}_rx_annual", f"{prefix}_rx_q4",
            f"{prefix}_rx_recent_qoq", f"{prefix}_rx_q4_share",
            f"{prefix}_rx_log", f"{prefix}_rx_q4_log",
            f"{prefix}_rx_quarters_observed", f"{prefix}_rx_bridge_records",
            f"{prefix}_rx_bridge_families",
        ]
        return pd.DataFrame(columns=cols)
    annual = (
        bridge.groupby(["year", "drug_key"], as_index=False)
        .agg(**{
            f"{prefix}_rx_annual": ("value", "sum"),
            f"{prefix}_rx_quarters_observed": ("quarter", "nunique"),
            f"{prefix}_rx_bridge_records": ("record_id", "nunique"),
            f"{prefix}_rx_bridge_families": ("bridge_family", "nunique"),
        })
    )
    by_q = (bridge.groupby(["year", "drug_key", "quarter"], as_index=False)["value"]
            .sum())
    q3 = by_q[by_q["quarter"] == 3][["year", "drug_key", "value"]].rename(
        columns={"value": f"{prefix}_rx_q3"})
    q4 = by_q[by_q["quarter"] == 4][["year", "drug_key", "value"]].rename(
        columns={"value": f"{prefix}_rx_q4"})
    out = annual.merge(q3, on=["year", "drug_key"], how="left")
    out = out.merge(q4, on=["year", "drug_key"], how="left")
    out[f"{prefix}_rx_recent_qoq"] = (
        (out[f"{prefix}_rx_q4"] - out[f"{prefix}_rx_q3"])
        / out[f"{prefix}_rx_q3"].replace(0, pd.NA)
    )
    out[f"{prefix}_rx_q4_share"] = (
        out[f"{prefix}_rx_q4"] / out[f"{prefix}_rx_annual"].replace(0, pd.NA)
    )
    out[f"{prefix}_rx_log"] = np.log1p(out[f"{prefix}_rx_annual"].clip(lower=0))
    out[f"{prefix}_rx_q4_log"] = np.log1p(out[f"{prefix}_rx_q4"].clip(lower=0))
    out = out.drop(columns=[f"{prefix}_rx_q3"])
    return out.sort_values(["drug_key", "year"]).reset_index(drop=True)


def build_medicaid_quarterly_annual_features(cfg) -> pd.DataFrame:
    """Annualize real Arkansas Medicaid SDUD quarterly demand by drug.

    The feature year is the same calendar year as the annual Part D feature
    row. In strict next-period evaluation, year t features still predict year
    t+1 demand, so these values do not include target-year information.
    """
    q = _load_medicaid_quarterly_source(cfg)
    if q.empty:
        return pd.DataFrame(columns=["year", "drug_key"] + MEDICAID_ALL_COLUMNS)
    required = {"year", "quarter", "drug", "value", "record_id"}
    if not required <= set(q.columns):
        return pd.DataFrame(columns=["year", "drug_key"] + MEDICAID_ALL_COLUMNS)
    keep = [c for c in [
        "record_id", "year", "quarter", "drug", "canonical_name", "ingredient",
        "original_name", "rxnorm_rxcui", "ndc", "value",
    ] if c in q.columns]
    q = q[keep].copy()
    q["year"] = pd.to_numeric(q["year"], errors="coerce")
    q["quarter"] = pd.to_numeric(q["quarter"], errors="coerce")
    q["value"] = pd.to_numeric(q["value"], errors="coerce")
    q = q.dropna(subset=["year", "quarter", "drug", "value", "record_id"])
    q["year"] = q["year"].astype(int)
    q["quarter"] = q["quarter"].astype(int)
    q["record_id"] = q["record_id"].astype(str)
    q = q[q["value"] >= 0]
    if q.empty:
        return pd.DataFrame(columns=["year", "drug_key"] + MEDICAID_ALL_COLUMNS)

    ndc_to_drug, drug_to_ingredient = _relationship_drug_maps(cfg)
    fda_product = _fda_product_ndc_map(cfg)
    dict_ndc, dict_rxcui = _dictionary_maps(cfg)

    bridge_rows: List[dict] = []
    for _, row in q.iterrows():
        keys = []

        def add_key(key: object, family: str) -> None:
            k = normalize_name(key)
            if k:
                keys.append((k, family))

        for col in ("drug", "canonical_name", "ingredient", "original_name"):
            if col in q.columns:
                add_key(row.get(col), f"text_{col}")

        ndc = _digits(row.get("ndc"))
        if ndc:
            rel_drug = ndc_to_drug.get(ndc)
            if rel_drug:
                add_key(rel_drug, "relationship_ndc_drug")
                add_key(drug_to_ingredient.get(rel_drug, ""), "relationship_ndc_ingredient")
            for key in dict_ndc.get(ndc, []):
                add_key(key, "dictionary_ndc")
            for product in _ndc_product_candidates(ndc):
                for key in fda_product.get(product, []):
                    add_key(key, "fda_product_ndc")

        rxcui = str(row.get("rxnorm_rxcui") or "").strip()
        if rxcui:
            for key in dict_rxcui.get(rxcui, []):
                add_key(key, "dictionary_rxcui")

        seen = set()
        for key, family in keys:
            pair = (key, family)
            if pair in seen:
                continue
            seen.add(pair)
            bridge_rows.append({
                "record_id": row["record_id"],
                "year": int(row["year"]),
                "quarter": int(row["quarter"]),
                "drug_key": key,
                "bridge_family": family,
                "value": float(row["value"]),
            })

    bridge = pd.DataFrame(bridge_rows)
    if bridge.empty:
        return pd.DataFrame(
            columns=["year", "drug_key"] + MEDICAID_ALL_COLUMNS)
    bridge = bridge.drop_duplicates(subset=["record_id", "drug_key"])
    exact_families = {
        "text_drug", "text_canonical_name", "text_ingredient", "text_original_name",
    }
    exact = bridge[bridge["bridge_family"].isin(exact_families)].copy()
    exact = exact.drop_duplicates(subset=["record_id", "drug_key"])
    exact_out = _annualize_medicaid_bridge(exact, "medicaid_exact")
    bridge_out = _annualize_medicaid_bridge(bridge, "medicaid").rename(columns={
        "medicaid_rx_annual": "medicaid_rx_annual",
    })
    out = exact_out.merge(bridge_out, on=["year", "drug_key"], how="outer")
    cols = ["year", "drug_key"] + MEDICAID_ALL_COLUMNS
    return out[cols].sort_values(["drug_key", "year"]).reset_index(drop=True)


def build_panel(cfg) -> pd.DataFrame:
    """Build the annual city x drug demand panel with external layers."""
    partd = io.load_csv(
        cfg.data_path(cfg.partd_provider_drug),
        usecols=["Prscrbr_City", "Gnrc_Name", "Prscrbr_Type", "year",
                 "Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Benes"],
    )
    partd = partd.rename(
        columns={
            "Prscrbr_City": "city",
            "Gnrc_Name": "drug",
            "Prscrbr_Type": "n_provider_types",
            "Tot_Clms": "demand_claims",
            "Tot_30day_Fills": "demand_fills",
            "Tot_Drug_Cst": "demand_cost",
            "Tot_Benes": "demand_benes",
        }
    )
    partd["drug"] = partd["drug"].astype(str).str.strip()
    partd = partd[
        partd["drug"].notna()
        & partd["city"].notna()
        & (partd["city"].astype(str).str.len() > 0)
    ]
    partd["drug_key"] = partd["drug"].map(normalize_name)
    partd = partd[partd["drug_key"] != ""]

    agg = {
        "demand_claims": "sum", "demand_fills": "sum", "demand_cost": "sum",
        "demand_benes": "sum",
        "n_provider_types": "nunique",
    }
    panel = (
        partd.groupby(["year", "city", "drug", "drug_key"], as_index=False).agg(agg)
    )
    panel["year"] = panel["year"].astype(int)
    panel["demand_benes"] = panel["demand_benes"].replace(0, pd.NA)
    panel["cost_per_fill"] = panel["demand_cost"] / panel["demand_fills"].replace(0, pd.NA)
    panel = panel.sort_values(["drug_key", "city", "year"]).reset_index(drop=True)

    # Real pharmacy-access exposure at the prescriber-city/year level.
    access = build_pharmacy_access_features(cfg)
    if not access.empty:
        panel["city_key"] = panel["city"].map(normalize_name)
        panel = panel.merge(access, on=["year", "city_key"], how="left")
        panel = panel.drop(columns=["city_key"])

    # --- drug identity mapping ------------------------------------------------
    identity = build_drug_identity(cfg)
    therapeutic = build_therapeutic_mapping(cfg)
    panel = project_supplier_exposure(panel, identity, therapeutic)

    # --- external annual features ---------------------------------------------
    external = build_external_annual_features(cfg)
    if not external.empty:
        panel = panel.merge(external, on="year", how="left")

    arcos = build_arcos_annual_features(cfg)
    if not arcos.empty:
        panel = panel.merge(arcos, on=["year", "drug_key"], how="left")

    # --- disease / supply / labeler layers ------------------------------------
    disease = build_disease_annual(cfg)
    if not disease.empty:
        panel = panel.merge(disease, on="year", how="left")
    supply_annual = build_supply_annual(cfg)
    if not supply_annual.empty:
        panel = panel.merge(supply_annual, on="year", how="left")
    supply_drug = build_supply_drug(cfg)
    if not supply_drug.empty:
        supply_drug = supply_drug.rename(columns={"drug": "drug_key"})
        panel = panel.merge(supply_drug, on=["year", "drug_key"], how="left")
    labeler_annual = build_labeler_annual(cfg)
    if not labeler_annual.empty:
        panel = panel.merge(labeler_annual, on=["year", "labeler"], how="left")

    # --- real Arkansas Medicaid quarterly demand pressure ---------------------
    medicaid = build_medicaid_quarterly_annual_features(cfg)
    if not medicaid.empty:
        panel = panel.merge(medicaid, on=["year", "drug_key"], how="left")
        if "ingredient" in panel.columns:
            fallback = medicaid.rename(columns={"drug_key": "ingredient_key"})
            panel["ingredient_key"] = panel["ingredient"].map(normalize_name)
            panel = panel.merge(
                fallback, on=["year", "ingredient_key"], how="left",
                suffixes=("", "_ingredient_match"))
            for col in MEDICAID_ALL_COLUMNS:
                fb = f"{col}_ingredient_match"
                if fb in panel.columns:
                    panel[col] = panel[col].combine_first(panel[fb])
            panel = panel.drop(
                columns=[c for c in panel.columns
                         if c == "ingredient_key" or c.endswith("_ingredient_match")])

    # --- deterministic text/event features ------------------------------------
    # Layer 1 is an upstream input to the predictive panel, not a detached
    # reporting artifact. Aggregate only observed article states by year; the
    # strict next-period evaluator then uses year t states for the year t+1
    # target. Risk/spread fields are means, while boolean/count states are
    # additive exposure measures.
    layer1_events = cfg.artifact_path("events/article_events.csv.gz")
    if layer1_events.exists():
        relevance_path = cfg.artifact_path("news/relevance_scores.csv.gz")
        relevance = io.load_csv(relevance_path) if relevance_path.exists() else None
        layer1 = build_layer1_news_annual(io.load_csv(layer1_events), relevance)
        if not layer1.empty:
            panel = panel.merge(layer1, on="year", how="left")

    annual_events, drug_events = build_text_features(cfg)
    event_wide = pivot_annual(annual_events)
    if not event_wide.empty:
        panel = panel.merge(event_wide, on="year", how="left")

    drug_wide = pivot_drug(drug_events)
    if not drug_wide.empty:
        drug_wide = drug_wide.rename(columns={"drug": "drug_key"})
        drug_wide["drug_key"] = drug_wide["drug_key"].astype(str)
        panel = panel.merge(drug_wide, on=["year", "drug_key"], how="left")

    # --- separately trained news-only SLM bridge -----------------------------
    # Optional for backwards compatibility. The adapter annualizes only
    # dated SLM features and joins them to feature year; it imports no target
    # or future validation data.
    news_only_path = cfg.artifact_path("news/news_only_catalog_features.csv.gz")
    source_path = cfg.repo_root / cfg.news_only_features
    if news_only_path.exists() or source_path.exists():
        news_only = build_news_only_annual_features(cfg)
        if not news_only.empty:
            panel = panel.merge(news_only, on="year", how="left")

    # --- lagged demand features (same city-drug, previous years) ---------------
    panel = panel.sort_values(["drug_key", "city", "year"]).reset_index(drop=True)
    g = panel.groupby(["drug_key", "city"], sort=False)
    panel["demand_claims_lag1"] = g["demand_claims"].shift(1)
    panel["demand_claims_lag2"] = g["demand_claims"].shift(2)
    panel["demand_claims_lag3"] = g["demand_claims"].shift(3)
    panel["demand_claims_ma2"] = (
        g["demand_claims"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    panel["demand_claims_ma3"] = (
        g["demand_claims"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    panel["demand_claims_delta"] = (
        panel["demand_claims_lag1"] - panel["demand_claims_lag2"]
    )
    panel["demand_claims_delta2"] = (
        panel["demand_claims_lag1"] - panel["demand_claims_lag3"]
    )

    # Log-scaled history features (stable inputs for log-target ridge/neural).
    panel["y_last"] = panel["demand_claims"]
    for c in ("y_last", "demand_claims_lag1", "demand_claims_lag2",
              "demand_claims_lag3", "demand_claims_ma2", "demand_claims_ma3"):
        panel[f"{c}_log"] = np.log1p(panel[c].clip(lower=0))
    panel["delta_log"] = panel["y_last_log"] - panel["demand_claims_lag1_log"]
    panel["delta2_log"] = panel["y_last_log"] - panel["demand_claims_lag3_log"]

    return panel.reset_index(drop=True)
