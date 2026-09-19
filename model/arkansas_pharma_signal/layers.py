"""Annual external feature layers from real local data.

Deterministic, no synthetic values. Each builder returns a long [year, ...]
or [year, drug, ...] / [year, labeler, ...] table that build-panel merges onto
the demand panel:

- disease: FluView ILI/WILI (AR + national), wastewater WVAL per pathogen
  (SARS-CoV-2 / influenza A / RSV, AR + national), NNDSS annual disease
  burden by label (AR + US RESIDENTS).
- supply: FDA shortage reasons, FDA enforcement recall classification counts,
  recall reason keywords, per-drug recall class counts, per-labeler
  shortage/recall exposure.
- provider: prescriber-type mix and beneficiary proxies are computed directly
  in features.build_panel from the raw CMS Part D file.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from . import io
from .entities import normalize_name

NNDSS_TOP_LABELS = 6
RECALL_KEYWORDS = ["sterility", "cgmp", "potency", "mislabel", "contamination"]


def build_disease_annual(cfg) -> pd.DataFrame:
    """Annual disease-layer features keyed by year."""
    flu = _fluview_annual(cfg)
    ww = _wastewater_annual(cfg)
    nd = _nndss_annual(cfg)
    frames = [f for f in (flu, ww, nd) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["year"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="year", how="outer")
    return out.sort_values("year").reset_index(drop=True)


def _fluview_annual(cfg) -> pd.DataFrame:
    path = cfg.data_path(cfg.disease_surveillance_dir) / "cdc_fluview_ar_national_weekly.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["year"])
    df = io.load_csv(path, usecols=["region", "issue", "ili", "wili"])
    df["year"] = pd.to_numeric(df["issue"], errors="coerce") // 100
    df = df.dropna(subset=["ili", "wili", "year"])
    df["year"] = df["year"].astype(int)
    rows: List[Dict] = []
    sub = df[df["region"] == "nat"]
    for col, name in (("ili", "nat_ili_mean"), ("wili", "nat_wili_mean")):
        for year, v in sub.groupby("year")[col].mean().items():
            rows.append({"year": year, "variable_id": name, "value": float(v)})
    return _to_wide(rows)


def _wastewater_annual(cfg) -> pd.DataFrame:
    rows: List[Dict] = []
    ar_path = cfg.data_path(cfg.disease_surveillance_dir) / "cdc_wastewater_ar_site_weekly.csv.gz"
    if ar_path.exists():
        df = io.load_csv(ar_path, usecols=["week_end", "pathogen_target", "site_wval"])
        df["year"] = pd.to_datetime(df["week_end"], errors="coerce").dt.year
        df = df.dropna(subset=["year", "site_wval"])
        df["year"] = df["year"].astype(int)
        for target, sub in df.groupby("pathogen_target"):
            slug = _pathogen_slug(target)
            for year, v in sub.groupby("year")["site_wval"].mean().items():
                rows.append({"year": year, "variable_id": f"ww_{slug}_ar", "value": float(v)})
    nat_path = cfg.data_path(cfg.disease_surveillance_dir) / "cdc_wastewater_national_weekly.csv.gz"
    if nat_path.exists():
        df = io.load_csv(nat_path, usecols=["week_end", "pathogen_target", "national_wval_mean"])
        df["year"] = pd.to_datetime(df["week_end"], errors="coerce").dt.year
        df = df.dropna(subset=["year", "national_wval_mean"])
        df["year"] = df["year"].astype(int)
        for target, sub in df.groupby("pathogen_target"):
            slug = _pathogen_slug(target)
            for year, v in sub.groupby("year")["national_wval_mean"].mean().items():
                rows.append({"year": year, "variable_id": f"ww_{slug}_nat", "value": float(v)})
    return _to_wide(rows)


def _pathogen_slug(target: str) -> str:
    t = target.lower()
    if "sars" in t:
        return "covid"
    if "influenza" in t:
        return "flu"
    if "rsv" in t:
        return "rsv"
    return re.sub(r"[^a-z0-9]", "_", t).strip("_")


def _nndss_annual(cfg) -> pd.DataFrame:
    path = cfg.data_path(cfg.disease_surveillance_dir) / "cdc_nndss_ar_national_weekly.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["year"])
    df = io.load_csv(path, usecols=["states", "year", "label", "m1", "m2", "m3", "m4"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for c in ("m1", "m2", "m3", "m4"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def burden(frame: pd.DataFrame) -> pd.Series:
        # Per (year, label) annual burden: sum of monthly maxima.
        frame = frame.copy()
        frame["_max"] = frame[["m1", "m2", "m3", "m4"]].max(axis=1)
        return frame.groupby(["year", "label"])["_max"].max()

    ar = burden(df[df["states"].astype(str).str.contains("Arkansas", case=False)])
    us = burden(df[df["states"].astype(str).str.contains("US RESIDENTS", case=False)])
    if ar.empty:
        return pd.DataFrame(columns=["year"])

    top = (ar.groupby("label").sum().sort_values(ascending=False)
              .head(NNDSS_TOP_LABELS).index.tolist())
    rows: List[Dict] = []
    for label in top:
        slug = _slug(label)
        for series, suffix in ((ar, "ar"), (us, "us")):
            sub = series.loc[series.index.get_level_values("label") == label]
            for (year, _), v in sub.items():
                rows.append({"year": year, "variable_id": f"nndss_{slug}_{suffix}",
                             "value": float(v)})
    return _to_wide(rows)


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s[:24]


def _to_wide(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["year"])
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="year", columns="variable_id", values="value",
                            aggfunc="first").reset_index()
    wide.columns.name = None
    wide["year"] = wide["year"].astype(int)
    return wide


def build_supply_annual(cfg) -> pd.DataFrame:
    """Annual shortage-reason and recall-classification counts."""
    rows: List[Dict] = []
    _shortage_reason_rows(cfg, rows)
    _recall_class_rows(cfg, rows)
    return _to_wide(rows)


def _shortage_reason_rows(cfg, rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.fda_shortages)
    if not path.exists():
        return
    short = io.load_csv(path, usecols=["initial_posting_date", "shortage_reason"])
    short["year"] = pd.to_datetime(short["initial_posting_date"], errors="coerce").dt.year
    short = short.dropna(subset=["year", "shortage_reason"])
    short["year"] = short["year"].astype(int)

    def bucket(reason: str) -> str:
        r = reason.lower()
        if "demand increase" in r:
            return "demand"
        if "discontinu" in r:
            return "discontinuation"
        if "active ingredient" in r:
            return "ingredient"
        return "other"

    short["_bucket"] = short["shortage_reason"].map(bucket)
    for year, sub in short.groupby("year"):
        counts = sub["_bucket"].value_counts()
        rows.append({"year": year, "variable_id": "shortage_reason_demand",
                     "value": float(counts.get("demand", 0))})
        rows.append({"year": year, "variable_id": "shortage_reason_discontinuation",
                     "value": float(counts.get("discontinuation", 0))})
        rows.append({"year": year, "variable_id": "shortage_reason_ingredient",
                     "value": float(counts.get("ingredient", 0))})
        rows.append({"year": year, "variable_id": "shortage_reason_other",
                     "value": float(counts.get("other", 0))})


def _recall_class_rows(cfg, rows: List[Dict]) -> None:
    path = cfg.data_path(cfg.fda_enforcement)
    if not path.exists():
        return
    enf = io.load_csv(path, usecols=["report_date", "classification", "reason_for_recall"])
    enf["year"] = (pd.to_numeric(enf["report_date"], errors="coerce") // 10000)
    enf = enf.dropna(subset=["year", "classification"])
    enf["year"] = enf["year"].astype(int)
    levels = {"Class I": 1, "Class II": 2, "Class III": 3}

    for year, sub in enf.groupby("year"):
        counts = sub["classification"].value_counts()
        for label, level in levels.items():
            rows.append({"year": year,
                         "variable_id": f"recall_class_{level}",
                         "value": float(counts.get(label, 0))})
        text = sub["reason_for_recall"].fillna("").astype(str).str.lower()
        for kw in RECALL_KEYWORDS:
            rows.append({"year": year,
                         "variable_id": f"recall_kw_{kw}",
                         "value": float(text.str.contains(kw, regex=False).sum())})


def build_supply_drug(cfg) -> pd.DataFrame:
    """Per (year, drug_key) recall-class counts from FDA enforcement."""
    path = cfg.data_path(cfg.fda_enforcement)
    if not path.exists():
        return pd.DataFrame(columns=["year", "drug"])
    enf = io.load_csv(path, usecols=["report_date", "product_description", "classification"])
    enf["year"] = (pd.to_numeric(enf["report_date"], errors="coerce") // 10000)
    enf = enf.dropna(subset=["year", "classification"])
    enf["year"] = enf["year"].astype(int)

    rows: List[Dict] = []
    levels = {"Class I": "1", "Class II": "2", "Class III": "3"}
    for _, r in enf.iterrows():
        for key in _drug_keys_from_description(r["product_description"]):
            level = levels.get(str(r["classification"]), "3")
            rows.append({"year": int(r["year"]), "drug": key,
                         "variable_id": f"recall_class{level}", "value": 1.0})
    wide = _to_wide_drug(rows)
    return wide


def _drug_keys_from_description(description: object) -> List[str]:
    if not isinstance(description, str):
        return []
    head = description.split(",")[0].strip()
    key = normalize_name(head)
    return [key] if key else []


def _to_wide_drug(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["year", "drug"])
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["year", "drug"], columns="variable_id",
                            values="value", aggfunc="sum").reset_index()
    wide.columns.name = None
    return wide


def build_labeler_annual(cfg) -> pd.DataFrame:
    """Per (year, labeler) shortage/recall exposure from drug-key joins."""
    identity = _identity_for_labeler(cfg)
    rows: List[Dict] = []

    short_path = cfg.data_path(cfg.fda_shortages)
    if short_path.exists():
        short = io.load_csv(short_path, usecols=["generic_name", "initial_posting_date"])
        short["year"] = pd.to_datetime(short["initial_posting_date"], errors="coerce").dt.year
        short = short.dropna(subset=["year"])
        short["year"] = short["year"].astype(int)
        short["drug_key"] = short["generic_name"].map(normalize_name)
        short = short.merge(identity, on="drug_key", how="left")
        short = short[short["labeler"] != ""]
        for (year, labeler), sub in short.groupby(["year", "labeler"]):
            rows.append({"year": year, "labeler": labeler,
                         "variable_id": "labeler_shortage_n", "value": float(len(sub))})

    enf_path = cfg.data_path(cfg.fda_enforcement)
    if enf_path.exists():
        enf = io.load_csv(enf_path, usecols=["report_date", "product_description"])
        enf["year"] = (pd.to_numeric(enf["report_date"], errors="coerce") // 10000)
        enf = enf.dropna(subset=["year"])
        enf["year"] = enf["year"].astype(int)
        matched: List[Dict] = []
        for _, r in enf.iterrows():
            for key in _drug_keys_from_description(r["product_description"]):
                matched.append({"year": int(r["year"]), "drug_key": key})
        if matched:
            m = pd.DataFrame(matched).merge(identity, on="drug_key", how="left")
            m = m[m["labeler"] != ""]
            for (year, labeler), sub in m.groupby(["year", "labeler"]):
                rows.append({"year": year, "labeler": labeler,
                             "variable_id": "labeler_recall_n", "value": float(len(sub))})

    return _to_wide_labeler(rows)


def _identity_for_labeler(cfg) -> pd.DataFrame:
    fda = io.load_csv(cfg.data_path(cfg.fda_ndc_products),
                      usecols=["NONPROPRIETARYNAME", "LABELERNAME"])
    fda["drug_key"] = fda["NONPROPRIETARYNAME"].map(normalize_name)
    fda = fda[fda["drug_key"] != ""]
    fda["labeler"] = fda["LABELERNAME"].astype(str).str.strip()
    fda = fda[fda["labeler"].str.lower() != "nan"]
    return fda[["drug_key", "labeler"]].drop_duplicates("drug_key")


def _to_wide_labeler(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["year", "labeler"])
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["year", "labeler"], columns="variable_id",
                            values="value", aggfunc="sum").reset_index()
    wide.columns.name = None
    return wide
