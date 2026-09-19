"""Entity mappings: normalized drug names and drug identity attributes.

Maps CMS Part D generic names to FDA NDC active ingredients, labelers, dosage
form, route, marketing category, DEA schedule, and pharmaceutical class by
normalized nonproprietary name. The FDA product file is the primary source;
the final_data entity graph (drug->active_ingredient edges) is used as a
secondary alignment for generic names absent from the FDA catalog.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

import pandas as pd

from . import io

_MULTI_SPACE = re.compile(r"\s+")
_NOISE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: object) -> str:
    """Lowercase, strip dosage-form noise and punctuation."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower()
    s = re.sub(r"\b(injection|tablet|tablets|capsule|capsules|oral|solution|"
               r"injection,|vial|vials|syringe|pack|kit|spray|cream|ointment|"
               r"gel|suspension|powder|for injection|film coated|extended release|"
               r"er\b|sr\b|dr\b|delayed release|oral solution|intravenous|"
               r"subcutaneous|sodium|hydrochloride|monohydrate|dihydrate|"
               r"hcl\b|acetaminophen;)\b", "", s)
    s = _MULTI_SPACE.sub(" ", s).strip(" -/")
    return _NOISE.sub(" ", s).strip()


def _mode(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    vals = vals[vals != ""]
    if vals.empty:
        return ""
    return vals.mode().iloc[0]


@dataclass
class DrugIdentity:
    """Per-drug static identity attributes resolved from real FDA data."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    ingredient_products: Dict[str, int] = field(default_factory=dict)
    labeler_products: Dict[str, int] = field(default_factory=dict)

    @property
    def has(self) -> bool:
        """Whether the identity table contains at least one resolved drug."""
        return not self.table.empty


def build_drug_identity(cfg) -> DrugIdentity:
    """Build a static per-drug table from FDA NDC products + entity graph."""
    fda = io.load_csv(
        cfg.data_path(cfg.fda_ndc_products),
        usecols=["NONPROPRIETARYNAME", "SUBSTANCENAME", "LABELERNAME",
                 "DOSAGEFORMNAME", "ROUTENAME", "MARKETINGCATEGORYNAME",
                 "DEASCHEDULE", "PHARM_CLASSES"],
    )
    fda["drug_key"] = fda["NONPROPRIETARYNAME"].map(normalize_name)
    fda = fda[fda["drug_key"] != ""].copy()

    ingredients: Dict[str, Set[str]] = {}
    labelers: Dict[str, Set[str]] = {}
    for key, subs in fda.groupby("drug_key")["SUBSTANCENAME"]:
        bag: Set[str] = set()
        for raw in subs:
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            for part in str(raw).split(";"):
                ing = normalize_name(part)
                if ing and ing != "nan":
                    bag.add(ing)
        if bag:
            ingredients[key] = bag

    for key, labs in fda.groupby("drug_key")["LABELERNAME"]:
        bag = {str(x).strip() for x in labs.astype(str) if str(x).strip().lower() != "nan"}
        if bag:
            labelers[key] = bag

    rows: List[Dict] = []
    for key, g in fda.groupby("drug_key"):
        rows.append({
            "drug_key": key,
            "ingredient": sorted(ingredients.get(key, []))[0] if ingredients.get(key) else "",
            "labeler": sorted(labelers.get(key, []))[0] if labelers.get(key) else "",
            "dosage_form": _mode(g["DOSAGEFORMNAME"]),
            "route": _mode(g["ROUTENAME"]),
            "market_cat": _mode(g["MARKETINGCATEGORYNAME"]),
            "dea_schedule": _mode(g["DEASCHEDULE"]),
            "pharm_class": _first_pharm_class(g["PHARM_CLASSES"]),
            "product_count": int(len(g)),
        })
    table = pd.DataFrame(rows)

    # Secondary alignment: final_data drug->active_ingredient edges.
    relationships = io.load_csv(
        cfg.data_path(cfg.relationships),
        usecols=["relationship_type", "source_entity_id", "target_entity_id"],
    )
    contains = relationships[relationships["relationship_type"] == "contains"]
    edge_rows: List[Dict] = []
    for _, row in contains.iterrows():
        src = str(row["source_entity_id"])
        tgt = str(row["target_entity_id"])
        if src.startswith("drug:") and tgt.startswith("ingredient:"):
            edge_rows.append({"drug_key": normalize_name(src[5:]),
                              "ingredient": normalize_name(tgt[11:])})
    if edge_rows:
        edges = pd.DataFrame(edge_rows)
        edges = edges[edges["drug_key"] != ""]
        edges = edges.groupby("drug_key", as_index=False)["ingredient"].agg(lambda s: sorted(set(s))[0])
        table = pd.concat([table, edges], ignore_index=True)
        table = table.drop_duplicates(subset=["drug_key"], keep="first")

    ing_counts: Dict[str, int] = {}
    for ing in ingredients.values():
        for i in ing:
            ing_counts[i] = ing_counts.get(i, 0) + 1
    lab_counts: Dict[str, int] = {}
    for labs in labelers.values():
        for l in labs:
            lab_counts[l] = lab_counts.get(l, 0) + 1

    return DrugIdentity(table=table.reset_index(drop=True),
                        ingredient_products=ing_counts,
                        labeler_products=lab_counts)


def _first_pharm_class(series: pd.Series) -> str:
    for raw in series.dropna().astype(str):
        if not raw:
            continue
        first = raw.split(",")[0].strip()
        if first:
            return first.split(" [")[0].strip()
    return ""


def build_therapeutic_mapping(cfg) -> Dict[str, str]:
    """Map drug_key -> FDA shortage therapeutic category (from shortages file)."""
    path = cfg.data_path(cfg.fda_shortages)
    if not path.exists():
        return {}
    short = io.load_csv(path, usecols=["generic_name", "therapeutic_category"])
    out: Dict[str, str] = {}
    for key, cats in short.groupby(short["generic_name"].map(normalize_name))["therapeutic_category"]:
        if not key:
            continue
        for c in cats.dropna().astype(str):
            parsed = _json_first(c)
            if parsed:
                out[key] = parsed
                break
    return out


def _json_first(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return str(value).strip().strip('"')
    if isinstance(parsed, list) and parsed:
        return str(parsed[0]).strip().strip('"')
    if isinstance(parsed, str):
        return parsed.strip().strip('"')
    return ""


def project_supplier_exposure(
    panel: pd.DataFrame,
    identity: DrugIdentity,
    therapeutic: Dict[str, str],
    key_column: str = "drug_key",
) -> pd.DataFrame:
    """Attach drug identity columns and exposure counts to a panel by drug key."""
    panel = panel.copy()
    t = identity.table
    if identity.has:
        panel = panel.merge(
            t, on=key_column, how="left"
        ).merge(
            pd.DataFrame([{key_column: k, "ingredient_products": v}
                          for k, v in identity.ingredient_products.items()]),
            on=key_column, how="left",
        ).merge(
            pd.DataFrame([{key_column: k, "labeler_products": v}
                          for k, v in identity.labeler_products.items()]),
            on=key_column, how="left",
        )
    else:
        for c in ("ingredient", "labeler", "dosage_form", "route", "market_cat",
                  "dea_schedule", "pharm_class", "product_count",
                  "ingredient_products", "labeler_products"):
            panel[c] = ""
    panel["ingredient_products"] = pd.to_numeric(panel["ingredient_products"], errors="coerce").fillna(0)
    panel["labeler_products"] = pd.to_numeric(panel["labeler_products"], errors="coerce").fillna(0)
    panel["product_count"] = pd.to_numeric(panel["product_count"], errors="coerce").fillna(0)
    panel["therapeutic_cat"] = panel[key_column].map(therapeutic).fillna("")
    return panel
