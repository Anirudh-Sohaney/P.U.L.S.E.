"""Supplier hierarchy contract and coverage audit."""

from __future__ import annotations

import re
import zipfile

import pandas as pd

from . import io
from .entities import normalize_name

SUPPLIER_COLUMNS = ["supplier_key", "labeler", "parent_company", "factory", "api_source",
                    "supplier_country", "drug_key", "ndc", "source", "source_timestamp",
                    "effective_date", "confidence", "evidence_type", "coverage_status"]
ESTABLISHMENT_CONTEXT_COLUMNS = ["supplier_key", "labeler", "factory", "api_source",
                                 "supplier_country", "source", "effective_date",
                                 "confidence", "evidence_type", "coverage_status"]


def _establishment_index(path) -> dict[str, dict]:
    """Read the FDA DRLS registration snapshot with its trailing empty field.

    The source has one extra tab at the end of data rows; ``index_col=False``
    is required or pandas shifts every named field one position left. The
    resulting index links only a labeler name to FDA-registered establishments
    and operations. It does not assert that a particular NDC was made there.
    """
    path = pd.io.common.stringify_path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("drls_reg.txt") as raw:
                establishments = pd.read_csv(raw, sep="\t", dtype=str,
                                              encoding="latin1", skipinitialspace=True,
                                              index_col=False)
    except (FileNotFoundError, KeyError, zipfile.BadZipFile):
        return {}
    establishments.columns = establishments.columns.astype(str).str.strip()
    required = {"FIRM_NAME", "ADDRESS", "OPERATIONS", "REGISTRANT_NAME", "EXPIRATION_DATE"}
    if not required <= set(establishments.columns):
        return {}
    establishments = establishments.fillna("")
    index: dict[str, dict] = {}
    for row in establishments.itertuples(index=False):
        firm = str(getattr(row, "FIRM_NAME", "")).strip()
        registrant = str(getattr(row, "REGISTRANT_NAME", "")).strip()
        address = str(getattr(row, "ADDRESS", "")).strip()
        operations = str(getattr(row, "OPERATIONS", "")).strip()
        if not firm:
            continue
        country_match = re.search(r"\(([A-Z]{3})\)\s*$", address)
        entry = {"factory": firm, "address": address, "operations": operations,
                 "supplier_country": country_match.group(1) if country_match else "",
                 "effective_date": str(getattr(row, "EXPIRATION_DATE", "")).strip(),
                 "direct_firm": True}
        for name, direct in ((firm, True), (registrant, False)):
            key = normalize_name(name)
            if not key:
                continue
            bucket = index.setdefault(key, {"rows": [], "direct_firm": False})
            bucket["rows"].append(entry)
            bucket["direct_firm"] = bucket["direct_firm"] or direct
    return index


def build_supplier_hierarchy(cfg) -> pd.DataFrame:
    """Extract only FDA labeler/product edges supported by local inputs.

    The establishment snapshot has firm/registrant operations but no
    NDC-to-establishment edge. An exact name match therefore cannot prove
    that an establishment makes a particular product or that a registrant is
    its parent. Those tiers remain unresolved until such an edge is available.
    """
    path = cfg.data_path(cfg.fda_ndc_products)
    establishment_index = _establishment_index(cfg.data_path(cfg.fda_establishments))
    products = io.load_csv(path, usecols=["PRODUCTID", "NONPROPRIETARYNAME", "LABELERNAME",
                                          "STARTMARKETINGDATE", "source_id", "retrieved_at_utc"])
    products = products.dropna(subset=["PRODUCTID", "LABELERNAME"]).drop_duplicates()
    rows = []
    for _, r in products.iterrows():
        labeler = str(r["LABELERNAME"]).strip()
        if not labeler:
            continue
        rows.append({
            "supplier_key": f"labeler:{normalize_name(labeler)}", "labeler": labeler,
            "parent_company": "", "factory": "", "api_source": "", "supplier_country": "",
            "drug_key": normalize_name(r.get("NONPROPRIETARYNAME", "")),
            "ndc": str(r["PRODUCTID"]), "source": str(r.get("source_id", "fda_ndc_listing")),
            "source_timestamp": str(r.get("retrieved_at_utc", "")),
            "effective_date": str(r.get("STARTMARKETINGDATE", "")), "confidence": 0.95,
            "evidence_type": "direct_labeler_only",
            "coverage_status": "labeler_only_parent_factory_api_missing",
        })
    return pd.DataFrame(rows, columns=SUPPLIER_COLUMNS).drop_duplicates(
        ["supplier_key", "drug_key", "ndc"])


def build_establishment_context(cfg) -> pd.DataFrame:
    """Build supplier-level FDA registration context without product claims."""
    products = io.load_csv(cfg.data_path(cfg.fda_ndc_products), usecols=["LABELERNAME"])
    index = _establishment_index(cfg.data_path(cfg.fda_establishments))
    rows = []
    for labeler in sorted({str(x).strip() for x in products["LABELERNAME"].dropna() if str(x).strip()}):
        matches = index.get(normalize_name(labeler), {})
        establishments = matches.get("rows", [])
        if not establishments:
            continue
        factories = sorted({x["factory"] for x in establishments if x.get("factory")})
        api_sources = sorted({x["factory"] for x in establishments
                              if "API" in str(x.get("operations", "")).upper()})
        countries = sorted({x["supplier_country"] for x in establishments
                            if x.get("supplier_country")})
        rows.append({
            "supplier_key": f"labeler:{normalize_name(labeler)}", "labeler": labeler,
            "factory": "; ".join(factories), "api_source": "; ".join(api_sources),
            "supplier_country": "; ".join(countries), "source": "FDA DRLS registration",
            "effective_date": "; ".join(sorted({x["effective_date"] for x in establishments if x.get("effective_date")})),
            "confidence": 0.82 if matches.get("direct_firm") else 0.72,
            "evidence_type": "supplier_registration_context",
            "coverage_status": "not_product_specific_no_ndc_establishment_edge",
        })
    return pd.DataFrame(rows, columns=ESTABLISHMENT_CONTEXT_COLUMNS).drop_duplicates("supplier_key")


def validate_supplier_hierarchy(frame: pd.DataFrame) -> None:
    missing = set(SUPPLIER_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"supplier hierarchy missing columns: {sorted(missing)}")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    if confidence.isna().any() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("supplier confidence must be in [0, 1]")
    labeler_only = frame["evidence_type"].astype(str).eq("direct_labeler_only")
    unsupported = labeler_only & frame[["parent_company", "factory", "api_source"]].fillna("").astype(str).ne("").any(axis=1)
    if unsupported.any():
        raise ValueError("labeler-only rows cannot claim parent, factory, or API edges")
