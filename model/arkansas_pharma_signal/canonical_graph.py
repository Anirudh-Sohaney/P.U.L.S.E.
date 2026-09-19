"""Canonical Arkansas pharmaceutical entity graph and edge provenance.

This module is deliberately dependency-light.  It does not manufacture links:
relationships are either read from a source, supplied by a caller, or marked
as weakly inferred with an explicit method and confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from . import io
from .entities import normalize_name

EDGE_COLUMNS = [
    "source_entity_id", "relationship_type", "target_entity_id", "source",
    "source_timestamp", "effective_date", "expiration_date", "confidence",
    "transformation_method", "evidence_type",
]

NODE_COLUMNS = ["entity_id", "entity_type", "name", "geography_level",
                "geography_id", "source", "source_timestamp", "attributes_json"]


@dataclass
class CanonicalGraph:
    """Node/edge tables for entity relationships with provenance metadata."""

    nodes: pd.DataFrame
    edges: pd.DataFrame
    version: str = "1"

    def validate(self) -> None:
        """Validate graph columns, edge uniqueness, and confidence bounds."""
        missing_nodes = set(NODE_COLUMNS) - set(self.nodes.columns)
        missing_edges = set(EDGE_COLUMNS) - set(self.edges.columns)
        if missing_nodes or missing_edges:
            raise ValueError(f"graph schema missing nodes={missing_nodes}, edges={missing_edges}")
        if self.edges.duplicated(["source_entity_id", "relationship_type", "target_entity_id",
                                  "effective_date"]).any():
            raise ValueError("duplicate canonical graph edges after normalization")
        c = pd.to_numeric(self.edges["confidence"], errors="coerce")
        if c.isna().any() or ((c < 0) | (c > 1)).any():
            raise ValueError("edge confidence must be numeric in [0, 1]")

    def save(self, cfg, prefix: str = "canonical_graph") -> None:
        """Write graph tables and metadata under the configured artifact root."""
        self.validate()
        io.write_csv(self.nodes, cfg.artifact_path(f"graph/{prefix}_nodes.csv"))
        io.write_csv(self.edges, cfg.artifact_path(f"graph/{prefix}_edges.csv"))
        io.write_metadata(cfg.artifact_path(f"graph/{prefix}.json"), version=self.version,
                          node_count=len(self.nodes), edge_count=len(self.edges),
                          edge_columns=EDGE_COLUMNS)


def _edge(src: str, rel: str, dst: str, *, source: str, timestamp: str = "",
          effective: str = "", expiration: str = "", confidence: float = 1.0,
          method: str = "direct_source", evidence: str = "direct") -> dict:
    return {"source_entity_id": src, "relationship_type": rel,
            "target_entity_id": dst, "source": source,
            "source_timestamp": timestamp, "effective_date": effective,
            "expiration_date": expiration, "confidence": float(confidence),
            "transformation_method": method, "evidence_type": evidence}


def _clean(value) -> str:
    """Serialize optional source values without turning missing data into 'nan'."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_canonical_graph(cfg, extra_edges: Iterable[Mapping] | None = None) -> CanonicalGraph:
    """Build a versioned graph from the existing entity store and NDC catalog.

    Existing final-data relationships are retained as direct source edges.
    NDC labeler and product relationships are direct FDA-derived mappings;
    parent/factory/API data is intentionally absent until a source provides it.
    """
    entities = io.load_csv(cfg.data_path(cfg.entities))
    nodes = entities.rename(columns={"source_id": "source"}).copy()
    nodes["source_timestamp"] = ""
    nodes["attributes_json"] = nodes.get("attributes_json", "{}").fillna("{}")
    nodes = nodes[[c for c in NODE_COLUMNS if c in nodes.columns]]
    for c in NODE_COLUMNS:
        if c not in nodes:
            nodes[c] = ""
    nodes = nodes[NODE_COLUMNS].drop_duplicates("entity_id")

    rel = io.load_csv(cfg.data_path(cfg.relationships))
    edges = []
    for _, r in rel.iterrows():
        attrs = r.get("attributes_json", "{}")
        try:
            attrs = json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
        except json.JSONDecodeError:
            attrs = {}
        edges.append(_edge(str(r["source_entity_id"]), str(r["relationship_type"]),
                           str(r["target_entity_id"]), source=str(r.get("source_id", "unknown")),
                           timestamp=str(attrs.get("source_timestamp", "")),
                           effective=str(attrs.get("effective_date", "")),
                           confidence=float(attrs.get("confidence", 1.0)),
                           method=str(attrs.get("transformation_method", "direct_source")),
                           evidence=str(attrs.get("evidence_type", "direct"))))

    # FDA NDC product rows provide a defensible labeler -> product relationship.
    # These rows can contain entities that are not in the older curated entity
    # file, so create explicit nodes for them instead of leaving dangling
    # edges in the canonical graph.
    generated_nodes = []
    p = cfg.data_path(cfg.fda_ndc_products)
    if p.exists():
        ndc = io.load_csv(p, usecols=["PRODUCTID", "LABELERNAME", "NONPROPRIETARYNAME",
                                      "STARTMARKETINGDATE", "source_id", "retrieved_at_utc"])
        for _, r in ndc.dropna(subset=["PRODUCTID", "LABELERNAME"]).drop_duplicates().iterrows():
            labeler = normalize_name(r["LABELERNAME"])
            product = str(r["PRODUCTID"]).strip()
            drug = normalize_name(r.get("NONPROPRIETARYNAME", ""))
            timestamp = _clean(r.get("retrieved_at_utc", ""))
            effective = _clean(r.get("STARTMARKETINGDATE", ""))
            if labeler and product:
                edges.append(_edge(f"labeler:{labeler}", "markets", f"ndc:{product}",
                                   source=str(r.get("source_id", "fda_ndc_listing")),
                                   timestamp=timestamp, effective=effective, confidence=0.99))
                generated_nodes.extend([
                    {"entity_id": f"labeler:{labeler}", "entity_type": "labeler",
                     "name": str(r["LABELERNAME"]), "source": "fda_ndc_listing"},
                    {"entity_id": f"ndc:{product}", "entity_type": "ndc",
                     "name": product, "source": "fda_ndc_listing"},
                ])
            if drug and labeler:
                edges.append(_edge(f"drug:{drug}", "manufactured_by", f"labeler:{labeler}",
                                   source=str(r.get("source_id", "fda_ndc_listing")),
                                   timestamp=timestamp, effective=effective, confidence=0.95))
                generated_nodes.append({"entity_id": f"drug:{drug}", "entity_type": "drug",
                                        "name": str(r.get("NONPROPRIETARYNAME", drug)),
                                        "source": "fda_ndc_listing"})

    # Supplier hierarchy rows are kept as separate, provenance-bearing edges.
    # Empty parent/factory/API fields intentionally produce no edge.
    try:
        from .supplier_hierarchy import build_supplier_hierarchy
        hierarchy = build_supplier_hierarchy(cfg)
    except (FileNotFoundError, KeyError, ValueError):
        hierarchy = pd.DataFrame()
    for _, r in hierarchy.iterrows():
        labeler = normalize_name(r.get("labeler", ""))
        if not labeler:
            continue
        src = str(r.get("source", "supplier_hierarchy"))
        ts = _clean(r.get("source_timestamp", ""))
        eff = _clean(r.get("effective_date", ""))
        conf = float(r.get("confidence", 0.0) or 0.0)
        for field, entity_type, relation in (("parent_company", "parent_company", "owned_by"),
                                              ("factory", "factory", "manufactures_at"),
                                              ("api_source", "api_source", "api_supplied_by")):
            value = _clean(r.get(field, ""))
            key = normalize_name(value)
            if not key or key == labeler:
                continue
            generated_nodes.append({"entity_id": f"{entity_type}:{key}", "entity_type": entity_type,
                                    "name": value, "source": src})
            edges.append(_edge(f"labeler:{labeler}", relation, f"{entity_type}:{key}",
                               source=src, timestamp=ts, effective=eff, confidence=conf,
                               method="supplier_hierarchy_direct", evidence=str(r.get("evidence_type", "direct"))))

    if extra_edges:
        edges.extend(dict(x) for x in extra_edges)
    if generated_nodes:
        nodes = pd.concat([nodes, pd.DataFrame(generated_nodes)], ignore_index=True)
    for c in NODE_COLUMNS:
        if c not in nodes:
            nodes[c] = ""
    nodes = nodes[NODE_COLUMNS].fillna("").drop_duplicates("entity_id")
    edge_frame = pd.DataFrame(edges, columns=EDGE_COLUMNS).fillna("").drop_duplicates(
        ["source_entity_id", "relationship_type", "target_entity_id", "effective_date"])
    graph = CanonicalGraph(nodes, edge_frame, version="arkansas-universal-v1")
    graph.validate()
    return graph


def validate_temporal_edges(edges: pd.DataFrame, as_of: object) -> pd.DataFrame:
    """Return only edges available at ``as_of``; missing dates remain usable."""
    t = pd.Timestamp(as_of)
    starts = pd.to_datetime(edges["effective_date"], errors="coerce")
    ends = pd.to_datetime(edges["expiration_date"], errors="coerce")
    return edges[(starts.isna() | (starts <= t)) & (ends.isna() | (ends > t))].copy()
