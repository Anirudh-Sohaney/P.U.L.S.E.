"""Real-label training for the modular Arkansas model.

The supervised target is the observed next-quarter Arkansas Medicaid SDUD
prescription count.  Rows are split by feature year (<=2020 train, 2021
validation, >2021 test), matching the existing quarterly evaluation contract.
No labels are synthesized.  Missing text/graph context is represented by an
explicit mask or unresolved entity ID and is never filled with a fabricated
event or supplier relationship.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from . import io
from .entities import normalize_name
from .modular_model import ArkansasPharmaMultimodalModel, ModularModelConfig, model_inventory
from .quarterly import add_nadac_price_layer, build_quarterly_view, metrics_dict
from .input_contract import summarize_dispositions
from .historical_metric_context import build_historical_metric_context, METRIC_CONTEXT_WIDTH


HISTORY_COLUMNS = [
    "value", "value_last", "ma2", "prev_q_log", "lag2_log", "ma2_log",
    "q1", "q2", "q3", "q4", "quarter", "fda_shortage_active",
    "fda_shortage_supplier_count", "log_demand_change",
    "relative_demand_change",
]
# These are deliberately bounded and named.  They are aggregated from the
# previous completed annual panel year so annual summaries cannot leak future
# quarters into the current-quarter representation.
ANNUAL_CONTEXT_COLUMNS = [
    "ar_disaster_active_mean", "ar_ili_mean", "ar_wili_mean",
    "ar_temperature_mean", "ar_precipitation_mean", "ar_wind_mean",
    "ar_unemployment_mean", "global_gscpi_mean", "us_tariff_rate_mean",
    "nat_ili_mean", "nat_wili_mean", "ww_covid_ar", "ww_flu_ar", "ww_rsv_ar",
    "news_shortage_count", "news_recall_count", "news_disease_outbreak_count",
    "news_disease_influenza_outbreak_risk", "news_relevance_mean",
]
NADAC_CONTEXT_COLUMNS = [
    "nadac_price_mean", "nadac_price_obs", "nadac_price_log", "nadac_price_qoq",
]
CONTEXT_COLUMNS = [*ANNUAL_CONTEXT_COLUMNS, *NADAC_CONTEXT_COLUMNS]
HISTORY_FEATURE_COLUMNS = [*HISTORY_COLUMNS, *CONTEXT_COLUMNS]
METRIC_FEATURE_COUNT = METRIC_CONTEXT_WIDTH


def modular_input_contract() -> dict:
    """Return the operational/research status of modular history features."""
    return summarize_dispositions(HISTORY_FEATURE_COLUMNS)


def require_operational_modular(contract: dict | None = None) -> None:
    """Reject modular training when its declared features are not live-ready."""
    contract = contract or modular_input_contract()
    if not contract.get("operational_ready", False):
        raise RuntimeError(
            "modular feature surface contains periodic or historical-only inputs; "
            "use the default research mode or remove those features before an "
            "operational checkpoint can be trained")
RELATIONS = {
    "unknown": 0, "contains": 1, "represents": 2, "manufactured_by": 3,
    "located_in": 4, "maps_to": 5, "affects": 6, "mentions": 7,
}


@dataclass
class TrainConfig:
    train_last_year: int = 2020
    validation_year: int = 2021
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    max_tokens: int = 64
    seed: int = 17
    freeze_news: bool = False
    device: str = "cpu"
    baseline_mode: str = "persistence"
    loss_mode: str = "log_huber"
    mask_drug_identity: bool = False
    use_drug_content_entities: bool = False
    prioritize_graph_evidence: bool = False
    use_ndc_labeler_entities: bool = False
    use_nadac_price_features: bool = False
    # Opt-in until a chronological ablation demonstrates incremental utility.
    use_historical_metric_context: bool = False
    operational_only: bool = False
    state_loss_weight: float = 0.1
    state_target_mode: str = "change"


def research_small_model_config() -> ModularModelConfig:
    """Return a CPU-testable model using the production input contracts."""
    return ModularModelConfig(
        vocab_size=4_096, max_tokens=64, text_width=64, text_layers=2,
        text_heads=4, text_ffn=128, entity_vocab=5_000, entity_width=32,
        graph_layers=1, temporal_layers=1, temporal_heads=4,
        temporal_ffn=64, history_features=len(HISTORY_FEATURE_COLUMNS),
        horizon_count=1, target_count=1,
    )


def _stable_id(value: str, modulo: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (modulo - 1)


def _ndc_labeler_lookup(root: Path) -> Dict[str, str]:
    """Map NDC-encoded labeler segments to observed FDA labeler entities."""
    paths = [
        (root / "data/demand_price/data/openfda_ndc_2012_2022.csv.gz",
         "product_ndc", "labeler_name"),
        (root / "data/targeted_additions/fda_ndc_directory/data/fda_ndc_products_current.csv.gz",
         "PRODUCTNDC", "LABELERNAME"),
    ]
    code_to_names: Dict[str, set[str]] = {}
    for path, ndc_column, labeler_column in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=[ndc_column, labeler_column],
                            low_memory=False).dropna()
        for raw_ndc, raw_labeler in frame.itertuples(index=False, name=None):
            segments = str(raw_ndc).split("-")
            if not segments or not segments[0].isdigit():
                continue
            code = segments[0].lstrip("0") or "0"
            labeler = normalize_name(raw_labeler)
            if labeler:
                code_to_names.setdefault(code, set()).add(labeler)
    # Keep only unambiguous code-to-labeler mappings; an NDC labeler code is
    # not enough to choose among contradictory catalog names.
    return {
        code: f"labeler:{next(iter(names))}"
        for code, names in code_to_names.items() if len(names) == 1
    }


def _labeler_entity_for_ndc(ndc: object, lookup: Dict[str, str]) -> str:
    digits = re.sub(r"\D", "", str(ndc or ""))
    if not digits:
        return ""
    # Standardized 11-digit NDCs use a five-digit labeler segment. Older
    # four-digit labeler codes are also accepted only when catalog-backed.
    for width in (5, 4):
        code = digits[:width].lstrip("0") or "0"
        if code in lookup:
            return lookup[code]
    return ""


def _tokenize(text: str, vocab_size: int, max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
    words = str(text or "").split()
    tokens = np.zeros(max_tokens, dtype=np.int64)
    mask = np.zeros(max_tokens, dtype=np.bool_)
    for i, word in enumerate(words[:max_tokens]):
        tokens[i] = _stable_id(word.lower(), vocab_size)
        mask[i] = True
    return tokens, mask


def _quarter_texts(corpus_path: Path, vocab_size: int, max_tokens: int) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    if not corpus_path.exists():
        return {}
    corpus = pd.read_csv(corpus_path, compression="gzip", low_memory=False,
                         usecols=["title", "body", "published_at"])
    dates = pd.to_datetime(corpus["published_at"], errors="coerce", utc=True)
    corpus = corpus.loc[dates.notna()].copy()
    corpus["year"] = dates.loc[corpus.index].dt.year.astype(int)
    corpus["quarter"] = dates.loc[corpus.index].dt.quarter.astype(int)
    out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for (year, quarter), group in corpus.groupby(["year", "quarter"]):
        # The corpus is real historical text.  Concatenation is only a
        # deterministic document-window transformation, not a new article.
        text = " ".join((group["title"].fillna("") + " " + group["body"].fillna(""))
                         .astype(str).tolist())
        out[(int(year), int(quarter))] = _tokenize(text, vocab_size, max_tokens)
    return out


def _graph_context(graph_lookup: Dict[str, List[Tuple[str, str]]], drug: str,
                   entity_vocab: int, *, use_content_entities: bool = False,
                   prioritize_evidence: bool = False,
                   observed_entities: Sequence[Tuple[str, str]] = (),
                   nodes: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve only observed outgoing canonical-graph edges for a drug."""
    nodes = nodes or (8 if (use_content_entities or prioritize_evidence
                             or observed_entities) else 4)
    drug_id = f"drug:{drug}"
    entities = [drug_id, "AR"]
    relations = np.zeros((nodes, nodes), dtype=np.int64)
    content_entities = []
    if use_content_entities:
        # Shared name tokens provide a content path for identities unseen in
        # an earlier rolling fold. They are evidence-free lexical features,
        # not inferred ingredient or supplier relationships.
        tokens = re.findall(r"[a-z0-9]+", drug.lower())
        content_entities = [f"drug_token:{token}" for token in dict.fromkeys(tokens)]
        content_entities = content_entities[: nodes - 2]
        entities.extend(content_entities)
        for i in range(2, 2 + len(content_entities)):
            relations[0, i] = RELATIONS["contains"]
    for relation_type, target_entity in observed_entities:
        if len(entities) >= nodes:
            break
        entities.append(target_entity)
        relations[0, len(entities) - 1] = RELATIONS.get(relation_type, 0)
    edge_start = len(entities)
    graph_edges = list(graph_lookup.get(drug_id, []))
    if prioritize_evidence:
        priority = {"contains": 0, "manufactured_by": 1, "represents": 2,
                    "api_supplied_by": 3, "owned_by": 4, "markets": 5}
        graph_edges.sort(key=lambda edge: priority.get(edge[0], 99))
    for relation_type, target_entity in graph_edges[:nodes - edge_start]:
            entities.append(target_entity)
            if len(entities) >= nodes:
                break
    for i, (relation_type, _) in enumerate(
            graph_edges[:nodes - edge_start], start=edge_start):
            if i < nodes:
                relations[0, i] = RELATIONS.get(relation_type, 0)
    entity_ids = np.asarray([_stable_id(x, entity_vocab) for x in entities[:nodes]], dtype=np.int64)
    if len(entity_ids) < nodes:
        entity_ids = np.pad(entity_ids, (0, nodes - len(entity_ids)))
    return entity_ids, relations


def _previous_year_context(root: Path) -> Dict[int, np.ndarray]:
    """Aggregate qualified context by year for previous-year joins only."""
    path = root / "model/artifacts/panel/panel.csv"
    if not path.exists():
        return {}
    available = set(pd.read_csv(path, nrows=0).columns)
    keep = [c for c in ["year", *ANNUAL_CONTEXT_COLUMNS] if c in available]
    if keep == ["year"]:
        return {}
    annual = pd.read_csv(path, usecols=keep, low_memory=False)
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce")
    for column in ANNUAL_CONTEXT_COLUMNS:
        if column not in annual:
            annual[column] = 0.0
        annual[column] = pd.to_numeric(annual[column], errors="coerce").fillna(0.0)
    grouped = annual.groupby("year", as_index=False)[ANNUAL_CONTEXT_COLUMNS].mean()
    return {
        int(row[0]): np.asarray(row[1:], dtype=np.float32)
        for row in grouped.itertuples(index=False, name=None)
        if np.isfinite(row[0])
    }


def build_real_training_arrays(root: Path, cfg: TrainConfig, model_cfg: ModularModelConfig,
                               *, train_last_year: int | None = None,
                               validation_year: int | None = None):
    panel = pd.read_csv(
        root / "model/artifacts/panel/medicaid_sdud_quarterly_panel.csv",
        dtype={"ndc": "string"}, low_memory=False)
    # Chronological order is required for both per-drug histories and the
    # fallback transition-ratio pools. Sorting by drug first would let a
    # later-year ratio from one drug influence an earlier row for another.
    view = build_quarterly_view(panel).sort_values(
        ["year", "quarter", "drug"]).reset_index(drop=True)
    if cfg.use_nadac_price_features:
        view, _ = add_nadac_price_layer(view)
    for column in NADAC_CONTEXT_COLUMNS:
        source_column = f"exo_{column}"
        if source_column not in view:
            view[source_column] = 0.0
    text_map = _quarter_texts(root / "model/artifacts/news/historical_corpus.csv.gz",
                              model_cfg.vocab_size, cfg.max_tokens)
    ndc_labelers = (_ndc_labeler_lookup(root)
                    if cfg.use_ndc_labeler_entities else {})
    context_by_year = _previous_year_context(root)
    metric_context_by_row = {}
    metric_context_report = {"enabled": False}
    if cfg.use_historical_metric_context:
        metric_context_by_row, metric_context_report = build_historical_metric_context(root, view)
    edge_path = root / "model/artifacts/graph/canonical_graph_edges.csv"
    edges = (pd.read_csv(edge_path, usecols=["source_entity_id", "relationship_type", "target_entity_id"])
             if edge_path.exists() else pd.DataFrame())
    graph_lookup: Dict[str, List[Tuple[str, str]]] = {}
    if not edges.empty:
        for source, group in edges.groupby(edges["source_entity_id"].astype(str)):
            graph_lookup[source] = [(str(r.relationship_type), str(r.target_entity_id))
                                    for r in group.itertuples(index=False)]
    samples = []
    histories: Dict[str, List[np.ndarray]] = {}
    ratio_by_drug_quarter: Dict[Tuple[str, int], List[float]] = {}
    ratio_by_quarter: Dict[int, List[float]] = {}
    all_ratios: List[float] = []
    last_q_id_by_drug: Dict[str, int] = {}
    for row in view.itertuples(index=False):
        drug_name = str(row.drug)
        q_id = (int(row.year) - 2012) * 4 + int(row.quarter) - 1
        # Do not carry a temporal state vector across an unobserved quarter.
        # The target view is strict, but the history queue also needs an
        # explicit reset or stale observations become false near-term lags.
        if (drug_name in last_q_id_by_drug
                and q_id - last_q_id_by_drug[drug_name] != 1):
            histories[drug_name] = []
        last_q_id_by_drug[drug_name] = q_id
        key = (str(row.drug), int(row.quarter))
        candidates = ratio_by_drug_quarter.get(key, [])
        if not candidates:
            candidates = ratio_by_quarter.get(int(row.quarter), [])
        if not candidates:
            candidates = all_ratios
        ratio = float(np.median(candidates)) if candidates else 1.0
        transition_base = max(float(row.value), 0.0) * max(ratio, 0.0)
        if cfg.baseline_mode == "absolute":
            # Research ablation for the residual objective: predict the raw
            # log1p level with no temporal anchor. base_log=0 keeps the
            # prediction/decoding path unchanged (expm1(base_log + point)).
            base_log_value = 0.0
        elif cfg.baseline_mode == "transition":
            base_log_value = float(np.log1p(transition_base))
        elif cfg.baseline_mode == "persistence":
            base_log_value = float(np.log1p(max(float(row.value), 0.0)))
        else:
            raise ValueError(f"unsupported baseline_mode: {cfg.baseline_mode}")
        core_values = []
        for column in HISTORY_COLUMNS:
            if column == "log_demand_change":
                current = pd.to_numeric(getattr(row, "value", 0.0), errors="coerce")
                previous = pd.to_numeric(getattr(row, "value_last", current), errors="coerce")
                raw = (np.log1p(max(float(current), 0.0))
                       - np.log1p(max(float(previous), 0.0))) \
                    if pd.notna(current) and pd.notna(previous) else 0.0
            elif column == "relative_demand_change":
                current = pd.to_numeric(getattr(row, "value", 0.0), errors="coerce")
                previous = pd.to_numeric(getattr(row, "value_last", current), errors="coerce")
                raw = ((float(current) - float(previous)) / (abs(float(previous)) + 1.0)
                       if pd.notna(current) and pd.notna(previous) else 0.0)
            else:
                raw = getattr(row, column, 0.0)
            numeric = pd.to_numeric(raw, errors="coerce")
            core_values.append(float(numeric) if pd.notna(numeric)
                               and np.isfinite(float(numeric)) else 0.0)
        core_feature = np.asarray(core_values, dtype=np.float32)
        # Feature quarter t may use only context from completed year t-1.
        annual_context = context_by_year.get(
            int(row.year) - 1,
            np.zeros(len(ANNUAL_CONTEXT_COLUMNS), dtype=np.float32))
        nadac_context = np.asarray([
            float(getattr(row, f"exo_{column}"))
            if np.isfinite(float(getattr(row, f"exo_{column}"))) else 0.0
            for column in NADAC_CONTEXT_COLUMNS
        ], dtype=np.float32)
        feature = np.concatenate(
            [core_feature, annual_context, nadac_context]).astype(np.float32)
        prior = histories.setdefault(drug_name, [])
        hist = np.zeros((4, len(HISTORY_FEATURE_COLUMNS)), dtype=np.float32)
        pad = np.ones(4, dtype=np.bool_)
        take = prior[-4:]
        if take:
            hist[-len(take):] = np.stack(take)
            pad[-len(take):] = False
        tokens, text_mask = text_map.get((int(row.year), int(row.quarter)),
                                         (np.zeros(cfg.max_tokens, dtype=np.int64),
                                          np.zeros(cfg.max_tokens, dtype=np.bool_)))
        ndc_labeler = _labeler_entity_for_ndc(
            getattr(row, "ndc", ""), ndc_labelers)
        observed_entities = ([('manufactured_by', ndc_labeler)]
                             if ndc_labeler else [])
        entity_ids, relation_ids = _graph_context(
            graph_lookup, str(row.drug), model_cfg.entity_vocab,
            use_content_entities=cfg.use_drug_content_entities,
            prioritize_evidence=cfg.prioritize_graph_evidence,
            observed_entities=observed_entities,
            nodes=(8 if (cfg.use_ndc_labeler_entities
                         or cfg.use_drug_content_entities
                         or cfg.prioritize_graph_evidence) else None))
        graph_relation_types = sorted(
            {str(rel) for rel, _ in graph_lookup.get(f"drug:{row.drug}", [])}
            | {rel for rel, _ in observed_entities})
        if cfg.mask_drug_identity:
            # Research ablation for cold-start robustness: retain Arkansas and
            # observed graph relation nodes while removing the learned hash of
            # the drug itself. This tests whether identity memorization is
            # harming products unseen in an earlier rolling fold.
            entity_ids[0] = 0
        samples.append({
            "year": int(row.year), "quarter": int(row.quarter),
            "drug": str(row.drug), "target": float(row.target), "history": hist,
            "base_log": base_log_value,
            "transition_log": float(np.log1p(transition_base)),
            "persistence_log": float(np.log1p(max(float(row.value), 0.0))),
            "history_padding": pad, "tokens": tokens, "text_mask": text_mask,
            "text_key": (int(row.year), int(row.quarter)),
            "entity_ids": entity_ids, "relation_ids": relation_ids,
            "graph_relation_types": graph_relation_types,
            "metric_features": metric_context_by_row.get(
                (int(row.year), int(row.quarter), drug_name),
                np.zeros(METRIC_FEATURE_COUNT, dtype=np.float32)),
            "metric_context_report": metric_context_report,
        })
        prior.append(feature)
        observed_ratio = float(np.clip(float(row.target) / max(float(row.value), 1e-9), 0.0, 10.0))
        ratio_by_drug_quarter.setdefault(key, []).append(observed_ratio)
        ratio_by_quarter.setdefault(int(row.quarter), []).append(observed_ratio)
        all_ratios.append(observed_ratio)
    if not samples:
        raise ValueError("real quarterly training panel produced no samples")
    train_cut = cfg.train_last_year if train_last_year is None else train_last_year
    validation_cut = cfg.validation_year if validation_year is None else validation_year
    train = [x for x in samples if x["year"] <= train_cut]
    val = [x for x in samples if x["year"] == validation_cut]
    test = [x for x in samples if x["year"] > validation_cut]
    if not train or not val or not test:
        raise ValueError("temporal split must contain train, validation, and test rows")
    if cfg.state_target_mode not in {"change", "level"}:
        raise ValueError("state_target_mode must be 'change' or 'level'")
    # State labels are derived from the training slice only.  Applying these
    # fixed cut points to validation/test makes the state head a real
    # out-of-sample target rather than a label encoding fitted on future data.
    for split in (train, val, test):
        for x in split:
            target_log = np.log1p(max(x["target"], 0.0))
            x["state_value"] = (target_log - x["persistence_log"]
                                 if cfg.state_target_mode == "change" else target_log)
    train_state_values = np.asarray([x["state_value"] for x in train])
    state_threshold_values = np.quantile(
        train_state_values, np.arange(1, STATE_COUNT) / STATE_COUNT)
    state_thresholds = tuple(float(value) for value in state_threshold_values) \
        if np.all(np.diff(state_threshold_values) > 0) else None
    for split in (train, val, test):
        for x in split:
            if state_thresholds is None:
                # A degenerate training slice cannot support a defensible
                # five-state target. -1 is an explicit unavailable sentinel;
                # it is never passed to cross-entropy or scored as a state.
                x["state_label"] = -1
            else:
                x["state_label"] = _state_from_log(x["state_value"], state_thresholds)
    for split in (train, val, test):
        for x in split:
            x["state_thresholds"] = state_thresholds
    hist_train = np.concatenate([x["history"] for x in train], axis=0)
    mean = hist_train.mean(axis=0)
    std = hist_train.std(axis=0)
    std[std < 1e-6] = 1.0
    for split in (train, val, test):
        for x in split:
            x["history"] = (x["history"] - mean) / std
    return train, val, test, mean, std, text_map


def _batches(rows: Sequence[dict], batch_size: int, seed: int) -> Iterable[List[dict]]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start:start + batch_size]]


def _batch(rows: Sequence[dict], device: torch.device) -> Dict[str, Tensor]:
    return {
        "tokens": torch.from_numpy(np.stack([x["tokens"] for x in rows])).to(device),
        "text_mask": torch.from_numpy(np.stack([x["text_mask"] for x in rows])).to(device),
        "entity_ids": torch.from_numpy(np.stack([x["entity_ids"] for x in rows])).to(device),
        "relation_ids": torch.from_numpy(np.stack([x["relation_ids"] for x in rows])).to(device),
        "history": torch.from_numpy(np.stack([x["history"] for x in rows])).to(device),
        "metric_features": torch.from_numpy(np.stack([
            x.get("metric_features", np.zeros(METRIC_FEATURE_COUNT, dtype=np.float32))
            for x in rows])).to(device),
        "history_padding": torch.from_numpy(np.stack([x["history_padding"] for x in rows])).to(device),
        "target_log": torch.tensor([np.log1p(x["target"]) for x in rows], dtype=torch.float32, device=device),
        "base_log": torch.tensor([x["base_log"] for x in rows], dtype=torch.float32, device=device),
        "persistence_log": torch.tensor([x["persistence_log"] for x in rows], dtype=torch.float32, device=device),
        "transition_log": torch.tensor([x["transition_log"] for x in rows], dtype=torch.float32, device=device),
        "target_raw": np.asarray([x["target"] for x in rows], dtype=float),
        # The publishable demand evaluator trains only the numeric head and
        # does not construct state labels; -1 is the explicit unavailable
        # sentinel consumed by the optional state-loss path.
        "state_label": torch.tensor([x.get("state_label", -1) for x in rows],
                                     dtype=torch.long, device=device),
    }


@torch.no_grad()
def _cache_document_states(model: ArkansasPharmaMultimodalModel,
                           rows: Sequence[dict], device: torch.device) -> Dict[Tuple[int, int], Tensor]:
    """Encode each real quarter's document window exactly once."""
    model.eval()
    unique = {}
    for row in rows:
        unique[row["text_key"]] = (row["tokens"], row["text_mask"])
    result = {}
    keys = list(unique)
    for start in range(0, len(keys), 4):
        batch_keys = keys[start:start + 4]
        tokens = torch.from_numpy(np.stack([unique[k][0] for k in batch_keys])).to(device)
        masks = torch.from_numpy(np.stack([unique[k][1] for k in batch_keys])).to(device)
        states = model.news(tokens, masks).cpu()
        result.update({key: state for key, state in zip(batch_keys, states)})
    return result


def _loss(out: Dict[str, Tensor], target_delta: Tensor,
          *, target_log: Tensor | None = None,
          state_label: Tensor | None = None,
          state_loss_weight: float = 0.0,
          mode: str = "log_huber",
          state_loss_mode: str = "categorical") -> Tensor:
    if mode not in {"log_huber", "wape_weighted"}:
        raise ValueError(f"unsupported loss mode: {mode}")
    point = out["point"][:, 0, 0]
    q = out["quantiles"][:, 0, 0, :]
    if mode == "wape_weighted":
        if target_log is None:
            raise ValueError("target_log is required for wape_weighted loss")
        # WAPE is absolute error weighted by the observed volume.  Use only
        # the current training label to approximate that weighting while
        # keeping the model target in log space.  The cap prevents a few very
        # large products from dominating the gradient.
        weights = torch.exp(target_log).clamp(max=10.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        point_error = nn.functional.smooth_l1_loss(point, target_delta, reduction="none")
        point_loss = (point_error * weights).mean()
    else:
        weights = None
        point_loss = nn.functional.smooth_l1_loss(point, target_delta)
    quantile_losses = []
    for pred, level in zip(q.unbind(dim=-1), (0.1, 0.5, 0.9)):
        err = target_delta - pred
        quantile_loss = torch.maximum((level - 1) * err, level * err)
        quantile_losses.append((quantile_loss * weights).mean() if weights is not None
                               else quantile_loss.mean())
    # The median quantile is anchored to the point head; this avoids an
    # unconstrained interval head while retaining probabilistic outputs.
    loss = (point_loss + 0.25 * sum(quantile_losses)
            + 0.25 * nn.functional.smooth_l1_loss(q[:, 1], point.detach()))
    if state_loss_weight:
        if state_label is None:
            raise ValueError("state_label is required for state_loss_weight")
        if state_label.ndim != 1 or state_label.shape[0] != point.shape[0]:
            raise ValueError("state_label must have shape [batch]")
        state_logits = out["target_state_risk"][:, 0, 0, :]
        valid = state_label >= 0
        if valid.any():
            categorical_loss = nn.functional.cross_entropy(
                state_logits[valid], state_label[valid])
            if state_loss_mode == "categorical":
                state_loss = categorical_loss
            elif state_loss_mode == "ordinal":
                thresholds = torch.arange(
                    STATE_COUNT - 1, device=state_logits.device).unsqueeze(0)
                ordinal_target = (state_label[valid].unsqueeze(1) > thresholds).to(
                    state_logits.dtype)
                ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                    state_logits[valid, :-1], ordinal_target)
                state_loss = categorical_loss + ordinal_loss
            else:
                raise ValueError("state_loss_mode must be categorical or ordinal")
            loss = loss + float(state_loss_weight) * state_loss
    return loss


STATE_COUNT = 5


def state_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, object]:
    """Score the required five-state contract with exact and balanced accuracy."""
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted state arrays must have equal shape")
    valid_states = np.arange(STATE_COUNT)
    if not np.isin(actual, valid_states).all() or not np.isin(predicted, valid_states).all():
        raise ValueError("state values must be one of 0, 1, 2, 3, or 4")
    recalls = []
    for state in valid_states:
        mask = actual == state
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == state)))
    return {
        "rows": int(actual.size),
        "exact_accuracy": float(np.mean(actual == predicted)) if actual.size else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "state_counts": {str(state): int(np.sum(actual == state)) for state in valid_states},
    }


def _state_from_log(value: float, thresholds: tuple[float, ...] | None) -> int:
    """Map a log1p value to one of five fitted states, or return unavailable."""
    if thresholds is None:
        return -1
    return int(np.searchsorted(np.asarray(thresholds), value, side="right"))


def select_transition_blend_weight(actual: np.ndarray, transition: np.ndarray,
                                   neural: np.ndarray) -> tuple[float, float]:
    """Select a convex transition/neural blend using validation data only."""
    best_weight, best_wape = 0.0, float("inf")
    actual = np.asarray(actual, dtype=float)
    transition = np.asarray(transition, dtype=float)
    neural = np.asarray(neural, dtype=float)
    for weight in np.linspace(0.0, 1.0, 101):
        candidate = (1.0 - weight) * transition + weight * neural
        score = metrics_dict(actual, candidate)["wape"]
        if score < best_wape:
            best_weight, best_wape = float(weight), float(score)
    return best_weight, best_wape


@torch.no_grad()
def _predict_vectors(model: nn.Module, rows: Sequence[dict], device: torch.device,
                     text_cache: Dict[Tuple[int, int], Tensor] | None = None,
                     component_ablation: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return held-out actuals and log-residual point predictions.

    ``component_ablation`` zeroes a single model component at scored time only
    (news, graph, temporal, or deep_connections).  The model weights are never
    re-optimized and no ablation decision ever touches validation or test
    labels, so the marginal contribution is measured on already-held-out rows.
    """
    model.eval()
    actual, pred = [], []
    for rows_b in _batches(rows, 128, 0):
        b = _batch(rows_b, device)
        if text_cache is None:
            out = model(b["tokens"], b["entity_ids"], b["relation_ids"], b["history"],
                        b["text_mask"], b["history_padding"], b["metric_features"],
                        component_ablation=component_ablation)
        else:
            text = torch.stack([text_cache[x["text_key"]] for x in rows_b]).to(device)
            out = model.forward_from_states(text, b["entity_ids"], b["relation_ids"],
                                            b["history"], b["history_padding"],
                                            b["metric_features"],
                                            component_ablation=component_ablation)
        actual.extend(b["target_raw"])
        pred.extend(np.expm1(b["base_log"].cpu().numpy()
                             + out["point"][:, 0, 0].cpu().numpy()))
    return np.asarray(actual), np.maximum(np.asarray(pred), 0.0)


@torch.no_grad()
def _predict_state_vectors(
        model: ArkansasPharmaMultimodalModel, rows: Sequence[dict],
        device: torch.device,
        text_cache: Dict[Tuple[int, int], Tensor] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return held-out five-state labels and target-specific predictions."""
    model.eval()
    actual, predicted = [], []
    for rows_b in _batches(rows, 128, 0):
        b = _batch(rows_b, device)
        if text_cache is None:
            out = model(b["tokens"], b["entity_ids"], b["relation_ids"], b["history"],
                        b["text_mask"], b["history_padding"], b["metric_features"])
        else:
            text = torch.stack([text_cache[x["text_key"]] for x in rows_b]).to(device)
            out = model.forward_from_states(text, b["entity_ids"], b["relation_ids"],
                                            b["history"], b["history_padding"],
                                            b["metric_features"])
        labels = b["state_label"].cpu().numpy()
        predictions = out["target_state_risk"][:, 0, 0, :].argmax(dim=-1).cpu().numpy()
        valid = labels >= 0
        actual.extend(labels[valid].tolist())
        predicted.extend(predictions[valid].tolist())
    return np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)


@torch.no_grad()
def predict(model: nn.Module, rows: Sequence[dict], device: torch.device,
            text_cache: Dict[Tuple[int, int], Tensor] | None = None) -> Dict[str, float]:
    actual, pred = _predict_vectors(model, rows, device, text_cache)
    return metrics_dict(actual, pred)


def _train_fold(root: Path, train_cfg: TrainConfig, model_cfg: ModularModelConfig,
                checkpoint_name: str, *, train_last_year: int,
                validation_year: int) -> Dict:
    """Train one rolling fold and return the best-checkpoint model plus rows."""
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    train, val, test, mean, std, text_map = build_real_training_arrays(
        root, train_cfg, model_cfg, train_last_year=train_last_year,
        validation_year=validation_year)
    model = ArkansasPharmaMultimodalModel(model_cfg).to(device)
    if train_cfg.freeze_news:
        for parameter in model.news.parameters():
            parameter.requires_grad = False
    text_cache = (_cache_document_states(model, train + val + test, device)
                  if train_cfg.freeze_news else None)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    history = []
    best_val = float("inf")
    input_contract = modular_input_contract()
    checkpoint_dir = root / "model/artifacts/trained" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(train_cfg.epochs):
        model.train()
        total = 0.0
        for rows in _batches(train, train_cfg.batch_size, train_cfg.seed + epoch):
            b = _batch(rows, device)
            optimizer.zero_grad(set_to_none=True)
            if text_cache is None:
                out = model(b["tokens"], b["entity_ids"], b["relation_ids"], b["history"],
                            b["text_mask"], b["history_padding"], b["metric_features"])
            else:
                text = torch.stack([text_cache[x["text_key"]] for x in rows]).to(device)
                out = model.forward_from_states(text, b["entity_ids"], b["relation_ids"],
                                                b["history"], b["history_padding"],
                                                b["metric_features"])
            loss = _loss(out, b["target_log"] - b["base_log"],
                         target_log=b["target_log"], state_label=b["state_label"],
                         state_loss_weight=train_cfg.state_loss_weight,
                         mode=train_cfg.loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        val_metrics = predict(model, val, device, text_cache)
        record = {"epoch": epoch + 1, "train_loss": total / len(train), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        if val_metrics["wape"] < best_val:
            best_val = val_metrics["wape"]
            torch.save({"model": model.state_dict(), "config": asdict(train_cfg),
                        "model_config": asdict(model_cfg), "history_mean": mean.tolist(),
                        "history_std": std.tolist(),
                        "state_thresholds": train[0]["state_thresholds"],
                        "state_loss_weight": train_cfg.state_loss_weight,
                        "state_target_mode": train_cfg.state_target_mode,
                        "input_contract": input_contract,
                        "forecast_mode": ("operational" if input_contract["operational_ready"]
                                           else "research_training_only")},
                       checkpoint_dir / "best.pt")
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    return {"model": model, "train": train, "val": val, "test": test,
            "text_cache": text_cache, "device": device, "history": history,
            "text_map": text_map, "checkpoint": str(checkpoint_dir / "best.pt")}


def train_real_model(root: Path, train_cfg: TrainConfig,
                     model_cfg: ModularModelConfig | None = None,
                     checkpoint_name: str = "modular",
                     *, train_last_year: int | None = None,
                     validation_year: int | None = None) -> Dict:
    if train_cfg.operational_only:
        require_operational_modular()
    model_cfg = model_cfg or ModularModelConfig(
        max_tokens=8_192, horizon_count=1, target_count=1,
        history_features=len(HISTORY_FEATURE_COLUMNS))
    fold = _train_fold(root, train_cfg, model_cfg, checkpoint_name,
                       train_last_year=train_last_year
                       if train_last_year is not None else train_cfg.train_last_year,
                       validation_year=validation_year
                       if validation_year is not None else train_cfg.validation_year)
    model, device, text_cache = fold["model"], fold["device"], fold["text_cache"]
    train, val, test, text_map, history = (fold["train"], fold["val"], fold["test"],
                                           fold["text_map"], fold["history"])
    input_contract = modular_input_contract()
    result = {
        "model_inventory": model_inventory(model),
        "training_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "input_contract": input_contract,
        "forecast_mode": ("operational" if input_contract["operational_ready"]
                           else "research_training_only"),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "text_quarters": len(text_map),
        "history": history,
        "validation": predict(model, val, device, text_cache),
        "test": predict(model, test, device, text_cache),
        "state_thresholds": train[0]["state_thresholds"],
        "state_loss_weight": train_cfg.state_loss_weight,
        "state_target_mode": train_cfg.state_target_mode,
        "validation_state": state_metrics(*_predict_state_vectors(model, val, device, text_cache)),
        "test_state": state_metrics(*_predict_state_vectors(model, test, device, text_cache)),
        "checkpoint": fold["checkpoint"],
    }
    # Persist row-level held-out predictions so baseline blends and subgroup
    # audits can be reproduced without another expensive Transformer pass.
    prediction_rows = []
    train_drugs = {row["drug"] for row in train}
    state_thresholds = train[0]["state_thresholds"]
    model.eval()
    with torch.no_grad():
        for split_name, rows in (("validation", val), ("test", test)):
            for rows_b in _batches(rows, 128, 0):
                b = _batch(rows_b, device)
                if text_cache is None:
                    out = model(b["tokens"], b["entity_ids"], b["relation_ids"], b["history"],
                                b["text_mask"], b["history_padding"], b["metric_features"])
                else:
                    text = torch.stack([text_cache[x["text_key"]] for x in rows_b]).to(device)
                    out = model.forward_from_states(text, b["entity_ids"], b["relation_ids"],
                                                    b["history"], b["history_padding"],
                                                    b["metric_features"])
                neural = np.maximum(np.expm1(b["base_log"].cpu().numpy()
                                              + out["point"][:, 0, 0].cpu().numpy()), 0.0)
                state_prediction = out["target_state_risk"][:, 0, 0, :].argmax(
                    dim=-1).cpu().numpy()
                predicted_target_log = (b["base_log"] + out["point"][:, 0, 0]).cpu().numpy()
                predicted_state_value = (
                    predicted_target_log - b["persistence_log"].cpu().numpy()
                    if train_cfg.state_target_mode == "change" else predicted_target_log)
                state_numeric_prediction = np.asarray([
                    _state_from_log(value, state_thresholds)
                    for value in predicted_state_value], dtype=int)
                for row, point, state_point, numeric_state in zip(
                        rows_b, neural, state_prediction, state_numeric_prediction):
                    prediction_rows.append({"split": split_name, "year": row["year"],
                                            "quarter": row["quarter"],
                                            "drug": row["drug"],
                                            "history_count": int((~row["history_padding"]).sum()),
                                            "seen_in_train": row["drug"] in train_drugs,
                                            "has_ingredient_edge": "contains" in row["graph_relation_types"],
                                            "has_manufacturer_edge": "manufactured_by" in row["graph_relation_types"],
                                            "has_supplier_edge": bool({"manufactured_by", "api_supplied_by", "owned_by"}
                                                                        & set(row["graph_relation_types"])),
                                            "actual": row["target"],
                                            "state_actual": row["state_label"],
                                            "state_prediction": int(state_point),
                                            "state_numeric": int(numeric_state),
                                            "state_persistence": _state_from_log(
                                                0.0 if train_cfg.state_target_mode == "change"
                                                else row["persistence_log"], state_thresholds),
                                            "persistence": np.expm1(row["persistence_log"]),
                                            "transition": np.expm1(row["transition_log"]),
                                            "neural": float(point)})
    prediction_frame = pd.DataFrame(prediction_rows)
    validation = prediction_frame[prediction_frame["split"].eq("validation")]
    best_weight, best_wape = select_transition_blend_weight(
        validation["actual"].to_numpy(float), validation["transition"].to_numpy(float),
        validation["neural"].to_numpy(float))
    prediction_frame["blended"] = (
        (1.0 - best_weight) * prediction_frame["transition"]
        + best_weight * prediction_frame["neural"])
    io.write_csv(prediction_frame,
                 root / "model/artifacts/evaluation/modular_predictions.csv")
    prediction_metrics = {}
    for split_name, group in prediction_frame.groupby("split"):
        prediction_metrics[split_name] = {
            name: metrics_dict(group["actual"].to_numpy(float), group[name].to_numpy(float))
            for name in ("persistence", "transition", "neural", "blended")
        }
    result["prediction_metrics"] = prediction_metrics
    result["prediction_state_metrics"] = {}
    for split_name, group in prediction_frame.groupby("split"):
        actual = group["state_actual"].to_numpy(int)
        result["prediction_state_metrics"][split_name] = {}
        for name in ("state_prediction", "state_numeric", "state_persistence"):
            predicted = group[name].to_numpy(int)
            valid = (actual >= 0) & (predicted >= 0)
            metric_name = {"state_prediction": "model",
                           "state_numeric": "numeric",
                           "state_persistence": "persistence"}[name]
            result["prediction_state_metrics"][split_name][metric_name] = (
                state_metrics(actual[valid], predicted[valid]))
    subgroup_metrics = {}
    for split_name, split_frame in prediction_frame.groupby("split", sort=True):
        subgroup_definitions = {
            "all": np.ones(len(split_frame), dtype=bool),
            "seen_in_train": split_frame["seen_in_train"].to_numpy(bool),
            "cold_start": ~split_frame["seen_in_train"].to_numpy(bool),
            "has_ingredient_edge": split_frame["has_ingredient_edge"].to_numpy(bool),
            "has_manufacturer_edge": split_frame["has_manufacturer_edge"].to_numpy(bool),
            "has_supplier_edge": split_frame["has_supplier_edge"].to_numpy(bool),
            "history_at_least_two": split_frame["history_count"].to_numpy(int) >= 2,
        }
        subgroup_metrics[split_name] = {}
        for subgroup, mask in subgroup_definitions.items():
            group = split_frame.loc[mask]
            if group.empty:
                continue
            subgroup_metrics[split_name][subgroup] = {
                "rows": int(len(group)),
                "actual_sum": float(group["actual"].sum()),
                **{name: metrics_dict(group["actual"].to_numpy(float), group[name].to_numpy(float))
                   for name in ("persistence", "transition", "neural", "blended")},
            }
    result["subgroup_metrics"] = subgroup_metrics
    result["validation_selected_blend_weight"] = best_weight
    result["validation_selected_blend_wape"] = best_wape
    io.write_json(result, root / "model/artifacts/evaluation/modular_training_metrics.json")
    return result


def evaluate_modular_component_rolling(
        root: Path, *, train_cfg: TrainConfig | None = None,
        model_cfg: ModularModelConfig | None = None,
        min_train_years: int = 4) -> Dict:
    """Report each model component's marginal contribution under rolling origins.

    Each fold trains a single full model on its train slice, selects the blend
    weight on validation only, and then scores the model with each named
    component held out at scored time.  Component names are the states exposed
    by :class:`ArkansasPharmaMultimodalModel`: ``news``, ``graph``,
    ``temporal``, and ``deep_connections``.

    This is a marginal-contribution diagnostic, not a promotion gate.  The
    per-component delta is the change in validation-selected blended test WAPE
    when that component is zeroed at scored time.  It is research-only by
    construction: the full-model rolling gate in
    :func:`evaluate_modular_rolling` remains authoritative and component rows
    never become a publishable candidate.
    """
    cfg = train_cfg or TrainConfig(epochs=1, batch_size=256, freeze_news=True)
    cfg_model = model_cfg or research_small_model_config()
    input_contract = modular_input_contract()
    panel_path = root / "model/artifacts/panel/medicaid_sdud_quarterly_panel.csv"
    panel = pd.read_csv(panel_path, usecols=["year"])
    cutoffs = modular_rolling_cutoffs(panel["year"].tolist(), min_train_years)
    folds = []
    for train_last, validation_year in cutoffs:
        fold = _train_fold(
            root, cfg, cfg_model, f"modular_component_{train_last}",
            train_last_year=train_last, validation_year=validation_year)
        model, device, text_cache = fold["model"], fold["device"], fold["text_cache"]
        val, test = fold["val"], fold["test"]
        val_actual = np.asarray([x["target"] for x in val], dtype=float)
        test_actual = np.asarray([x["target"] for x in test], dtype=float)
        val_transition = np.expm1(np.asarray([x["transition_log"] for x in val], dtype=float))
        test_transition = np.expm1(np.asarray([x["transition_log"] for x in test], dtype=float))
        baseline = min(metrics_dict(test_actual, test_transition)["wape"],
                       metrics_dict(test_actual,
                                    np.expm1(np.asarray([x["persistence_log"] for x in test],
                                                        dtype=float)))["wape"])
        components = {}
        full_blended_wape = None
        # Score the intact model first so every ablation has a defined
        # held-out reference when its delta is recorded.
        for name in ("full", "news", "graph", "temporal", "deep_connections"):
            _, neural = _predict_vectors(model, val, device, text_cache,
                                         None if name == "full" else name)
            weight, _ = select_transition_blend_weight(val_actual, val_transition, neural)
            test_neural = _predict_vectors(model, test, device, text_cache,
                                           None if name == "full" else name)[1]
            blended = (1.0 - weight) * test_transition + weight * test_neural
            blended_wape = metrics_dict(test_actual, blended)["wape"]
            record = {
                "blend_weight": float(weight),
                "test_blended_wape": float(blended_wape),
                "test_neural_wape": float(metrics_dict(test_actual, test_neural)["wape"]),
                "improvement_vs_strongest_naive": float(
                    (baseline - blended_wape) / baseline) if baseline > 0 else float("nan"),
            }
            if name == "full":
                full_blended_wape = blended_wape
                components["full"] = record
            else:
                record["delta_vs_full_test_blended_wape"] = float(
                    blended_wape - full_blended_wape)
                components[name] = record
        folds.append({
            "train_last_year": train_last, "validation_year": validation_year,
            "test_start_year": validation_year + 1, "test_rows": len(test),
            "strongest_naive_wape": float(baseline), "components": components,
        })
    if not folds:
        return {"fold_count": 0, "components": {}, "folds": [],
                "publishable_component_candidate": False,
                "promotion_reason": "research-only: no eligible rolling folds"}
    component_summary = {}
    for name in ("news", "graph", "temporal", "deep_connections"):
        deltas = [fold["components"][name]["delta_vs_full_test_blended_wape"]
                  for fold in folds]
        improvements = [fold["components"][name]["improvement_vs_strongest_naive"]
                        for fold in folds]
        component_summary[name] = {
            "mean_test_blended_wape": float(np.mean([
                fold["components"][name]["test_blended_wape"] for fold in folds])),
            "mean_delta_vs_full": float(np.mean(deltas)),
            "folds_where_removal_hurt": int(sum(d > 1e-9 for d in deltas)),
            "mean_improvement_vs_strongest_naive": float(np.mean(improvements)),
            "mean_test_neural_wape": float(np.mean([
                fold["components"][name]["test_neural_wape"] for fold in folds])),
            "mean_neural_delta_vs_full": float(np.mean([
                fold["components"][name]["test_neural_wape"]
                - fold["components"]["full"]["test_neural_wape"]
                for fold in folds])),
            "folds_where_neural_removal_hurt": int(sum(
                fold["components"][name]["test_neural_wape"]
                > fold["components"]["full"]["test_neural_wape"] + 1e-9
                for fold in folds)),
        }
    return {
        "split": "rolling_origin_modular_component_marginal",
        "fold_count": int(len(folds)),
        "min_train_years": int(min_train_years),
        "ablations": "scored_time_state_zeroing_on_held_out_rows",
        "training_config": asdict(cfg),
        "model_config": asdict(cfg_model),
        "input_contract": input_contract,
        "forecast_mode": ("operational" if input_contract["operational_ready"]
                           else "research_training_only"),
        "components": component_summary,
        "publishable_component_candidate": False,
        "promotion_reason": (
            "component-level diagnostics are research-only; the full-model "
            "rolling gate in evaluate-modular remains authoritative and no "
            "component row is a promotion candidate"),
        "folds": folds,
    }


def modular_rolling_cutoffs(years: Iterable[int], min_train_years: int = 4) -> list[tuple[int, int]]:
    """Return (train_last_year, validation_year) chronological fold cutoffs."""
    unique = sorted({int(year) for year in years})
    if len(unique) < min_train_years + 2:
        return []
    # Leave at least one year after validation for the held-out test window.
    return [(unique[i], unique[i + 1])
            for i in range(min_train_years - 1, len(unique) - 2)]


def evaluate_modular_rolling(root: Path, *, train_cfg: TrainConfig | None = None,
                             model_cfg: ModularModelConfig | None = None,
                             min_train_years: int = 4) -> Dict:
    """Run research-model rolling folds with a validation-selected baseline.

    Each fold trains through year Y, selects on Y+1, and tests on all later
    years. The reported candidate is publishable only if every fold beats the
    strongest persistence/transition baseline by the configured threshold.
    """
    cfg = train_cfg or TrainConfig(epochs=1, batch_size=256, freeze_news=True)
    cfg_model = model_cfg or research_small_model_config()
    input_contract = modular_input_contract()
    panel_path = root / "model/artifacts/panel/medicaid_sdud_quarterly_panel.csv"
    panel = pd.read_csv(panel_path, usecols=["year"])
    cutoffs = modular_rolling_cutoffs(panel["year"].tolist(), min_train_years)
    folds = []
    for train_last, validation_year in cutoffs:
        result = train_real_model(
            root, cfg, model_cfg=cfg_model,
            checkpoint_name=f"modular_research_rolling_{train_last}",
            train_last_year=train_last, validation_year=validation_year)
        test = result["prediction_metrics"].get("test", {})
        neural = float(test.get("neural", {}).get("wape", np.nan))
        # The deployed policy is selected on validation WAPE.  Score that
        # policy in rolling evaluation rather than silently evaluating the
        # unblended neural head.
        selected = float(test.get("blended", {}).get("wape", np.nan))
        baseline = min(float(test.get(name, {}).get("wape", np.nan))
                       for name in ("persistence", "transition"))
        improvement = float((baseline - selected) / baseline) if baseline > 0 else float("nan")
        folds.append({"train_last_year": train_last, "validation_year": validation_year,
                      "test_start_year": validation_year + 1,
                      "test_rows": result["rows"]["test"],
                      "neural_wape": neural, "selected_wape": selected,
                      "selected_blend_weight": float(result["validation_selected_blend_weight"]),
                      "strongest_naive_wape": baseline,
                      "improvement_vs_strongest_naive": improvement,
                      "subgroups": {
                          name: {
                              "rows": int(values["rows"]),
                              "neural_wape": float(values["neural"]["wape"]),
                              "selected_wape": float(values["blended"]["wape"]),
                              "transition_wape": float(values["transition"]["wape"]),
                          }
                          for name, values in result.get("subgroup_metrics", {})
                          .get("test", {}).items()
                      }})
    frame = pd.DataFrame(folds)
    finite = frame["improvement_vs_strongest_naive"].replace([np.inf, -np.inf], np.nan).dropna() if not frame.empty else pd.Series(dtype=float)
    mean_improvement = float(finite.mean()) if len(finite) else float("nan")
    publishable = bool(len(frame) >= 3 and len(finite) == len(frame)
                      and (frame["improvement_vs_strongest_naive"] >= 0.10).all())
    return {
        "split": "rolling_origin_modular_next_quarter",
        "fold_count": int(len(frame)),
        "min_train_years": int(min_train_years),
        "training_config": asdict(cfg),
        "model_config": asdict(cfg_model),
        "input_contract": input_contract,
        "forecast_mode": ("operational" if input_contract["operational_ready"]
                           else "research_training_only"),
        "mean_improvement_vs_strongest_naive": mean_improvement,
        "publishable_rolling_candidate": publishable,
        "publishability_reason": (
            "every fold beats both temporal baselines by at least 10%"
            if publishable else
            "research-only: rolling model does not meet the all-fold 10% gate"),
        "folds": frame.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--freeze-news", action="store_true")
    parser.add_argument("--baseline-mode", choices=["persistence", "transition", "absolute"], default="persistence")
    parser.add_argument("--loss-mode", choices=["log_huber", "wape_weighted"], default="log_huber")
    parser.add_argument("--state-target-mode", choices=["change", "level"], default="change",
                        help="Five-state target: next-period demand change (default) or level.")
    parser.add_argument("--mask-drug-identity", action="store_true",
                        help="Research ablation that masks the learned drug hash.")
    parser.add_argument("--use-drug-content-entities", action="store_true",
                        help="Research ablation adding shared drug-name graph tokens.")
    parser.add_argument("--prioritize-graph-evidence", action="store_true",
                        help="Research ablation prioritizing observed ingredient/supplier edges.")
    parser.add_argument("--use-ndc-labeler-entities", action="store_true",
                        help="Research ablation adding FDA-backed NDC labeler entities.")
    parser.add_argument("--use-nadac-price-features", action="store_true",
                        help="Research ablation adding exact-NDC NADAC context.")
    parser.add_argument("--use-historical-metric-context", action="store_true",
                        help="Research ablation adding dated exact-NDC shortage/recall context.")
    parser.add_argument("--small", action="store_true",
                        help="Use a CPU-testable research model; never a promotion artifact.")
    parser.add_argument("--operational-only", action="store_true",
                        help="Fail unless every modular feature is rebuildable from "
                             "operational inputs at application time.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.operational_only:
        require_operational_modular()
    model_cfg = research_small_model_config() if args.small else None
    result = train_real_model(Path(args.root).resolve(), TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        freeze_news=args.freeze_news, device=args.device, baseline_mode=args.baseline_mode,
        loss_mode=args.loss_mode, mask_drug_identity=args.mask_drug_identity,
        state_target_mode=args.state_target_mode,
        use_drug_content_entities=args.use_drug_content_entities,
        prioritize_graph_evidence=args.prioritize_graph_evidence,
        use_ndc_labeler_entities=args.use_ndc_labeler_entities,
        use_nadac_price_features=args.use_nadac_price_features,
        use_historical_metric_context=args.use_historical_metric_context,
        operational_only=args.operational_only),
        model_cfg=model_cfg, checkpoint_name="modular_research_small" if args.small else "modular")
    print(json.dumps({"rows": result["rows"], "validation": result["validation"],
                      "test": result["test"],
                      "validation_state": result["validation_state"],
                      "test_state": result["test_state"],
                      "prediction_state_metrics": result["prediction_state_metrics"],
                      "inventory": result["model_inventory"]}, indent=2))


if __name__ == "__main__":
    main()
