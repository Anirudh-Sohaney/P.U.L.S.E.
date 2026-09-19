"""Inspectable multimodal architecture for the Arkansas pharmaceutical model.

This module is deliberately separate from the current statistical production
baseline.  It defines the trainable architecture and its contracts; it does
not pretend that randomly initialized weights are a forecast model.  Training
and promotion require real, time-stamped labels and the existing evaluation
gates.

The default configuration is a roughly 300M-class active model:

    document encoder -> typed graph reasoning -> temporal fusion -> Arkansas heads

The document encoder is an encoder-only Transformer, not a keyword counter.
The graph module keeps disease, clinical, supply, and geography streams
inspectable.  The temporal module consumes historical state vectors and emits
multi-horizon representations.  The final heads produce point, quantile, and
state-risk outputs with an evidence-preserving driver vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModularModelConfig:
    vocab_size: int = 32_000
    max_tokens: int = 8_192
    text_width: int = 1_024
    text_layers: int = 16
    text_heads: int = 16
    text_ffn: int = 4_096
    entity_vocab: int = 100_000
    entity_width: int = 400
    graph_layers: int = 4
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_ffn: int = 1_536
    interaction_layers: int = 2
    interaction_heads: int = 8
    interaction_ffn: int = 1_536
    connection_layers: int = 2
    connection_heads: int = 8
    connection_ffn: int = 1_536
    # The history vector is assembled by the active data adapter. Production
    # and research adapters may expose different documented widths.
    history_features: int = 38
    # Eight qualified values plus one explicit missingness channel each.
    metric_feature_count: int = 16
    horizon_count: int = 5
    target_count: int = 6
    # Promoted categorical targets require five states. The separate
    # three-channel aggregate state_risk head is retained for compatibility;
    # target_state_risk is the contract-facing head.
    state_count: int = 5


class NewsDocumentEncoder(nn.Module):
    """Long-context text encoder producing a document representation."""

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.text_width)
        self.position_embedding = nn.Embedding(cfg.max_tokens, cfg.text_width)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_width,
            nhead=cfg.text_heads,
            dim_feedforward=cfg.text_ffn,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.text_layers)
        self.norm = nn.LayerNorm(cfg.text_width)

    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, tokens]")
        if token_ids.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError("token sequence exceeds configured context length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        safe_attention_mask = attention_mask
        if attention_mask is not None:
            # PyTorch attention is undefined when every token in a row is
            # masked.  Keep missing-document rows numerically safe, then
            # explicitly return a zero representation for those rows.
            safe_attention_mask = attention_mask.bool().clone()
            empty = ~safe_attention_mask.any(dim=1)
            safe_attention_mask[empty, 0] = True
        key_padding_mask = None if safe_attention_mask is None else ~safe_attention_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        if attention_mask is None:
            return x[:, 0]
        weights = attention_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled * (weights.sum(dim=1) > 0).to(x.dtype)


class TypedGraphReasoner(nn.Module):
    """Relation-aware message passing over typed entity evidence.

    ``entity_ids`` has shape [batch, nodes] and ``relation_ids`` has shape
    [batch, nodes, nodes].  Relation embeddings gate messages; this makes the
    disease, clinical, supply-chain, geography, and temporal paths explicit in
    the input contract rather than hiding them in a flat feature vector.
    """

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        # ID 0 is reserved for padding/unknown entities so cold-start
        # ablations do not turn missing identity into a learned entity.
        self.entity_embedding = nn.Embedding(
            cfg.entity_vocab, cfg.entity_width, padding_idx=0)
        self.relation_embedding = nn.Embedding(32, cfg.entity_width, padding_idx=0)
        # No bias: a node with no valid incoming edge must receive no learned
        # graph message rather than an unconditional relation-independent term.
        self.message_projection = nn.Linear(cfg.entity_width, cfg.entity_width, bias=False)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.entity_width,
                nhead=8,
                dim_feedforward=cfg.entity_width * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(cfg.graph_layers)
        ])
        self.norm = nn.LayerNorm(cfg.entity_width)

    def forward(self, entity_ids: Tensor, relation_ids: Tensor) -> Tensor:
        if entity_ids.ndim != 2 or relation_ids.ndim != 3:
            raise ValueError("entity_ids=[batch,nodes], relation_ids=[batch,nodes,nodes]")
        if relation_ids.shape[:2] != entity_ids.shape:
            raise ValueError("relation matrix does not match entity nodes")
        node_mask = entity_ids.ne(0)
        safe_node_mask = node_mask.clone()
        empty = ~safe_node_mask.any(dim=1)
        if empty.any():
            safe_node_mask[empty, 0] = True
        edge_mask = (relation_ids > 0)
        edge_mask = edge_mask & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        x = self.entity_embedding(entity_ids)
        for layer in self.layers:
            # relation_ids[i, j] describes a directed edge whose receiver is
            # node i and source is node j. Unknown/padding relations are not
            # allowed to create messages, and padded nodes never enter the
            # degree normalization or Transformer attention.
            rel = self.relation_embedding(relation_ids.clamp_min(0).clamp_max(31))
            source = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
            messages = source + rel
            messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)
            degree = edge_mask.sum(dim=2, keepdim=True).clamp_min(1).to(messages.dtype)
            messages = self.message_projection(messages.sum(dim=2) / degree)
            x = layer(x + messages, src_key_padding_mask=~safe_node_mask)
            x = x.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        x = self.norm(x)
        valid = node_mask.to(x.dtype).unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class TemporalFusionReasoner(nn.Module):
    """Causal temporal encoder for observed history and external state."""

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        # The feature width is part of the serialized data contract.  Keeping
        # it explicit makes parameter accounting possible before the first
        # batch is seen and prevents lazy, un-auditable parameters.
        self.input_projection = nn.Linear(cfg.history_features, cfg.entity_width)
        self.metric_projection = nn.Linear(
            cfg.metric_feature_count, cfg.entity_width, bias=False)
        self.metric_norm = nn.LayerNorm(cfg.entity_width)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.entity_width,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.temporal_ffn,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        self.norm = nn.LayerNorm(cfg.entity_width)

    def metric_state(self, metric_features: Optional[Tensor], batch_size: int,
                     device: torch.device) -> Tensor:
        if metric_features is None:
            return torch.zeros(batch_size, self.input_projection.out_features, device=device)
        if metric_features.ndim != 2 or metric_features.shape[0] != batch_size:
            raise ValueError("metric_features must have shape [batch, metric_feature_count]")
        if metric_features.shape[1] != self.metric_projection.in_features:
            raise ValueError("metric_features width does not match model configuration")
        return self.metric_norm(self.metric_projection(metric_features))

    def forward(self, history: Tensor, padding_mask: Optional[Tensor] = None,
                metric_features: Optional[Tensor] = None) -> Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, features]")
        x = self.input_projection(history)
        metric_state = self.metric_state(metric_features, history.shape[0], history.device)
        time = x.shape[1]
        causal = torch.triu(torch.ones(time, time, device=x.device, dtype=torch.bool), diagonal=1)
        safe_padding_mask = padding_mask
        empty = None
        if padding_mask is not None:
            # A new entity can have no observed history.  PyTorch attention
            # produces NaNs when every timestep is masked, so temporarily
            # unmask the final slot and zero the pooled result below.
            safe_padding_mask = padding_mask.bool().clone()
            empty = safe_padding_mask.all(dim=1)
            safe_padding_mask[empty, -1] = False
        x = self.encoder(x, mask=causal, src_key_padding_mask=safe_padding_mask)
        if empty is not None and empty.any():
            # Earlier causal positions still have no visible key in an
            # all-padded row; clear their undefined attention result.
            x[empty] = 0.0
        # Causal attention can also leave fully padded prefix queries without
        # a visible key when the first observation arrives late in the window.
        # Those positions are excluded from pooling, but clearing them keeps
        # downstream diagnostics and cached states finite.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if padding_mask is None:
            return self.norm(x[:, -1])
        valid = (~padding_mask).to(x.dtype).unsqueeze(-1)
        return self.norm((x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)) + metric_state


class CrossModalInteractionReasoner(nn.Module):
    """Learn interactions among text, graph exposure, and temporal context.

    The three upstream representations are kept as typed tokens rather than
    concatenated immediately. This gives the model a dedicated stage in which
    news states can attend to supply/geography graph evidence and observed
    economic or demand history before the final output heads are reached.
    """

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        self.news_projection = nn.Linear(cfg.text_width, cfg.entity_width)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.entity_width,
            nhead=cfg.interaction_heads,
            dim_feedforward=cfg.interaction_ffn,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.interaction_layers)
        self.norm = nn.LayerNorm(cfg.entity_width)

    def forward(self, news: Tensor, graph: Tensor, temporal: Tensor) -> Tensor:
        if news.ndim != 2 or graph.ndim != 2 or temporal.ndim != 2:
            raise ValueError("cross-modal states must have shape [batch, width]")
        if not (news.shape[0] == graph.shape[0] == temporal.shape[0]):
            raise ValueError("cross-modal states must have matching batch sizes")
        tokens = torch.stack([self.news_projection(news), graph, temporal], dim=1)
        return self.norm(self.encoder(tokens))


class DeepConnectionReasoner(nn.Module):
    """Learn a second-stage interaction among typed upstream states.

    The role embedding keeps news, graph/geography, and temporal/economic
    states distinguishable after cross-modal fusion. This is intentionally a
    learned representation stage; no political, geographic, or supply rule is
    encoded as a fixed coefficient.
    """

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        self.role_embedding = nn.Parameter(torch.zeros(1, 3, cfg.entity_width))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.entity_width,
            nhead=cfg.connection_heads,
            dim_feedforward=cfg.connection_ffn,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.connection_layers)
        self.norm = nn.LayerNorm(cfg.entity_width)

    def forward(self, interaction: Tensor) -> Tensor:
        if interaction.ndim != 3 or interaction.shape[1] != 3:
            raise ValueError("interaction must have shape [batch, 3, width]")
        return self.norm(self.encoder(interaction + self.role_embedding))


class ArkansasPredictionHeads(nn.Module):
    """Arkansas outputs: point, quantiles, state risk, and driver attribution."""

    def __init__(self, cfg: ModularModelConfig) -> None:
        super().__init__()
        width = cfg.entity_width * 3
        self.trunk = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.point = nn.Linear(width, cfg.horizon_count * cfg.target_count)
        self.quantiles = nn.Linear(width, cfg.horizon_count * cfg.target_count * 3)
        self.target_state_risk = nn.Linear(
            width, cfg.horizon_count * cfg.target_count * cfg.state_count)
        # Retained for compatibility with older research consumers. New
        # metric heads should use target_state_risk instead.
        self.state_risk = nn.Linear(width, cfg.horizon_count * 3)
        self.driver = nn.Linear(width, 10)
        self.horizon_count = cfg.horizon_count
        self.target_count = cfg.target_count
        self.state_count = cfg.state_count

    def forward(self, fused: Tensor) -> Dict[str, Tensor]:
        z = self.trunk(fused)
        b = fused.shape[0]
        return {
            "point": self.point(z).view(b, self.horizon_count, self.target_count),
            "quantiles": self.quantiles(z).view(b, self.horizon_count, self.target_count, 3),
            "state_risk": self.state_risk(z).view(b, self.horizon_count, 3),
            "target_state_risk": self.target_state_risk(z).view(
                b, self.horizon_count, self.target_count, self.state_count),
            "driver_contribution": self.driver(z),
        }


class ArkansasPharmaMultimodalModel(nn.Module):
    """Composable document/graph/temporal model with auditable intermediate states."""

    def __init__(self, cfg: ModularModelConfig = ModularModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.news = NewsDocumentEncoder(cfg)
        self.graph = TypedGraphReasoner(cfg)
        self.temporal = TemporalFusionReasoner(cfg)
        self.cross_modal = CrossModalInteractionReasoner(cfg)
        self.deep_connections = DeepConnectionReasoner(cfg)
        self.heads = ArkansasPredictionHeads(cfg)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _fuse_with_ablation(
        self,
        text: Tensor,
        graph: Tensor,
        temporal: Tensor,
        component_ablation: Optional[str] = None,
    ) -> tuple[Tensor, Tensor]:
        """Run cross-modal, deep-connection, and fusion with one component held.

        ``component_ablation`` zeroes a single upstream state (or bypasses the
        deep-connection stage) at scored time.  This is a marginal-contribution
        diagnostic for an already-trained model; it is not a re-training
        ablation and it is never applied on training rows.
        """
        if component_ablation == "news":
            text = torch.zeros_like(text)
        elif component_ablation == "graph":
            graph = torch.zeros_like(graph)
        elif component_ablation == "temporal":
            temporal = torch.zeros_like(temporal)
        interaction = self.cross_modal(text, graph, temporal)
        if component_ablation == "deep_connections":
            deep_connections = interaction
        else:
            deep_connections = self.deep_connections(interaction)
        return interaction, deep_connections

    def forward(
        self,
        token_ids: Tensor,
        entity_ids: Tensor,
        relation_ids: Tensor,
        history: Tensor,
        attention_mask: Optional[Tensor] = None,
        history_padding_mask: Optional[Tensor] = None,
        metric_features: Optional[Tensor] = None,
        component_ablation: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        text = self.news(token_ids, attention_mask)
        graph = self.graph(entity_ids, relation_ids)
        temporal = self.temporal(history, history_padding_mask, metric_features)
        interaction, deep_connections = self._fuse_with_ablation(
            text, graph, temporal, component_ablation)
        fused = deep_connections.flatten(start_dim=1)
        out = self.heads(fused)
        out["intermediate_news"] = text
        out["intermediate_graph"] = graph
        out["intermediate_temporal"] = temporal
        out["intermediate_metric_context"] = self.temporal.metric_state(
            metric_features, history.shape[0], history.device)
        out["intermediate_cross_modal"] = interaction
        out["intermediate_deep_connections"] = deep_connections
        return out

    def forward_from_states(
        self,
        text: Tensor,
        entity_ids: Tensor,
        relation_ids: Tensor,
        history: Tensor,
        history_padding_mask: Optional[Tensor] = None,
        metric_features: Optional[Tensor] = None,
        component_ablation: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Run graph/temporal/prediction layers from cached document states.

        Historical document windows are shared by many drug rows.  Training
        can therefore encode each real quarter once, cache its representation,
        and avoid retaining a full Transformer activation graph for every
        repeated row.  The cached state is only valid for a frozen document
        encoder; any text fine-tuning must use :meth:`forward` directly.
        """
        graph = self.graph(entity_ids, relation_ids)
        temporal = self.temporal(history, history_padding_mask, metric_features)
        interaction, deep_connections = self._fuse_with_ablation(
            text, graph, temporal, component_ablation)
        fused = deep_connections.flatten(start_dim=1)
        out = self.heads(fused)
        out["intermediate_news"] = text
        out["intermediate_graph"] = graph
        out["intermediate_temporal"] = temporal
        out["intermediate_metric_context"] = self.temporal.metric_state(
            metric_features, history.shape[0], history.device)
        out["intermediate_cross_modal"] = interaction
        out["intermediate_deep_connections"] = deep_connections
        return out


def model_inventory(model: ArkansasPharmaMultimodalModel) -> Dict[str, object]:
    """Return reproducible layer accounting for metadata and audits."""
    total = model.parameter_count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layer_names = ("news", "graph", "temporal", "cross_modal", "deep_connections", "heads")
    layer_parameter_counts = {
        name: int(sum(parameter.numel() for parameter in getattr(model, name).parameters()))
        for name in layer_names
    }
    return {
        "architecture": "news_encoder_masked_relation_message_graph_temporal_deep_connections_arkansas_heads",
        "layer_count": len(layer_names),
        "layer_names": list(layer_names),
        "layer_parameter_counts": layer_parameter_counts,
        "layer_contracts": {
            "news": "timestamped document tokens -> pooled document state",
            "graph": "masked typed directed entities -> pooled geography/supplier state",
            "temporal": "causal observed history + optional qualified metric context -> current temporal state",
            "cross_modal": "news + graph + temporal states -> typed interaction tokens",
            "deep_connections": "typed interaction tokens -> multi-domain connection states",
            "heads": "connection states -> point, quantile, state-risk, driver outputs",
        },
        "active_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }
