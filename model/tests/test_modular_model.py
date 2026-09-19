import pytest


torch = pytest.importorskip("torch")

from arkansas_pharma_signal.modular_model import (  # noqa: E402
    ArkansasPharmaMultimodalModel,
    ModularModelConfig,
    model_inventory,
)


def test_default_architecture_is_genuinely_300m_class():
    model = ArkansasPharmaMultimodalModel()
    inventory = model_inventory(model)
    assert inventory["active_parameters"] == 307_798_423
    assert inventory["active_parameters"] == inventory["trainable_parameters"]
    assert inventory["layer_count"] == 6
    assert inventory["layer_parameter_counts"] == {
        "news": 242_698_240,
        "graph": 47_874_400,
        "temporal": 7_519_344,
        "cross_modal": 4_158_672,
        "deep_connections": 3_749_872,
        "heads": 1_797_895,
    }
    assert sum(inventory["layer_parameter_counts"].values()) == inventory["active_parameters"]
    assert set(inventory["layer_contracts"]) == set(inventory["layer_names"])


def test_modular_forward_exposes_intermediate_states():
    cfg = ModularModelConfig(
        vocab_size=128,
        max_tokens=16,
        text_width=96,
        text_layers=2,
        text_heads=8,
        text_ffn=192,
        entity_vocab=256,
        entity_width=64,
        graph_layers=1,
        temporal_layers=1,
        temporal_heads=8,
        temporal_ffn=128,
        history_features=11,
    )
    model = ArkansasPharmaMultimodalModel(cfg).eval()
    with torch.no_grad():
        out = model(
            torch.randint(0, cfg.vocab_size, (2, 12)),
            torch.randint(0, cfg.entity_vocab, (2, 6)),
            torch.randint(0, 32, (2, 6, 6)),
            torch.randn(2, 4, 11),
        )
    assert out["point"].shape == (2, cfg.horizon_count, cfg.target_count)
    assert out["quantiles"].shape == (2, cfg.horizon_count, cfg.target_count, 3)
    assert out["state_risk"].shape == (2, cfg.horizon_count, 3)
    assert out["target_state_risk"].shape == (
        2, cfg.horizon_count, cfg.target_count, cfg.state_count)
    assert out["driver_contribution"].shape == (2, 10)
    assert out["intermediate_news"].shape == (2, cfg.text_width)
    assert out["intermediate_graph"].shape == (2, cfg.entity_width)
    assert out["intermediate_temporal"].shape == (2, cfg.entity_width)
    assert out["intermediate_metric_context"].shape == (2, cfg.entity_width)
    assert out["intermediate_cross_modal"].shape == (2, 3, cfg.entity_width)
    assert out["intermediate_deep_connections"].shape == (2, 3, cfg.entity_width)


def test_metric_context_is_optional_but_shape_checked_and_reaches_temporal_state():
    cfg = ModularModelConfig(
        vocab_size=64, max_tokens=8, text_width=32, text_layers=1,
        text_heads=4, text_ffn=64, entity_vocab=128, entity_width=32,
        graph_layers=1, temporal_layers=1, temporal_heads=4,
        temporal_ffn=64, history_features=11, metric_feature_count=14,
        horizon_count=1, target_count=1,
    )
    model = ArkansasPharmaMultimodalModel(cfg).eval()
    inputs = (
        torch.randint(0, cfg.vocab_size, (2, 4)),
        torch.randint(0, cfg.entity_vocab, (2, 3)),
        torch.randint(0, 32, (2, 3, 3)),
        torch.randn(2, 4, 11),
    )
    with torch.no_grad():
        empty = model(*inputs)
        supplied = model(*inputs, metric_features=torch.ones(2, 14))
    assert torch.allclose(empty["intermediate_metric_context"], torch.zeros(2, 32))
    assert not torch.allclose(
        supplied["intermediate_metric_context"], empty["intermediate_metric_context"])
    with pytest.raises(ValueError, match="width"):
        model(*inputs, metric_features=torch.zeros(2, 13))


def test_temporal_layer_handles_entities_without_history():
    cfg = ModularModelConfig(
        vocab_size=64, max_tokens=8, text_width=32, text_layers=1,
        text_heads=4, text_ffn=64, entity_vocab=128, entity_width=32,
        graph_layers=1, temporal_layers=1, temporal_heads=4,
        temporal_ffn=64, history_features=11, horizon_count=1, target_count=1,
    )
    model = ArkansasPharmaMultimodalModel(cfg).eval()
    with torch.no_grad():
        out = model(
            torch.zeros(2, 4, dtype=torch.long),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 3, 3, dtype=torch.long),
            torch.zeros(2, 4, 11),
            attention_mask=torch.ones(2, 4, dtype=torch.bool),
            history_padding_mask=torch.ones(2, 4, dtype=torch.bool),
        )
    assert torch.isfinite(out["intermediate_temporal"]).all()
    assert torch.isfinite(out["point"]).all()


def test_graph_layer_masks_padding_and_uses_typed_neighbor_messages():
    cfg = ModularModelConfig(
        vocab_size=32, max_tokens=4, text_width=32, text_layers=1,
        text_heads=4, text_ffn=64, entity_vocab=64, entity_width=32,
        graph_layers=1, temporal_layers=1, temporal_heads=4,
        temporal_ffn=64, history_features=11, horizon_count=1, target_count=1,
    )
    model = ArkansasPharmaMultimodalModel(cfg).eval()
    entities = torch.tensor([[1, 2, 0], [1, 2, 0]])
    relation_a = torch.zeros(2, 3, 3, dtype=torch.long)
    relation_b = relation_a.clone()
    relation_a[0, 0, 1] = 1
    relation_b[1, 0, 1] = 2
    with torch.no_grad():
        graph_a = model.graph(entities, relation_a)
        graph_b = model.graph(entities, relation_b)
        graph_padded = model.graph(
            torch.tensor([[1, 2, 0], [1, 2, 0]]),
            torch.stack([relation_a[0], relation_a[0]]),
        )
    assert torch.isfinite(graph_a).all()
    assert torch.isfinite(graph_padded).all()
    assert not torch.allclose(graph_a[0], graph_b[1])
    assert torch.allclose(graph_a[0], graph_padded[0], atol=1e-6)
