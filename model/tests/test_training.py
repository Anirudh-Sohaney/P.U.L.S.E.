from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import arkansas_pharma_signal.training as training_module
from arkansas_pharma_signal.input_contract import (
    is_local_history_feature, is_operational_feature,
)
from arkansas_pharma_signal.modular_model import ModularModelConfig
from arkansas_pharma_signal.training import (
    TrainConfig, _graph_context, _labeler_entity_for_ndc, _loss,
    build_real_training_arrays,
    evaluate_modular_rolling, modular_input_contract, modular_rolling_cutoffs,
    require_operational_modular,
    select_transition_blend_weight, state_metrics, _state_from_log,
)


def test_modular_input_contract_marks_periodic_context_as_research_only():
    contract = modular_input_contract()
    assert contract["operational_ready"] is False
    assert contract["periodic_training_only_count"] >= 2
    assert is_operational_feature("value")
    assert is_operational_feature("value_last")
    assert is_operational_feature("quarter")
    assert is_local_history_feature("value")
    assert contract["local_history_count"] == 10
    assert contract["non_operational_feature_count"] == 2
    assert not is_operational_feature("global_gscpi_mean")
    with pytest.raises(RuntimeError, match="periodic or historical-only"):
        require_operational_modular(contract)
    with pytest.raises(RuntimeError, match="periodic or historical-only"):
        training_module.train_real_model(
            Path(__file__).resolve().parents[2],
            TrainConfig(operational_only=True),
            model_cfg=ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1),
        )


def test_real_training_arrays_have_strict_temporal_partitions():
    root = Path(__file__).resolve().parents[2]
    cfg = TrainConfig(max_tokens=8)
    model_cfg = ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1)
    train, validation, test, _, _, _ = build_real_training_arrays(root, cfg, model_cfg)
    assert train and validation and test
    assert max(x["year"] for x in train) <= cfg.train_last_year
    assert {x["year"] for x in validation} == {cfg.validation_year}
    assert min(x["year"] for x in test) > cfg.validation_year
    for row in train + validation + test:
        assert np.isfinite(row["target"])
        assert row["target"] >= 0
        assert row["history"].shape == (4, 38)
        assert row["text_mask"].shape == (8,)
        assert row["state_label"] in (-1, 0, 1, 2, 3, 4)
        assert row["metric_features"].shape == (16,)
        assert np.isfinite(row["state_value"])
        if row["state_thresholds"] is not None:
            assert len(row["state_thresholds"]) == 4


def test_change_state_is_relative_to_observed_persistence_level():
    root = Path(__file__).resolve().parents[2]
    cfg = TrainConfig(max_tokens=8, state_target_mode="change")
    model_cfg = ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1)
    train, _, _, _, _, _ = build_real_training_arrays(root, cfg, model_cfg)
    row = next(x for x in train if x["persistence_log"] != 0.0)
    expected = np.log1p(row["target"]) - row["persistence_log"]
    assert row["state_value"] == pytest.approx(expected)


def test_modular_rolling_cutoffs_are_chronological_and_have_future_test():
    folds = modular_rolling_cutoffs(range(2012, 2025), min_train_years=4)
    assert len(folds) >= 3
    assert all(validation == train_last + 1
               for train_last, validation in folds)
    assert folds == sorted(folds)


def test_transition_blend_weight_uses_validation_and_is_bounded():
    weight, score = select_transition_blend_weight(
        np.array([10.0, 20.0]), np.array([10.0, 20.0]), np.array([11.0, 19.0]))
    assert 0.0 <= weight <= 1.0
    assert score >= 0.0


def test_training_transition_pool_never_uses_later_year_ratios(tmp_path):
    rows = []
    for year in range(2019, 2023):
        for quarter in range(1, 5):
            # Drug A develops extreme later-year transitions. Drug B's early
            # row must not see those ratios through the quarter fallback pool.
            a_value = 100.0 if year == 2019 else 100.0 ** quarter
            for drug, value in (("a", a_value), ("b", 100.0)):
                rows.append({"year": year, "quarter": quarter,
                             "drug": drug, "value": value})
    panel_path = tmp_path / "model" / "artifacts" / "panel"
    panel_path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(panel_path / "medicaid_sdud_quarterly_panel.csv", index=False)
    cfg = TrainConfig(max_tokens=8)
    model_cfg = ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1)
    train, _, _, _, _, _ = build_real_training_arrays(tmp_path, cfg, model_cfg)
    early_b = next(row for row in train
                   if row["drug"] == "b" and row["year"] == 2019
                   and row["quarter"] == 2)
    # The only available prior-quarter ratios at this point are 1.0, so the
    # transition base must remain 100 rather than use A's later ratios.
    assert np.isclose(np.expm1(early_b["transition_log"]), 100.0)


def test_cold_start_ablation_masks_only_drug_identity():
    root = Path(__file__).resolve().parents[2]
    cfg = TrainConfig(max_tokens=8, mask_drug_identity=True)
    model_cfg = ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1)
    train, _, _, _, _, _ = build_real_training_arrays(root, cfg, model_cfg)
    assert all(row["entity_ids"][0] == 0 for row in train)
    assert any(row["entity_ids"][1] != 0 for row in train)


def test_content_entities_are_shared_and_do_not_claim_graph_edges():
    first, first_rel = _graph_context({}, "acetaminophen 500", 1000,
                                     use_content_entities=True)
    second, second_rel = _graph_context({}, "acetaminophen 250", 1000,
                                       use_content_entities=True)
    assert first.shape == (8,)
    assert first_rel.shape == (8, 8)
    assert first[2] == second[2]  # shared lexical token
    assert first_rel[0, 2] == 1  # declared contains relation only
    assert np.count_nonzero(first_rel[0, 4:]) == 0


def test_graph_evidence_mode_prioritizes_observed_supplier_edges():
    lookup = {"drug:x": [("markets", "labeler:late"),
                          ("manufactured_by", "labeler:direct"),
                          ("contains", "ingredient:x")]}
    ids, relations = _graph_context(lookup, "x", 1000,
                                    prioritize_evidence=True)
    assert ids.shape == (8,)
    assert relations.shape == (8, 8)
    # contains and manufactured_by are placed before lower-priority markets.
    assert relations[0, 2] == 1
    assert relations[0, 3] == 3


def test_ndc_labeler_entity_uses_only_catalog_backed_segment():
    lookup = {"68180": "labeler:lupin_pharmaceuticals"}
    assert _labeler_entity_for_ndc("68180028607", lookup) == \
        "labeler:lupin_pharmaceuticals"
    assert _labeler_entity_for_ndc("99999028607", lookup) == ""


def test_absolute_objective_removes_only_the_residual_anchor(tmp_path):
    rows = []
    for year in range(2018, 2023):
        for quarter in range(1, 5):
            rows.append({"year": year, "quarter": quarter,
                         "drug": "c", "value": 50.0 + year + quarter})
    panel_path = tmp_path / "model" / "artifacts" / "panel"
    panel_path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(panel_path / "medicaid_sdud_quarterly_panel.csv", index=False)
    cfg = TrainConfig(max_tokens=8, baseline_mode="absolute")
    model_cfg = ModularModelConfig(max_tokens=8, horizon_count=1, target_count=1)
    train, _, _, _, _, _ = build_real_training_arrays(tmp_path, cfg, model_cfg)
    assert train
    for row in train:
        assert row["base_log"] == 0.0
        # Baselines remain intact for scoring; only the anchor is removed.
        for row in train:
            expected_value = 50.0 + row["year"] + row["quarter"]
            assert np.isclose(np.expm1(row["persistence_log"]), expected_value)
            assert np.expm1(row["transition_log"]) >= 0.0
    with pytest.raises(ValueError, match="unsupported baseline_mode"):
        build_real_training_arrays(tmp_path, TrainConfig(baseline_mode="unknown"),
                                   model_cfg)


def test_wape_weighted_loss_is_finite_and_requires_labels():
    out = {
        "point": torch.zeros(3, 1, 1),
        "quantiles": torch.zeros(3, 1, 1, 3),
    }
    target_delta = torch.tensor([0.0, 0.5, -0.5])
    target_log = torch.tensor([0.0, 2.0, 4.0])
    loss = _loss(out, target_delta, target_log=target_log, mode="wape_weighted")
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="target_log"):
        _loss(out, target_delta, mode="wape_weighted")
    with pytest.raises(ValueError, match="unsupported"):
        _loss(out, target_delta, mode="unknown")


def test_state_loss_uses_target_specific_three_state_logits():
    out = {
        "point": torch.zeros(3, 1, 1),
        "quantiles": torch.zeros(3, 1, 1, 3),
        "target_state_risk": torch.zeros(3, 1, 1, 3, requires_grad=True),
    }
    labels = torch.tensor([0, 1, 2])
    loss = _loss(out, torch.zeros(3), state_label=labels,
                 state_loss_weight=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert out["target_state_risk"].grad is not None
    with pytest.raises(ValueError, match="state_label"):
        _loss(out, torch.zeros(3), state_loss_weight=0.5)


def test_ordinal_state_loss_is_finite_and_validates_mode():
    out = {
        "point": torch.zeros(3, 1, 1),
        "quantiles": torch.zeros(3, 1, 1, 3),
        "target_state_risk": torch.zeros(3, 1, 1, 5, requires_grad=True),
    }
    labels = torch.tensor([0, 2, 4])
    loss = _loss(out, torch.zeros(3), state_label=labels,
                 state_loss_weight=0.5, state_loss_mode="ordinal")
    assert torch.isfinite(loss)
    loss.backward()
    assert out["target_state_risk"].grad is not None
    with pytest.raises(ValueError, match="categorical or ordinal"):
        _loss(out, torch.zeros(3), state_label=labels,
              state_loss_weight=0.5, state_loss_mode="invalid")


def test_state_metrics_requires_five_state_contract():
    result = state_metrics(np.array([0, 1, 2, 3, 4]), np.array([0, 2, 2, 4, 1]))
    assert result["rows"] == 5
    assert result["exact_accuracy"] == pytest.approx(0.4)
    assert set(result["state_counts"]) == {"0", "1", "2", "3", "4"}
    with pytest.raises(ValueError, match="0, 1, 2, 3, or 4"):
        state_metrics(np.array([0, 5]), np.array([0, 1]))
    assert _state_from_log(0.0, None) == -1
    thresholds = (1.0, 2.0, 3.0, 4.0)
    assert _state_from_log(1.0, thresholds) == 1
    assert _state_from_log(1.5, thresholds) == 1
    assert _state_from_log(2.5, thresholds) == 2
    assert _state_from_log(4.5, thresholds) == 4


def test_modular_rolling_scores_validation_selected_policy(tmp_path, monkeypatch):
    panel_path = tmp_path / "model" / "artifacts" / "panel"
    panel_path.mkdir(parents=True)
    pd.DataFrame({"year": range(2012, 2023)}).to_csv(
        panel_path / "medicaid_sdud_quarterly_panel.csv", index=False)

    def fake_train(*args, **kwargs):
        return {
            "rows": {"test": 10},
            "prediction_metrics": {"test": {
                "neural": {"wape": 0.50},
                "blended": {"wape": 0.10},
                "persistence": {"wape": 0.20},
                "transition": {"wape": 0.15},
            }},
            "validation_selected_blend_weight": 0.3,
        }

    monkeypatch.setattr(training_module, "train_real_model", fake_train)
    result = evaluate_modular_rolling(
        tmp_path, train_cfg=TrainConfig(),
        model_cfg=ModularModelConfig(), min_train_years=4)
    assert result["fold_count"] == 6
    assert result["mean_improvement_vs_strongest_naive"] == pytest.approx(1 / 3)
    assert result["forecast_mode"] == "research_training_only"
    assert result["input_contract"]["non_operational_feature_count"] == 2
    assert all(row["selected_wape"] == 0.10 for row in result["folds"])
    assert all(row["selected_blend_weight"] == 0.3 for row in result["folds"])
