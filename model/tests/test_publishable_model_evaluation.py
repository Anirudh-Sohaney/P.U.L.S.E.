from pathlib import Path

import numpy as np
import pandas as pd

from arkansas_pharma_signal.publishable_model_evaluation import (
    _suite_context,
    _fit_stacked_ridge,
    _scale_residual_prediction,
    _apply_train_only_transition_predictions,
    _assign_public_state_labels,
    _normalize_public_histories,
    _fit_and_apply_deep_stack_blend,
    build_publishable_demand_rows,
    evaluate_publishable_rolling,
)
from arkansas_pharma_signal.training import research_small_model_config


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "data/targeted_additions/publishable_test_dataset"


def test_suite_context_has_model_width():
    row = pd.Series({"prior_ar_ili_mean": 1.0, "prior_news_fda_recall": 2.0})
    assert _suite_context(row).shape == (19,)


def test_public_adapter_preserves_strict_splits_and_width():
    rows, metadata = build_publishable_demand_rows(ROOT, SUITE,
                                                    research_small_model_config())
    assert metadata["feature_width"] == 38
    assert metadata["train_rows"] > 0
    assert metadata["validation_rows"] > 0
    assert metadata["test_rows"] > 0
    assert len(rows) == (metadata["train_rows"] + metadata["validation_rows"]
                         + metadata["test_rows"])
    assert len(rows[0]["raw_features"]) == len(metadata["raw_feature_names"])
    # The cache is built for the full panel, including rows that cannot form a
    # strict next-period target, so it may exceed the emitted sample count.
    assert metadata["graph_context_unique_count"] >= len({row["drug"] for row in rows})
    assert all(row["history"].shape == (4, 38) for row in rows[:10])
    assert all(row["entity_ids"].shape == (4,) for row in rows[:10])


def test_public_adapter_can_mask_ndc_identity():
    rows, metadata = build_publishable_demand_rows(
        ROOT, SUITE, research_small_model_config(), mask_drug_identity=True)
    assert metadata["mask_drug_identity"] is True
    assert all(row["entity_ids"][0] == 0 for row in rows[:20])


def test_public_state_bands_fit_training_rows_only_and_expose_five_states():
    def row(year, value, base=10.0):
        return {"year": year, "target": value, "base_log": np.log1p(base)}

    train = [row(2018, value) for value in (5.0, 8.0, 10.0, 14.0, 20.0)]
    validation = [row(2019, 1000.0)]
    test = [row(2020, 0.0)]
    thresholds = _assign_public_state_labels(train, validation, test)

    assert thresholds is not None
    assert len(thresholds) == 4
    assert {item["state_label"] for item in train} == {0, 1, 2, 3, 4}
    assert validation[0]["state_label"] == 4
    assert test[0]["state_label"] == 0
    assert all(item["state_thresholds"] == thresholds
               for item in (*train, *validation, *test))


def test_public_absolute_state_bands_use_next_period_level():
    def row(year, value, base=10.0):
        return {"year": year, "target": value, "base_log": np.log1p(base)}

    train = [row(2018, value) for value in (5.0, 8.0, 10.0, 14.0, 20.0)]
    validation = [row(2019, 1000.0)]
    test = [row(2020, 0.0)]
    thresholds = _assign_public_state_labels(
        train, validation, test, state_target_mode="level")

    assert thresholds is not None
    assert {item["state_label"] for item in train} == {0, 1, 2, 3, 4}
    assert validation[0]["state_label"] == 4
    assert test[0]["state_label"] == 0


def test_public_history_normalization_uses_train_fold_only():
    def item(value):
        return {"history": np.asarray([[[value]]], dtype=np.float32)}

    train = [item(1.0), item(3.0)]
    validation = [item(100.0)]
    test = [item(200.0)]
    mean, std = _normalize_public_histories(train, validation, test)
    assert np.isclose(mean[0], 2.0)
    assert np.isclose(std[0], 1.0)
    assert np.isclose(validation[0]["history"][0, 0, 0], 98.0)


def test_transition_predictions_use_only_prior_training_targets():
    def item(year, drug, target, current=10.0):
        return {
            "year": year, "quarter": 1, "drug": drug, "target": target,
            "base_log": np.log1p(current), "raw_features": np.zeros(2),
        }

    train = [item(2017, "a", 20.0), item(2018, "a", 30.0)]
    validation = [item(2019, "a", 999.0)]
    test = [item(2020, "a", 999.0)]
    _apply_train_only_transition_predictions(train, validation, test)
    first_validation = validation[0]["transition_prediction"]
    validation[0]["target"] = 1.0
    test[0]["target"] = 1.0
    _apply_train_only_transition_predictions(train, validation, test)
    assert np.isclose(first_validation, validation[0]["transition_prediction"])
    assert np.isclose(validation[0]["raw_features"][-1], first_validation)


def test_stacked_ridge_uses_latent_and_raw_features_without_nan():
    def rows(values):
        return [{"target": float(value), "raw_features": [float(value), 1.0]}
                for value in values]

    result = _fit_stacked_ridge(
        rows([1, 2, 3, 4]), rows([5, 6]), rows([7, 8]),
        train_latent=np.asarray([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]]),
        validation_latent=np.asarray([[0.4, 1.4], [0.5, 1.5]]),
        test_latent=np.asarray([[0.6, 1.6], [0.7, 1.7]]),
        feature_names=["raw_a", "raw_b", "latent_a", "latent_b"],
    )
    assert result["feature_count"] == 4
    assert np.isfinite(result["test"]["wape"])

    prepared = tuple(np.asarray([
        [float(value), 1.0, latent]
        for value, latent in zip(values, latent_values)
    ]) for values, latent_values in (
        ([1, 2, 3, 4], [0.0, 0.1, 0.2, 0.3]),
        ([5, 6], [0.4, 0.5]),
        ([7, 8], [0.6, 0.7]),
    ))
    cached = _fit_stacked_ridge(
        rows([1, 2, 3, 4]), rows([5, 6]), rows([7, 8]),
        train_latent=np.asarray([[0.0], [0.1], [0.2], [0.3]]),
        validation_latent=np.asarray([[0.4], [0.5]]),
        test_latent=np.asarray([[0.6], [0.7]]),
        feature_names=["raw_a", "raw_b", "latent_a"],
        prepared_matrices=prepared,
    )
    assert np.isclose(cached["test"]["wape"], result["test"]["wape"])

    restricted = _fit_stacked_ridge(
        rows([1, 2, 3, 4]), rows([5, 6]), rows([7, 8]),
        train_latent=np.asarray([[0.0], [0.1], [0.2], [0.3]]),
        validation_latent=np.asarray([[0.4], [0.5]]),
        test_latent=np.asarray([[0.6], [0.7]]),
        feature_names=["raw_a", "raw_b", "latent_a"],
        alpha_values=(1.0,),
    )
    assert restricted["alpha_grid"] == [1.0]


def test_residual_stacked_ridge_reconstructs_from_persistence_anchor():
    def rows(values):
        return [{"target": float(value), "base_log": np.log1p(value - 1.0),
                 "raw_features": [float(value), 1.0]}
                for value in values]

    result = _fit_stacked_ridge(
        rows([2, 3, 4, 5]), rows([6, 7]), rows([8, 9]),
        train_latent=np.zeros((4, 1)), validation_latent=np.zeros((2, 1)),
        test_latent=np.zeros((2, 1)),
        feature_names=["raw_a", "raw_b", "latent_a"],
        residual_target=True,
    )
    assert result["target_mode"] == "persistence_residual_log1p"
    assert np.isfinite(result["test"]["wape"])


def test_residual_stacked_ridge_cap_uses_absolute_training_targets():
    def rows(values):
        return [{"target": float(value), "base_log": np.log1p(value),
                 "raw_features": [1.0, 0.0]} for value in values]

    result = _fit_stacked_ridge(
        rows([100, 200]), rows([150]), rows([175]),
        train_latent=np.zeros((2, 1)), validation_latent=np.zeros((1, 1)),
        test_latent=np.zeros((1, 1)),
        feature_names=["raw_a", "raw_b", "latent_a"],
        residual_target=True, prediction_cap_quantile=1.0,
    )
    assert result["prediction_cap"] == 200.0


def test_residual_scale_zero_is_persistence_anchor():
    rows = [{"base_log": np.log1p(10.0)}, {"base_log": np.log1p(20.0)}]
    prediction = _scale_residual_prediction(
        rows, np.asarray([2.0, -2.0]), 0.0, 100.0)
    assert np.allclose(prediction, [10.0, 20.0])


def test_deep_stack_blend_reports_the_selected_convex_combination():
    result = _fit_and_apply_deep_stack_blend(
        np.asarray([10.0, 20.0]),
        np.asarray([10.0, 20.0]),
        np.asarray([20.0, 10.0]),
        np.asarray([30.0, 40.0]),
        np.asarray([40.0, 30.0]),
    )
    assert 0.0 <= result["weight"] <= 1.0
    assert np.allclose(
        result["test_prediction"],
        ((1.0 - result["weight"]) * np.asarray([30.0, 40.0])
         + result["weight"] * np.asarray([40.0, 30.0])),
    )


def test_rolling_gate_compares_stacked_to_fold_ridge(monkeypatch, tmp_path):
    import arkansas_pharma_signal.publishable_model_evaluation as module

    def fake_train(*args, train_last_year, validation_year, **kwargs):
        return {
            "split": {"train_rows": 10, "validation_rows": 5, "test_rows": 3},
            "best_validation_wape": 0.5,
            "test": {
                "neural": {"wape": 0.9, "within_5pct": 0.4,
                           "within_tolerance": 0.4},
                "stacked_ridge": {"wape": 0.8, "within_5pct": 0.5},
                "scaled_stacked_ridge": {"wape": 0.8, "within_5pct": 0.5},
                "stacked_blend": {"wape": 0.8, "within_5pct": 0.5,
                                  "within_tolerance": 0.5},
                "raw_ridge": {"wape": 1.0},
            },
            "scaled_stacked_ridge": {"residual_scale": 0.8},
            "stacked_blend": {"weight_on_deep_stack": 1.0},
        }

    monkeypatch.setattr(module, "train_publishable_demand_model", fake_train)
    result = evaluate_publishable_rolling(tmp_path, tmp_path, epochs=1)
    assert result["fold_count"] == 3
    assert result["publishable_rolling_candidate"] is True
    assert result["contract_75pct_numeric_accuracy"] is False
    assert all(np.isclose(row["stacked_improvement_vs_fold_ridge"], 0.2)
               for row in result["folds"])


def test_rolling_validation_window_is_predeclared_and_chronological(monkeypatch, tmp_path):
    import arkansas_pharma_signal.publishable_model_evaluation as module
    calls = []

    def fake_train(*args, train_last_year, validation_year,
                   validation_end_year, **kwargs):
        calls.append((train_last_year, validation_year, validation_end_year))
        return {
            "split": {"train_rows": 10, "validation_rows": 5, "test_rows": 3},
            "best_validation_wape": 0.5,
            "test": {
                "neural": {"wape": 0.9, "within_5pct": 0.4,
                           "within_tolerance": 0.4},
                "stacked_ridge": {"wape": 0.8, "within_5pct": 0.5},
                "scaled_stacked_ridge": {"wape": 0.8, "within_5pct": 0.5},
                "stacked_blend": {"wape": 0.8, "within_5pct": 0.5,
                                  "within_tolerance": 0.5},
                "raw_ridge": {"wape": 1.0},
            },
            "scaled_stacked_ridge": {"residual_scale": 0.8},
            "stacked_blend": {"weight_on_deep_stack": 1.0},
        }

    monkeypatch.setattr(module, "train_publishable_demand_model", fake_train)
    result = module.evaluate_publishable_rolling(
        tmp_path, tmp_path, epochs=1, validation_window_years=2)
    assert calls == [(2017, 2018, 2019), (2018, 2019, 2020), (2019, 2020, 2021)]
    assert all(row["validation_window_years"] == 2 for row in result["folds"])
