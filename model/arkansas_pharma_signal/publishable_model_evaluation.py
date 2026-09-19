"""End-to-end demand evaluation on the canonical public NDC9-quarter suite.

This module deliberately does not reuse the historical canonical-drug panel
loader. The public suite's primary grain is NDC9 x Arkansas feature quarter,
so the adapter preserves NDC9 identity, requires an observed consecutive next
quarter, and evaluates only the demand head that has a matching label.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from .modular_model import (ArkansasPharmaMultimodalModel, ModularModelConfig,
                            model_inventory)
from .event_accuracy import event_route_passes, score_event_predictions
from .publishable_evaluation import verify_suite
from .training import (_batches, _batch, _cache_document_states, _graph_context,
                       _loss, _quarter_texts, metrics_dict,
                       research_small_model_config,
                       select_transition_blend_weight, state_metrics,
                       _state_from_log)
from .regression import RidgeLinear


PUBLIC_RAW_BASE_FEATURES = [
    "medicaid_prescriptions", "feature_lag1_medicaid_prescriptions",
    "feature_lag2_medicaid_prescriptions", "feature_ma2_medicaid_prescriptions",
    "fda_shortage_active", "fda_shortage_supplier_count", "prior_context_missing",
]

STATE_COUNT = 5


def _assign_public_state_labels(train: Sequence[dict],
                                validation: Sequence[dict],
                                test: Sequence[dict],
                                *, state_target_mode: str = "change") -> tuple[float, ...] | None:
    """Fit five state cut points on training rows only.

    ``change`` describes the next-quarter log change from the observed current
    value; ``level`` describes the next-quarter absolute log demand level.
    Quantiles are fit only on the training split, then frozen for validation
    and test so state accuracy cannot use future label distributions.
    """
    if state_target_mode not in {"change", "level"}:
        raise ValueError("state_target_mode must be 'change' or 'level'")
    train_values = np.asarray([
        (np.log1p(max(float(row["target"]), 0.0)) - float(row["base_log"])
         if state_target_mode == "change"
         else np.log1p(max(float(row["target"]), 0.0)))
        for row in train], dtype=float)
    thresholds = np.quantile(train_values, np.arange(1, STATE_COUNT) / STATE_COUNT)
    fitted = tuple(float(value) for value in thresholds) \
        if np.all(np.diff(thresholds) > 0) else None
    for split in (train, validation, test):
        for row in split:
            value = (np.log1p(max(float(row["target"]), 0.0)) - float(row["base_log"])
                     if state_target_mode == "change"
                     else np.log1p(max(float(row["target"]), 0.0)))
            row["state_value"] = float(value)
            row["state_label"] = (_state_from_log(value, fitted)
                                   if fitted is not None else -1)
            row["state_thresholds"] = fitted
    return fitted


def _suite_context(row: pd.Series) -> np.ndarray:
    """Map strictly prior public context into the model's fixed 36 fields."""
    def value(name: str) -> float:
        raw = pd.to_numeric(row.get(name, 0.0), errors="coerce")
        return float(raw) if pd.notna(raw) and np.isfinite(raw) else 0.0

    news_shortage = sum(value(name) for name in (
        "prior_news_medication_access", "prior_news_pharmaceutical_supply_chain",
        "prior_news_manufacturing_disruption"))
    news_recall = value("prior_news_fda_recall")
    news_outbreak = value("prior_news_influenza") + value("prior_news_covid")
    mapped = {
        "ar_ili_mean": value("prior_ar_ili_mean"),
        "ar_wili_mean": value("prior_ar_wili_mean"),
        "nat_ili_mean": value("prior_nat_ili_mean"),
        "nat_wili_mean": value("prior_nat_wili_mean"),
        "news_shortage_count": news_shortage,
        "news_recall_count": news_recall,
        "news_disease_outbreak_count": news_outbreak,
        "news_disease_influenza_outbreak_risk": value("prior_news_influenza"),
    }
    columns = [
        "ar_disaster_active_mean", "ar_ili_mean", "ar_wili_mean",
        "ar_temperature_mean", "ar_precipitation_mean", "ar_wind_mean",
        "ar_unemployment_mean", "global_gscpi_mean", "us_tariff_rate_mean",
        "nat_ili_mean", "nat_wili_mean", "ww_covid_ar", "ww_flu_ar", "ww_rsv_ar",
        "news_shortage_count", "news_recall_count", "news_disease_outbreak_count",
        "news_disease_influenza_outbreak_risk", "news_relevance_mean",
    ]
    return np.asarray([mapped.get(column, 0.0) for column in columns], dtype=np.float32)


def build_publishable_demand_rows(root: Path, suite_root: Path,
                                  model_cfg: ModularModelConfig,
                                  *, max_tokens: int = 64,
                                  mask_drug_identity: bool = False) -> tuple[list[dict], dict]:
    """Build model rows from the raw public panel without target leakage."""
    verify_suite(suite_root)
    path = suite_root / "data" / "arkansas_ndc_quarter_panel.csv.gz"
    panel = pd.read_csv(path, dtype={"ndc9": str}, low_memory=False)
    panel["ndc9"] = panel["ndc9"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(9)
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel["quarter"] = pd.to_numeric(panel["quarter"], errors="raise").astype(int)
    panel["qid"] = panel["year"] * 4 + panel["quarter"]
    panel = panel.sort_values(["ndc9", "qid"]).reset_index(drop=True)
    group = panel.groupby("ndc9", sort=False)
    panel["previous_qid"] = group["qid"].shift(1)
    panel["next_qid"] = group["qid"].shift(-1)
    panel["value_last"] = group["medicaid_prescriptions"].shift(1).where(
        panel["qid"].sub(panel["previous_qid"]).eq(1))
    previous_two_qid = group["qid"].shift(2)
    panel["value_last2"] = group["medicaid_prescriptions"].shift(2).where(
        panel["qid"].sub(previous_two_qid).eq(2))
    panel["target"] = group["medicaid_prescriptions"].shift(-1).where(
        panel["next_qid"].sub(panel["qid"]).eq(1))

    text_map = _quarter_texts(root / "model/artifacts/news/historical_corpus.csv.gz",
                              model_cfg.vocab_size, max_tokens)
    prior_columns = sorted(c for c in panel.columns
                           if c.startswith("prior_") and c != "prior_context_missing")
    raw_feature_names = [*PUBLIC_RAW_BASE_FEATURES, *prior_columns,
                         "train_only_transition_prediction"]
    samples = []
    histories: Dict[str, list[np.ndarray]] = {}
    last_qid: Dict[str, int] = {}
    graph_context_cache: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in panel.itertuples(index=False):
        ndc9 = str(row.ndc9)
        qid = int(row.qid)
        if ndc9 in last_qid and qid - last_qid[ndc9] != 1:
            histories[ndc9] = []
        last_qid[ndc9] = qid
        current = max(float(row.medicaid_prescriptions), 0.0)
        prior = getattr(row, "value_last")
        prior = float(prior) if pd.notna(prior) else current
        prior2_raw = getattr(row, "value_last2")
        prior2 = float(prior2_raw) if pd.notna(prior2_raw) else prior
        ma2 = 0.5 * (current + prior)
        core = np.asarray([
            current, prior, ma2, np.log1p(current), np.log1p(prior),
            np.log1p(ma2), float(row.quarter == 1), float(row.quarter == 2),
            float(row.quarter == 3), float(row.quarter == 4), float(row.quarter),
            float(row.fda_shortage_active),
            float(row.fda_shortage_supplier_count),
            float(np.log1p(current) - np.log1p(prior)),
            float((current - prior) / (abs(prior) + 1.0)),
        ], dtype=np.float32)
        # The public demand task has no aligned historical NADAC series in its
        # manifest. Keep the four declared price slots explicit and neutral;
        # do not join a later or differently grained price observation.
        row_dict = row._asdict()
        feature = np.concatenate([
            core, _suite_context(pd.Series(row_dict)),
            np.zeros(4, dtype=np.float32),
        ], axis=0)
        missing_context = pd.to_numeric(
            row_dict.get("prior_context_missing", 0.0), errors="coerce")
        missing_context = (float(missing_context)
                           if pd.notna(missing_context) and np.isfinite(float(missing_context))
                           else 0.0)
        raw_feature_values = [
            current, prior, prior2, ma2,
            float(row.fda_shortage_active),
            float(row.fda_shortage_supplier_count),
            missing_context,
        ]
        for name in prior_columns:
            raw = pd.to_numeric(row_dict.get(name, 0.0), errors="coerce")
            raw_feature_values.append(float(raw) if pd.notna(raw) and np.isfinite(raw) else 0.0)
        raw_feature_values.append(0.0)
        history = np.zeros((4, len(feature)), dtype=np.float32)
        padding = np.ones(4, dtype=np.bool_)
        prior_history = histories.setdefault(ndc9, [])[-4:]
        if prior_history:
            history[-len(prior_history):] = np.stack(prior_history)
            padding[-len(prior_history):] = False
        tokens, mask = text_map.get((int(row.year), int(row.quarter)),
                                    (np.zeros(max_tokens, dtype=np.int64),
                                     np.zeros(max_tokens, dtype=np.bool_)))
        if ndc9 not in graph_context_cache:
            graph_context_cache[ndc9] = _graph_context(
                {}, f"ndc9:{ndc9}", model_cfg.entity_vocab, nodes=4)
        cached_entity_ids, cached_relation_ids = graph_context_cache[ndc9]
        # The identity ablation may mask the first node, so never hand a
        # mutable cached array directly to a sample.
        entity_ids = cached_entity_ids.copy()
        relation_ids = cached_relation_ids.copy()
        if mask_drug_identity:
            entity_ids[0] = 0
        if pd.notna(row.target):
            target = max(float(row.target), 0.0)
            samples.append({
                "year": int(row.year), "quarter": int(row.quarter), "drug": ndc9,
                "target": target, "history": history, "history_padding": padding,
                "base_log": float(np.log1p(current)),
                "persistence_log": float(np.log1p(current)),
                "transition_log": float(np.log1p(current)),
                "tokens": tokens, "text_mask": mask,
                "text_key": (int(row.year), int(row.quarter)),
                "entity_ids": entity_ids, "relation_ids": relation_ids,
                "graph_relation_types": [],
                "raw_features": np.asarray(raw_feature_values, dtype=np.float32),
            })
        histories.setdefault(ndc9, []).append(feature)

    if not samples:
        raise ValueError("public NDC9 panel produced no strict demand rows")
    train = [x for x in samples if x["year"] <= 2019]
    validation = [x for x in samples if x["year"] == 2020]
    test = [x for x in samples if x["year"] >= 2021]
    if not train or not validation or not test:
        raise ValueError("public suite split must contain train, validation, and test")
    return samples, {
        "train_rows": len(train), "validation_rows": len(validation),
        "test_rows": len(test), "feature_width": int(len(feature)),
        "raw_feature_names": raw_feature_names,
        "graph_context_unique_count": len(graph_context_cache),
        "mask_drug_identity": mask_drug_identity,
        "splits": {"train_feature_years": "<=2019",
                   "validation_feature_year": 2020,
                   "test_feature_years": ">=2021"},
    }


def _apply_train_only_transition_predictions(train: Sequence[dict],
                                              validation: Sequence[dict],
                                              test: Sequence[dict]) -> None:
    """Add expanding transition estimates without future target labels."""
    by_drug_quarter: dict[tuple[str, int], list[float]] = {}
    by_quarter: dict[int, list[float]] = {}
    global_ratios: list[float] = []

    def estimate(row: dict) -> float:
        current = max(float(np.expm1(row["base_log"])), 0.0)
        key = (str(row["drug"]), int(row["quarter"]))
        candidates = by_drug_quarter.get(key, [])
        if not candidates:
            candidates = by_quarter.get(int(row["quarter"]), [])
        if not candidates:
            candidates = global_ratios
        ratio = float(np.median(candidates)) if candidates else 1.0
        return max(current * max(ratio, 0.0), 0.0)

    def assign(rows: Sequence[dict], update: bool) -> None:
        ordered = sorted(rows, key=lambda row: (int(row["year"]),
                                                 int(row["quarter"]),
                                                 str(row["drug"])))
        for row in ordered:
            prediction = estimate(row)
            row["transition_prediction"] = prediction
            row["transition_log"] = float(np.log1p(prediction))
            row["raw_features"][-1] = prediction
            if update:
                current = max(float(np.expm1(row["base_log"])), 0.0)
                ratio = float(np.clip(row["target"] / max(current, 1e-9), 0.0, 10.0))
                by_drug_quarter.setdefault(
                    (str(row["drug"]), int(row["quarter"])), []).append(ratio)
                by_quarter.setdefault(int(row["quarter"]), []).append(ratio)
                global_ratios.append(ratio)

    assign(train, update=True)
    assign(validation, update=False)
    assign(test, update=False)


def _normalize_public_histories(train: Sequence[dict],
                                validation: Sequence[dict],
                                test: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fit temporal normalization on the selected training fold only."""
    train_history = np.concatenate([item["history"] for item in train], axis=0)
    mean = train_history.mean(axis=0)
    std = train_history.std(axis=0)
    std[std < 1e-6] = 1.0
    for split in (train, validation, test):
        for item in split:
            item["history"] = (item["history"] - mean) / std
    return mean, std


@torch.no_grad()
def _predict_rows(model: ArkansasPharmaMultimodalModel, rows: Sequence[dict],
                  device: torch.device, text_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    actual, prediction = [], []
    for batch_rows in _batches(rows, 256, 0):
        batch = _batch(batch_rows, device)
        text = torch.stack([text_cache[x["text_key"]] for x in batch_rows]).to(device)
        output = model.forward_from_states(text, batch["entity_ids"], batch["relation_ids"],
                                            batch["history"], batch["history_padding"])
        prediction.extend(np.expm1(batch["base_log"].cpu().numpy()
                                    + output["point"][:, 0, 0].cpu().numpy()))
        actual.extend(batch["target_raw"])
    return np.asarray(actual), np.maximum(np.asarray(prediction), 0.0)


@torch.no_grad()
def _predict_rows_with_latent(model: ArkansasPharmaMultimodalModel,
                              rows: Sequence[dict], device: torch.device,
                              text_cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return demand predictions and post-connection states for stacking."""
    model.eval()
    actual, prediction, latent = [], [], []
    for batch_rows in _batches(rows, 256, 0):
        batch = _batch(batch_rows, device)
        text = torch.stack([text_cache[x["text_key"]] for x in batch_rows]).to(device)
        output = model.forward_from_states(text, batch["entity_ids"], batch["relation_ids"],
                                            batch["history"], batch["history_padding"])
        prediction.extend(np.expm1(batch["base_log"].cpu().numpy()
                                    + output["point"][:, 0, 0].cpu().numpy()))
        actual.extend(batch["target_raw"])
        latent.append(output["intermediate_deep_connections"].flatten(start_dim=1).cpu().numpy())
    return (np.asarray(actual), np.maximum(np.asarray(prediction), 0.0),
            np.concatenate(latent, axis=0))


@torch.no_grad()
def _predict_public_states(model: ArkansasPharmaMultimodalModel,
                           rows: Sequence[dict], device: torch.device,
                           text_cache: dict, *,
                           state_target_mode: str = "change") -> dict:
    """Score the learned state head and numeric-head state projection.

    The second score maps the numeric residual forecast through the exact same
    training-only cut points. Comparing both paths distinguishes a weak state
    head from a weak shared representation without changing the deployment
    target or using held-out labels for fitting.
    """
    if state_target_mode not in {"change", "level"}:
        raise ValueError("state_target_mode must be 'change' or 'level'")
    actual, predicted, numeric_predicted = [], [], []
    thresholds = rows[0].get("state_thresholds") if rows else None
    model.eval()
    for batch_rows in _batches(rows, 256, 0):
        batch = _batch(batch_rows, device)
        text = torch.stack([text_cache[x["text_key"]] for x in batch_rows]).to(device)
        output = model.forward_from_states(
            text, batch["entity_ids"], batch["relation_ids"],
            batch["history"], batch["history_padding"])
        labels = batch["state_label"].cpu().numpy()
        guesses = output["target_state_risk"][:, 0, 0, :].argmax(dim=-1).cpu().numpy()
        numeric_point = output["point"][:, 0, 0].cpu().numpy()
        if state_target_mode == "level":
            numeric_point = (batch["base_log"].cpu().numpy() + numeric_point)
        numeric_guesses = np.asarray([
            _state_from_log(float(value), thresholds) for value in numeric_point],
            dtype=int)
        valid = labels >= 0
        actual.extend(labels[valid].tolist())
        predicted.extend(guesses[valid].tolist())
        numeric_predicted.extend(numeric_guesses[valid].tolist())

    def score(predictions: list[int]) -> dict:
        actual_array = np.asarray(actual, dtype=int)
        predicted_array = np.asarray(predictions, dtype=int)
        result = state_metrics(actual_array, predicted_array)
        result["event_metrics"] = score_event_predictions(
            actual_array, predicted_array, kind="state", event_states={3, 4})
        return result

    result = score(predicted)
    result["numeric_projection"] = score(numeric_predicted)
    return result


def _fit_stacked_ridge(train_rows: Sequence[dict], validation_rows: Sequence[dict],
                       test_rows: Sequence[dict], train_latent: np.ndarray,
                       validation_latent: np.ndarray, test_latent: np.ndarray,
                       feature_names: list[str],
                       prediction_cap_quantile: float | None = None,
                       return_predictions: bool = False,
                       residual_target: bool = False,
                       robust_huber_multiplier: float | None = None,
                       recency_gamma: float = 1.0,
                       prepared_matrices: tuple[np.ndarray, np.ndarray, np.ndarray]
                       | None = None,
                       alpha_values: Sequence[float] | None = None) -> dict:
    """Fit a validation-selected ridge on raw public inputs plus learned states.

    The neural path is trained on the log persistence residual.  The deep
    stack can therefore use the same target geometry when requested, while the
    raw ridge remains an absolute-level benchmark.
    """
    def matrix(rows: Sequence[dict], latent: np.ndarray) -> np.ndarray:
        raw = np.stack([x["raw_features"] for x in rows]).astype(float)
        return np.concatenate([raw, latent], axis=1)

    if prepared_matrices is None:
        x_train = matrix(train_rows, train_latent)
        x_validation = matrix(validation_rows, validation_latent)
        x_test = matrix(test_rows, test_latent)
    else:
        x_train, x_validation, x_test = prepared_matrices
    train_target_log = np.log1p(np.asarray(
        [x["target"] for x in train_rows], dtype=float))
    if residual_target:
        train_base_log = np.asarray(
            [x["base_log"] for x in train_rows], dtype=float)
    else:
        train_base_log = np.zeros(len(train_rows), dtype=float)
    y_train = (train_target_log - train_base_log
               if residual_target else train_target_log)
    y_validation = np.asarray([x["target"] for x in validation_rows], dtype=float)
    y_test = np.asarray([x["target"] for x in test_rows], dtype=float)
    if not 0.0 < float(recency_gamma) <= 1.0:
        raise ValueError("recency_gamma must be in (0, 1]")
    if robust_huber_multiplier is not None and robust_huber_multiplier <= 0:
        raise ValueError("robust_huber_multiplier must be positive")
    if residual_target:
        validation_base_log = np.asarray(
            [x["base_log"] for x in validation_rows], dtype=float)
        test_base_log = np.asarray(
            [x["base_log"] for x in test_rows], dtype=float)
    else:
        validation_base_log = np.zeros(len(validation_rows), dtype=float)
        test_base_log = np.zeros(len(test_rows), dtype=float)
    # The cap is an absolute-demand bound even when the ridge target is a
    # persistence residual. Using expm1(y_train) in residual mode caps the
    # reconstructed forecast at a residual-scale value and can collapse every
    # stacked prediction toward zero.
    train_target_values = np.expm1(train_target_log)
    if prediction_cap_quantile is None:
        prediction_cap = float(np.max(train_target_values))
    else:
        if not 0.0 < prediction_cap_quantile <= 1.0:
            raise ValueError("prediction_cap_quantile must be in (0, 1]")
        prediction_cap = float(np.quantile(train_target_values, prediction_cap_quantile))
    max_log_target = float(np.log1p(prediction_cap))
    train_years = np.asarray([x.get("year", 0) for x in train_rows], dtype=float)
    recency_weights = np.power(
        float(recency_gamma), np.max(train_years) - train_years)
    candidates = {}
    fit_alphas = tuple(alpha_values or (0.1, 1.0, 10.0, 100.0, 1000.0))
    for alpha in fit_alphas:
        model = RidgeLinear(alpha=alpha).fit(
            x_train, y_train, feature_names, sample_weight=recency_weights)
        if robust_huber_multiplier is not None:
            for _ in range(3):
                residual = y_train - model.predict(x_train)
                centered = residual - np.median(residual)
                mad_scale = 1.4826 * float(np.median(np.abs(centered)))
                cutoff = float(robust_huber_multiplier) * max(mad_scale, 1e-6)
                huber_weights = np.minimum(
                    1.0, cutoff / np.maximum(np.abs(residual), 1e-6))
                model.fit(
                    x_train, y_train, feature_names,
                    sample_weight=recency_weights * huber_weights)
        validation_raw_prediction = model.predict(x_validation)
        test_raw_prediction = model.predict(x_test)
        if residual_target:
            validation_log_prediction = np.clip(
                validation_base_log + validation_raw_prediction,
                -validation_base_log, max_log_target)
            test_log_prediction = np.clip(
                test_base_log + test_raw_prediction,
                -test_base_log, max_log_target)
        else:
            validation_log_prediction = np.clip(
                validation_raw_prediction, 0.0, max_log_target)
            test_log_prediction = np.clip(
                test_raw_prediction, 0.0, max_log_target)
        validation_prediction = np.maximum(
            np.expm1(validation_log_prediction), 0.0)
        test_prediction = np.maximum(np.expm1(test_log_prediction), 0.0)
        candidates[alpha] = (validation_prediction, test_prediction,
                             validation_raw_prediction, test_raw_prediction)
    selected_alpha = min(candidates, key=lambda alpha: metrics_dict(
        y_validation, candidates[alpha][0])["wape"])
    validation_prediction, test_prediction, validation_raw_prediction, test_raw_prediction = (
        candidates[selected_alpha])
    result = {
        "selected_alpha": selected_alpha,
        "alpha_grid": [float(alpha) for alpha in fit_alphas],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "prediction_cap": prediction_cap,
        "prediction_cap_quantile": prediction_cap_quantile,
        "target_mode": ("persistence_residual_log1p" if residual_target
                         else "absolute_log1p"),
        "robust_huber_multiplier": robust_huber_multiplier,
        "recency_gamma": float(recency_gamma),
        "validation": metrics_dict(y_validation, validation_prediction),
        "test": metrics_dict(y_test, test_prediction),
    }
    if return_predictions:
        result["_validation_prediction"] = validation_prediction
        result["_test_prediction"] = test_prediction
        if residual_target:
            result["_validation_residual_prediction"] = validation_raw_prediction
            result["_test_residual_prediction"] = test_raw_prediction
    return result


def _fit_and_apply_deep_stack_blend(
    validation_actual: np.ndarray,
    validation_raw: np.ndarray,
    validation_deep: np.ndarray,
    test_raw: np.ndarray,
    test_deep: np.ndarray,
) -> dict:
    """Select and apply the raw-ridge/deep-stack blend without policy substitution."""
    weight, validation_wape = select_transition_blend_weight(
        validation_actual, validation_raw, validation_deep)
    validation_prediction = ((1.0 - weight) * validation_raw
                             + weight * validation_deep)
    test_prediction = ((1.0 - weight) * test_raw + weight * test_deep)
    return {
        "weight": float(weight),
        "validation_wape": float(validation_wape),
        "validation_prediction": validation_prediction,
        "test_prediction": test_prediction,
    }


def _scale_residual_prediction(rows: Sequence[dict], residual_prediction: np.ndarray,
                               scale: float, prediction_cap: float) -> np.ndarray:
    """Reconstruct a persistence-anchored prediction at a validation scale."""
    if not 0.0 <= float(scale) <= 1.0:
        raise ValueError("residual scale must be between zero and one")
    base_log = np.asarray([x["base_log"] for x in rows], dtype=float)
    residual = np.asarray(residual_prediction, dtype=float) * float(scale)
    max_log_target = float(np.log1p(max(prediction_cap, 0.0)))
    log_prediction = np.clip(
        base_log + residual, -base_log, max_log_target)
    return np.maximum(np.expm1(log_prediction), 0.0)


def train_publishable_demand_model(root: Path, suite_root: Path, *,
                                   epochs: int = 1, batch_size: int = 512,
                                   seed: int = 17, device: str = "cpu",
                                   train_last_year: int = 2019,
                                   validation_year: int = 2020,
                                   validation_end_year: int | None = None,
                                   prediction_cap_quantile: float | None = None,
                                   mask_drug_identity: bool = False,
                                   state_loss_weight: float = 0.1,
                                   state_loss_mode: str = "categorical",
                                   state_target_mode: str = "change") -> dict:
    """Train/evaluate a small research model on the public NDC9 demand task."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = research_small_model_config()
    rows, split_meta = build_publishable_demand_rows(
        root, suite_root, cfg, max_tokens=cfg.max_tokens,
        mask_drug_identity=mask_drug_identity)
    train = [x for x in rows if x["year"] <= train_last_year]
    validation_end_year = (validation_year if validation_end_year is None
                           else int(validation_end_year))
    if validation_end_year < validation_year:
        raise ValueError("validation_end_year must be >= validation_year")
    validation = [x for x in rows
                  if validation_year <= x["year"] <= validation_end_year]
    test = [x for x in rows if x["year"] > validation_end_year]
    if not train or not validation or not test:
        raise ValueError("rolling split must contain train, validation, and test rows")
    state_thresholds = _assign_public_state_labels(
        train, validation, test, state_target_mode=state_target_mode)
    split_meta["splits"] = {
        "train_feature_years": f"<={train_last_year}",
        "validation_feature_years": (
            str(validation_year) if validation_end_year == validation_year
            else f"{validation_year}-{validation_end_year}"),
        "validation_window_years": validation_end_year - validation_year + 1,
        "test_feature_years": f">{validation_end_year}",
    }
    split_meta["train_rows"] = len(train)
    split_meta["validation_rows"] = len(validation)
    split_meta["test_rows"] = len(test)
    _apply_train_only_transition_predictions(train, validation, test)
    split_meta["transition_features"] = "expanding_train_only"
    _, history_std = _normalize_public_histories(train, validation, test)
    split_meta["history_normalization"] = "selected_train_fold_only"
    split_meta["history_scale_min"] = float(np.min(history_std))
    split_meta["state_target_mode"] = state_target_mode
    split_meta["state_loss_weight"] = float(state_loss_weight)
    split_meta["state_loss_mode"] = state_loss_mode
    split_meta["state_thresholds"] = (list(state_thresholds)
                                       if state_thresholds is not None else None)
    target_device = torch.device(device)
    model = ArkansasPharmaMultimodalModel(cfg).to(target_device)
    for parameter in model.news.parameters():
        parameter.requires_grad = False
    text_cache = _cache_document_states(model, rows, target_device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=2e-5, weight_decay=1e-4)
    history = []
    best_validation_wape = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch_rows in _batches(train, batch_size, seed + epoch):
            batch = _batch(batch_rows, target_device)
            text = torch.stack([text_cache[x["text_key"]] for x in batch_rows]).to(target_device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_from_states(text, batch["entity_ids"], batch["relation_ids"],
                                                batch["history"], batch["history_padding"])
            loss = _loss(output, batch["target_log"] - batch["base_log"],
                         target_log=batch["target_log"],
                         state_label=batch["state_label"],
                         state_loss_weight=state_loss_weight,
                         state_loss_mode=state_loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch_rows)
        actual, pred = _predict_rows(model, validation, target_device, text_cache)
        validation_metrics = metrics_dict(actual, pred)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train),
                        "validation": validation_metrics})
        if validation_metrics["wape"] < best_validation_wape:
            best_validation_wape = float(validation_metrics["wape"])
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("public model training produced no validation checkpoint")
    model.load_state_dict(best_state)
    validation_actual, validation_pred, validation_latent = _predict_rows_with_latent(
        model, validation, target_device, text_cache)
    test_actual, test_pred, test_latent = _predict_rows_with_latent(
        model, test, target_device, text_cache)
    _, _, train_latent = _predict_rows_with_latent(model, train, target_device, text_cache)
    deep_stack_matrices = tuple(np.concatenate([
        np.stack([row["raw_features"] for row in split]).astype(float), latent
    ], axis=1) for split, latent in zip(
        (train, validation, test), (train_latent, validation_latent, test_latent)))
    raw_stack_matrices = tuple(
        np.stack([row["raw_features"] for row in split]).astype(float)
        for split in (train, validation, test))
    transition_validation = np.expm1(np.asarray([x["transition_log"] for x in validation]))
    transition_test = np.expm1(np.asarray([x["transition_log"] for x in test]))
    weight, _ = select_transition_blend_weight(validation_actual, transition_validation,
                                               validation_pred)
    blended_test = (1.0 - weight) * transition_test + weight * test_pred
    latent_width = int(train_latent.shape[1])
    stack_feature_names = [*split_meta["raw_feature_names"],
                           *[f"deep_connection_{i}" for i in range(latent_width)]]
    stacked = _fit_stacked_ridge(
        train, validation, test, train_latent, validation_latent, test_latent,
        stack_feature_names,
        prediction_cap_quantile=prediction_cap_quantile,
        return_predictions=True,
        residual_target=True,
        prepared_matrices=deep_stack_matrices)
    residual_scales = np.linspace(0.0, 1.0, 6)
    def scaled_candidate(stack: dict, model_family: str) -> dict:
        validation_predictions = {
            float(scale): _scale_residual_prediction(
                validation, stack["_validation_residual_prediction"], scale,
                stack["prediction_cap"])
            for scale in residual_scales
        }
        selected_scale = min(
            validation_predictions,
            key=lambda scale: metrics_dict(
                validation_actual, validation_predictions[scale])["wape"])
        validation_prediction = validation_predictions[selected_scale]
        test_prediction = _scale_residual_prediction(
            test, stack["_test_residual_prediction"], selected_scale,
            stack["prediction_cap"])
        return {
            "model_family": model_family,
            "residual_scale": float(selected_scale),
            "scale_grid": [float(scale) for scale in residual_scales],
            "robust_huber_multiplier": stack.get("robust_huber_multiplier"),
            "recency_gamma": stack.get("recency_gamma", 1.0),
            "selected_alpha": stack["selected_alpha"],
            "prediction_cap": stack["prediction_cap"],
            "prediction_cap_quantile": stack["prediction_cap_quantile"],
            "validation": metrics_dict(validation_actual, validation_prediction),
            "test": metrics_dict(test_actual, test_prediction),
            "_validation_prediction": validation_prediction,
            "_test_prediction": test_prediction,
        }

    ordinary_scaled_stacked = scaled_candidate(stacked, "residual_ridge")
    selected_scaled_stacked = ordinary_scaled_stacked
    for huber_multiplier in (0.7, 1.0, 1.5):
        for recency_gamma in (0.5, 0.7, 0.85, 1.0):
            robust_stack = _fit_stacked_ridge(
                train, validation, test, train_latent, validation_latent,
                test_latent, stack_feature_names,
                prediction_cap_quantile=prediction_cap_quantile,
                return_predictions=True, residual_target=True,
                robust_huber_multiplier=huber_multiplier,
                recency_gamma=recency_gamma,
                prepared_matrices=deep_stack_matrices,
                alpha_values=(stacked["selected_alpha"],))
            robust_scaled = scaled_candidate(
                robust_stack, "huber_recency_residual_ridge")
            if (robust_scaled["validation"]["wape"]
                    < selected_scaled_stacked["validation"]["wape"]):
                selected_scaled_stacked = robust_scaled
            for key in ("_validation_residual_prediction",
                        "_test_residual_prediction"):
                robust_stack.pop(key, None)
    scaled_stacked = selected_scaled_stacked
    empty_train = np.zeros((len(train), 0), dtype=float)
    empty_validation = np.zeros((len(validation), 0), dtype=float)
    empty_test = np.zeros((len(test), 0), dtype=float)
    raw_ridge = _fit_stacked_ridge(
        train, validation, test, empty_train, empty_validation, empty_test,
        split_meta["raw_feature_names"],
        prediction_cap_quantile=prediction_cap_quantile,
        return_predictions=True,
        prepared_matrices=raw_stack_matrices)
    validation_neural_blend = (1.0 - weight) * transition_validation + weight * validation_pred
    validation_policies = {
        "transition": transition_validation,
        "neural": validation_pred,
        "transition_neural_blend": validation_neural_blend,
        "raw_ridge": raw_ridge["_validation_prediction"],
        "deep_stack": scaled_stacked["_validation_prediction"],
    }
    selected_source = min(
        validation_policies,
        key=lambda name: metrics_dict(
            validation_actual, validation_policies[name])["wape"],
    )
    test_policies = {
        "transition": transition_test,
        "neural": test_pred,
        "transition_neural_blend": blended_test,
        "raw_ridge": raw_ridge["_test_prediction"],
        "deep_stack": scaled_stacked["_test_prediction"],
    }
    deep_stack_blend = _fit_and_apply_deep_stack_blend(
        validation_actual, raw_ridge["_validation_prediction"],
        scaled_stacked["_validation_prediction"],
        raw_ridge["_test_prediction"], scaled_stacked["_test_prediction"])
    stack_weight = deep_stack_blend["weight"]
    stack_validation_wape = deep_stack_blend["validation_wape"]
    stacked_blend_validation = deep_stack_blend["validation_prediction"]
    stacked_blend_test = deep_stack_blend["test_prediction"]
    stacked_blend = {
        "weight_on_deep_stack": float(stack_weight),
        "validation_wape": float(stack_validation_wape),
        "candidate_validation_wape": {
            name: float(metrics_dict(validation_actual, prediction)["wape"])
            for name, prediction in validation_policies.items()
        },
        "selected_policy": selected_source,
        "validation": metrics_dict(validation_actual, stacked_blend_validation),
        "test": metrics_dict(test_actual, stacked_blend_test),
    }
    validation_state = _predict_public_states(
        model, validation, target_device, text_cache,
        state_target_mode=state_target_mode)
    test_state = _predict_public_states(
        model, test, target_device, text_cache,
        state_target_mode=state_target_mode)
    # Internal prediction arrays are selection intermediates, not report data.
    for result in (stacked, ordinary_scaled_stacked, scaled_stacked, raw_ridge):
        result.pop("_validation_prediction", None)
        result.pop("_test_prediction", None)
        result.pop("_validation_residual_prediction", None)
        result.pop("_test_residual_prediction", None)
    return {
        "protocol": "grain_preserving_public_suite_end_to_end_demand_v1",
        "suite_manifest": json.loads((suite_root / "manifest.json").read_text())["dataset_version"],
        "model_inventory": model_inventory(model), "split": split_meta,
        "history": history, "validation_selected_blend_weight": weight,
        "best_validation_wape": best_validation_wape,
        "validation": {"neural": metrics_dict(validation_actual, validation_pred),
                        "transition": metrics_dict(validation_actual, transition_validation)},
        "test": {"neural": metrics_dict(test_actual, test_pred),
                  "transition": metrics_dict(test_actual, transition_test),
                  "blended": metrics_dict(test_actual, blended_test),
                  "stacked_ridge": stacked["test"],
                  "scaled_stacked_ridge": scaled_stacked["test"],
                  "raw_ridge": raw_ridge["test"],
                  "stacked_blend": stacked_blend["test"]},
        "validation_state": validation_state,
        "test_state": test_state,
        "validation_numeric_state": validation_state["numeric_projection"],
        "test_numeric_state": test_state["numeric_projection"],
        "state_thresholds": (list(state_thresholds)
                              if state_thresholds is not None else None),
        "state_loss_weight": float(state_loss_weight),
        "state_loss_mode": state_loss_mode,
        "state_target_mode": state_target_mode,
        "robust_alpha_policy": "reuse_ordinary_stack_selected_alpha",
        "raw_ridge": raw_ridge,
        "stacked_ridge": stacked,
        "scaled_stacked_ridge": scaled_stacked,
        "stacked_blend": stacked_blend,
        "promotion": {"status": "research_only",
                       "reason": "rolling-origin promotion is required"},
    }


def evaluate_publishable_rolling(root: Path, suite_root: Path, *,
                                 folds: Sequence[tuple[int, int]] = ((2017, 2018),
                                                                      (2018, 2019),
                                                                      (2019, 2020)),
                                 epochs: int = 1, batch_size: int = 2048,
                                 seed: int = 17,
                                 mask_drug_identity: bool = False,
                                 state_loss_mode: str = "categorical",
                                 state_target_mode: str = "change",
                                 validation_window_years: int = 1) -> dict:
    """Evaluate rolling origins with a predeclared validation window."""
    if validation_window_years < 1:
        raise ValueError("validation_window_years must be at least one")
    fold_results = []
    for train_last_year, validation_year in folds:
        validation_end_year = validation_year + validation_window_years - 1
        result = train_publishable_demand_model(
            root, suite_root, epochs=epochs, batch_size=batch_size,
            seed=seed, train_last_year=train_last_year,
            validation_year=validation_year,
            validation_end_year=validation_end_year,
            prediction_cap_quantile=0.995,
            mask_drug_identity=mask_drug_identity,
            state_loss_mode=state_loss_mode,
            state_target_mode=state_target_mode)
        baseline = result["test"]["raw_ridge"]["wape"]
        candidate = result["test"]["stacked_blend"]["wape"]
        improvement = ((baseline - candidate) / baseline
                       if baseline > 0 else float("nan"))
        fold_results.append({
            "train_last_year": train_last_year,
            "validation_year": validation_year,
            "validation_end_year": validation_end_year,
            "validation_window_years": validation_window_years,
            "test_start_year": validation_end_year + 1,
            "train_rows": result["split"]["train_rows"],
            "validation_rows": result["split"]["validation_rows"],
            "test_rows": result["split"]["test_rows"],
            "neural_wape": result["test"]["neural"]["wape"],
            "neural_within_5pct": result["test"]["neural"]["within_5pct"],
            "neural_within_tolerance": result["test"]["neural"][
                "within_tolerance"],
            "seasonal_wape": result.get("test", {}).get("seasonal", {}).get("wape"),
            "validation_state_exact_accuracy": result.get(
                "validation_state", {}).get("exact_accuracy"),
            "test_state_exact_accuracy": result.get(
                "test_state", {}).get("exact_accuracy"),
            "test_state_balanced_accuracy": result.get(
                "test_state", {}).get("balanced_accuracy"),
            "test_state_event_precision": result.get(
                "test_state", {}).get("event_metrics", {}).get(
                    "true_positive_precision"),
            "test_numeric_state_exact_accuracy": result.get(
                "test_numeric_state", {}).get("exact_accuracy"),
            "test_numeric_state_event_precision": result.get(
                "test_numeric_state", {}).get("event_metrics", {}).get(
                    "true_positive_precision"),
            "stacked_ridge_wape": result["test"]["stacked_ridge"]["wape"],
            "stacked_ridge_within_5pct": result["test"]["stacked_ridge"][
                "within_5pct"],
            "scaled_stacked_ridge_wape": result["test"][
                "scaled_stacked_ridge"]["wape"],
            "scaled_stacked_ridge_within_5pct": result["test"][
                "scaled_stacked_ridge"]["within_5pct"],
            "stacked_blend_wape": candidate,
            "stacked_blend_within_5pct": result["test"]["stacked_blend"][
                "within_5pct"],
            "stacked_blend_within_tolerance": result["test"]["stacked_blend"][
                "within_tolerance"],
            "selected_residual_scale": result["scaled_stacked_ridge"].get(
                "residual_scale"),
            "selected_robust_huber_multiplier": result[
                "scaled_stacked_ridge"].get("robust_huber_multiplier"),
            "selected_recency_gamma": result["scaled_stacked_ridge"].get(
                "recency_gamma"),
            "stacked_blend_weight_on_deep": result["stacked_blend"][
                "weight_on_deep_stack"],
            "fold_ridge_wape": baseline,
            "stacked_improvement_vs_fold_ridge": improvement,
            "best_validation_wape": result["best_validation_wape"],
            "prediction_cap_quantile": 0.995,
            "mask_drug_identity": mask_drug_identity,
            "state_loss_mode": state_loss_mode,
            "state_target_mode": state_target_mode,
        })
        fold_results[-1]["test_numeric_state_event_route_passed"] = event_route_passes(
            fold_results[-1]["test_numeric_state_exact_accuracy"],
            fold_results[-1]["test_numeric_state_event_precision"])
    improvements = np.asarray([
        row["stacked_improvement_vs_fold_ridge"] for row in fold_results
    ], dtype=float)
    finite = improvements[np.isfinite(improvements)]
    publishable = bool(len(fold_results) >= 3 and len(finite) == len(fold_results)
                       and np.all(finite >= 0.10))
    within_5 = np.asarray([
        row["stacked_blend_within_5pct"] for row in fold_results], dtype=float)
    within_tolerance = np.asarray([
        row["stacked_blend_within_tolerance"] for row in fold_results], dtype=float)
    state_exact = np.asarray([
        row["test_state_exact_accuracy"] for row in fold_results], dtype=float)
    finite_state = state_exact[np.isfinite(state_exact)]
    event_route = [bool(row["test_numeric_state_event_route_passed"])
                   for row in fold_results]
    return {
        "protocol": "grain_preserving_public_suite_rolling_end_to_end_v1",
        "folds": fold_results,
        "fold_count": len(fold_results),
        "mean_stacked_blend_improvement_vs_fold_ridge": (
            float(np.mean(finite)) if len(finite) else None),
        "mean_stacked_improvement_vs_fold_ridge": (
            float(np.mean(finite)) if len(finite) else None),
        "mean_stacked_blend_within_5pct": (
            float(np.mean(within_5)) if len(within_5) else None),
        "mean_stacked_blend_within_tolerance": (
            float(np.mean(within_tolerance)) if len(within_tolerance) else None),
        "mean_test_state_exact_accuracy": (
            float(np.mean(finite_state)) if len(finite_state) else None),
        "contract_75pct_numeric_accuracy": bool(
            len(within_5) >= 3 and np.all(within_5 >= 0.75)),
        "contract_75pct_state_accuracy": bool(
            len(finite_state) >= 3 and np.all(finite_state >= 0.75)),
        "contract_80pct_true_positive": bool(
            len(event_route) >= 3 and all(event_route)),
        "event_route_definition": {
            "raw_accuracy_floor": 0.65,
            "true_positive_precision_floor": 0.80,
            "signal": (
                "numeric_head_projected_to_absolute_five_state_demand_level"
                if state_target_mode == "level" else
                "numeric_head_projected_to_five_state_log_change"),
            "high_states": [3, 4],
        },
        "publishable_rolling_candidate": publishable,
        "promotion_reason": (
            "validation-gated deep-stack blend beats fold ridge by >=10% on every fold"
            if publishable else
            "research-only: validation-gated deep-stack blend does not meet the all-fold 10% gate"),
        "state_target_mode": state_target_mode,
    }
