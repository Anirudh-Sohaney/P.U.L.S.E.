"""Strict publishability audit; absence of evidence fails closed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import io
from .event_schema import EVENT_COLUMNS
from .universal_forecast import UNIVERSAL_OUTPUT_COLUMNS


def _gate(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_publishability_audit(cfg) -> dict:
    gates = []
    forecast_path = cfg.forecasts_dir / "universal_forecast.csv"
    if forecast_path.exists():
        forecast = io.load_csv(forecast_path)
        schema_ok = set(UNIVERSAL_OUTPUT_COLUMNS) <= set(forecast.columns)
        duplicate_columns = [c for c in ["forecast_timestamp", "horizon", "geography_level",
                                         "geography_id", "county_fips", "drug_key",
                                         "labeler", "target"] if c in forecast.columns]
        no_dupes = not forecast.duplicated(duplicate_columns).any()
        gates.append(_gate("universal_output_schema", schema_ok and no_dupes,
                           f"rows={len(forecast)} schema={schema_ok} duplicate_keys={not no_dupes}"))
        projected = forecast[forecast["geography_level"].astype(str).eq(
            "arkansas_supplier_drug")]
        projection_ok = bool(
            not projected.empty
            and projected["geography_id"].astype(str).eq("AR").all()
            and projected["target"].astype(str).eq(
                "arkansas_supplier_drug_shortage_pressure").all()
            and projected["county_fips"].fillna("").astype(str).eq("").all()
            and projected["confidence"].between(0.0, 0.25).all())
        gates.append(_gate(
            "arkansas_supplier_projection",
            projection_ok,
            f"rows={len(projected)} statewide_exact_drug_context={projection_ok}"))
    else:
        gates.append(_gate("universal_output_schema", False, "forecast artifact missing"))
        gates.append(_gate("arkansas_supplier_projection", False,
                           "forecast artifact missing"))

    event_path = cfg.artifact_path("events/article_events.csv.gz")
    if event_path.exists():
        events = io.load_csv(event_path)
        event_schema = set(EVENT_COLUMNS) <= set(events.columns)
        gates.append(_gate("event_schema", event_schema, f"events={len(events)}"))
        gold_path = cfg.artifact_path("events/expert_event_gold.csv.gz")
        gold_meta_path = cfg.artifact_path("events/expert_event_gold.json")
        gold_meta = (json.loads(gold_meta_path.read_text())
                     if gold_meta_path.exists() else {})
        gold_rows = int(gold_meta.get("rows", 0))
        gold_sources = gold_meta.get("source_datasets", [])
        gold_types = gold_meta.get("unit_types", {})
        gold_ok = bool(
            gold_path.exists() and gold_rows >= 2000
            and len(gold_sources) >= 2
            and "article_event" in gold_types
            and "sentence_event" in gold_types
            and int(gold_meta.get("positive_rows", 0)) > 0)
        gates.append(_gate(
            "expert_event_gold_set", gold_ok,
            f"rows={gold_rows}; unique_articles={gold_meta.get('unique_articles', 0)}; "
            f"sources={len(gold_sources)}; transfer_benchmark=True" if gold_ok else
            "no validated external expert-annotated >=2,000 article/event benchmark found"))
    else:
        gates.append(_gate("event_schema", False, "event artifact missing"))
        gates.append(_gate("expert_event_gold_set", False, "event artifact missing"))

    layer1_meta_path = cfg.artifact_path("news/layer1_news_state_features.json")
    if layer1_meta_path.exists():
        layer1_meta = json.loads(layer1_meta_path.read_text())
        variable_count = int(layer1_meta.get("variable_count", 0))
        gates.append(_gate("layer1_variable_contract", 50 <= variable_count <= 300,
                           f"variables={variable_count}; diseases={layer1_meta.get('disease_state_count', 0)}"))
    else:
        gates.append(_gate("layer1_variable_contract", False, "Layer 1 artifact metadata missing"))

    relevance_model_path = cfg.artifact_path("news/news_relevance_model.json")
    relevance_scores_path = cfg.artifact_path("news/relevance_scores.csv.gz")
    relevance_ok = False
    relevance_rows = 0
    relevance_test_balanced = float("nan")
    if relevance_model_path.exists() and relevance_scores_path.exists():
        try:
            relevance = json.loads(relevance_model_path.read_text())
            scores = io.load_csv(relevance_scores_path)
            relevance_rows = len(scores)
            relevance_test_balanced = float(
                relevance.get("metrics", {}).get("test", {}).get("balanced_accuracy", float("nan")))
            relevance_ok = relevance_rows > 0 and math.isfinite(relevance_test_balanced)
        except (OSError, ValueError, TypeError, KeyError):
            relevance_ok = False
    gates.append(_gate(
        "learned_news_relevance_layer", relevance_ok,
        f"score_rows={relevance_rows}; test_balanced_accuracy={relevance_test_balanced:.4f}; "
        "weak relevance evidence only, not event truth"))

    crosswalk_path = cfg.geography_dir / "city_county_crosswalk.csv"
    outcome_meta_path = cfg.artifact_path("outcomes/county_demand.json")
    outcome_meta = json.loads(outcome_meta_path.read_text()) if outcome_meta_path.exists() else {}
    if crosswalk_path.exists():
        crosswalk = io.load_csv(crosswalk_path)
        resolved = crosswalk["county_fips"].fillna("").astype(str).ne("").mean()
        weighted = float(outcome_meta.get("mapped_panel_row_fraction", 0.0))
        gates.append(_gate("county_coverage", weighted >= 0.90,
                           f"resolved_city_fraction={resolved:.4f}; mapped_panel_row_fraction={weighted:.4f}"))
    else:
        gates.append(_gate("county_coverage", False, "crosswalk missing"))
    outcomes_path = cfg.artifact_path("outcomes/county_demand.csv.gz")
    gates.append(_gate("county_level_outcome_labels", outcomes_path.exists() and
                       float(outcome_meta.get("mapped_panel_row_fraction", 0.0)) >= 0.90,
                       "real CMS city/provider demand aggregated to mapped counties" if outcomes_path.exists()
                       else "county outcome artifact missing"))

    supplier_path = cfg.artifact_path("suppliers/supplier_hierarchy.csv.gz")
    if supplier_path.exists():
        supplier = io.load_csv(supplier_path)
        labeler = supplier["labeler"].fillna("").astype(str).str.strip()
        labeler_coverage = float(labeler.ne("").mean()) if len(supplier) else 0.0
        context_path = cfg.artifact_path("suppliers/establishment_context.csv.gz")
        context_rows = len(io.load_csv(context_path)) if context_path.exists() else 0
        gates.append(_gate("supplier_local_usage_coverage",
                           labeler_coverage >= 0.90 and context_rows > 0,
                           f"product_labeler_fraction={labeler_coverage:.4f}; "
                           f"supplier_context_rows={context_rows}; "
                           "factory/API edges remain non-product-specific"))
    else:
        gates.append(_gate("supplier_local_usage_coverage", False, "supplier artifact missing"))

    supplier_rolling_path = cfg.evaluation_dir / "supplier_shortage_rolling_metrics.json"
    supplier_rolling = (json.loads(supplier_rolling_path.read_text())
                        if supplier_rolling_path.exists() else {})
    supplier_folds = supplier_rolling.get("folds", [])
    supplier_protocol_ok = bool(
        supplier_rolling.get("fold_count", 0) >= 3
        and len(supplier_folds) == supplier_rolling.get("fold_count", 0)
        and all(fold.get("validation_rows", 0) > 0
                and fold.get("fit_train_months")
                and fold.get("validation_months")
                and fold.get("test_months")
                for fold in supplier_folds))
    gates.append(_gate(
        "supplier_shortage_rolling_protocol",
        supplier_protocol_ok,
        f"folds={supplier_rolling.get('fold_count', 0)}; validation_reserved={supplier_protocol_ok}; "
        "FDA event evidence only"))

    rolling = cfg.evaluation_dir / "demand_rolling_metrics.json"
    rolling_data = json.loads(rolling.read_text()) if rolling.exists() else {}
    rolling_pass = bool(rolling_data.get("publishable_rolling_candidate", False))
    gates.append(_gate("rolling_origin_forecast_gate", rolling_pass,
                       "requires all required rolling folds and >=10% improvement"))
    quarterly = cfg.evaluation_dir / "quarterly_rolling_1y_metrics.json"
    qdata = json.loads(quarterly.read_text()) if quarterly.exists() else {}
    gates.append(_gate("quarterly_rolling_gate", bool(qdata.get("publishable_rolling_candidate", False)),
                       "quarterly gate evaluated from validation-selected rolling model"))

    arcos_path = cfg.evaluation_dir / "arcos_regional_metrics.json"
    arcos_roll_path = cfg.evaluation_dir / "arcos_regional_rolling_metrics.json"
    arcos = json.loads(arcos_path.read_text()) if arcos_path.exists() else {}
    arcos_roll = (json.loads(arcos_roll_path.read_text())
                  if arcos_roll_path.exists() else {})
    gates.append(_gate(
        "arcos_regional_pointwise_evaluation",
        bool(arcos.get("publishable_candidate", False)),
        f"pointwise_improvement={float(arcos.get('improvement_vs_best_naive', float('nan'))):.4f}; "
        f"rolling_candidate={bool(arcos_roll.get('publishable_rolling_candidate', False))}; "
        "rolling status remains reported separately"))

    metrics_path = cfg.evaluation_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    risk_rows = metrics.get("shortage_risk", {}).get("leaderboard", [])
    logistic = next((row for row in risk_rows if row.get("model") == "logistic_full"), None)
    boolean_accuracy = float(logistic.get("test_binary_accuracy", float("nan"))) if logistic else float("nan")
    balanced_accuracy = float(logistic.get("test_balanced_accuracy", float("nan"))) if logistic else float("nan")
    gates.append(_gate(
        "shortage_boolean_accuracy",
        bool(math.isfinite(boolean_accuracy) and boolean_accuracy >= 0.75
             and math.isfinite(balanced_accuracy) and balanced_accuracy >= 0.75),
        f"held-out logistic raw_accuracy={boolean_accuracy:.4f}; "
        f"balanced_accuracy={balanced_accuracy:.4f}; requires both >=0.75"))

    modular_rolling_path = cfg.evaluation_dir / "modular_rolling_metrics.json"
    modular_rolling = (json.loads(modular_rolling_path.read_text())
                       if modular_rolling_path.exists() else {})
    gates.append(_gate(
        "modular_rolling_gate",
        bool(modular_rolling.get("publishable_rolling_candidate", False)),
        f"folds={int(modular_rolling.get('fold_count', 0))}; "
        f"mean_improvement={float(modular_rolling.get('mean_improvement_vs_strongest_naive', float('nan'))):.4f}; "
        "requires every research-model fold to beat both temporal baselines by >=10%"))

    result = {"model": "arkansas-universal", "publishable": all(x["passed"] for x in gates),
              "gates": gates, "failed_gate_count": sum(not x["passed"] for x in gates)}
    io.write_json(result, cfg.evaluation_dir / "publishability_audit.json")
    return result
