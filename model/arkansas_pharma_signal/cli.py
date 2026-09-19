"""End-to-end CLI: build-panel, train, forecast, evaluate.

Example::

    PYTHONPATH=model data/.venv/bin/python -m arkansas_pharma_signal.cli build-panel --root .
"""

from __future__ import annotations

import argparse
from datetime import date
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import io
from .config import Config, resolve_root
from .datasets import ablation_features, build_next_period, feature_columns_for_mode
from .evaluate import (
    _fit_ridge_log,
    _neural_subsample,
    _simplex_weights,
    _select_binary_threshold,
    evaluate_demand_rolling,
    evaluate_next_period,
    evaluate_risk_rolling,
)
from .features import build_panel
from .forecast import build_forecast_grid, forecast_input_contract
from .neural import NeuralSignalBlender
from .quarterly import (
    build_quarterly_forecast_grid,
    evaluate_quarterly,
    evaluate_quarterly_rolling,
    load_medicaid_sdud_panel,
)
from .canonical_graph import build_canonical_graph
from .expert_gold import build_expert_event_gold, expert_event_gold_metadata
from .news_corpus import load_historical_corpus
from .universal_forecast import build_universal_forecast_grid, validate_universal_forecast
from .geography import (
    build_city_county_crosswalk, fetch_census_county_weather_points,
    load_city_county_crosswalk, mapping_dict,
)
from .historical_weather import build_county_weather_panel, load_noaa_daily_summaries
from .event_extraction import extract_events
from .supplier_hierarchy import (build_establishment_context, build_supplier_hierarchy,
                                  validate_supplier_hierarchy)
from .publishability import run_publishability_audit
from .county_outcomes import build_county_outcomes, validate_county_outcomes
from .county_evaluation import (
    evaluate_county_demand,
    evaluate_county_demand_by_region,
    evaluate_county_demand_rolling,
    load_county_demand,
)
from .annotation import build_annotation_manifest
from .external_validation import (
    evaluate_band_reference, evaluate_biocaster_reference, evaluate_cirad_reference,
    evaluate_daniel_reference, evaluate_expert_event_gold,
    evaluate_eventepi_reference,
)
from .news_signals import DISEASE_SIGNAL_COLUMNS, build_news_state_features
from .regression import LogisticRidge, _fill_nan
from .input_contract import summarize_dispositions
from .supplier_shortage import (
    build_supplier_shortage_scores,
    evaluate_supplier_shortage,
    evaluate_supplier_shortage_rolling,
    load_fda_shortage_records,
    normalize_drug,
    normalize_supplier,
)
from .live_inputs import (
    build_nws_daily_weather_features,
    fetch_nws_county_hourly_forecasts,
    fetch_medicaid_sdud_csv,
    fetch_nws_hourly_forecast,
    fetch_openfda_shortages,
)
from .signal_utility import (
    build_fluview_weather_view, build_hospital_weather_view,
    build_hospital_disaster_view,
    evaluate_hospital_disaster_incremental_utility,
    evaluate_hospital_weather_incremental_utility,
    evaluate_weather_incremental_utility,
)
from .arcos_evaluation import (
    SOURCE_RELATIVE, build_next_quarter_view, evaluate_arcos_rolling_view,
    evaluate_arcos_view, load_arcos,
)
from .news_relevance import score_corpus, train_news_relevance
from .learned_event_state import train_cirad_event_state
from .news_only_adapter import materialize_news_only_features


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="arkansas_pharma_signal",
        description="Arkansas pharmaceutical demand, supply-risk, and shortage-"
                    "impact forecasting from real local data.",
    )
    p.add_argument("--root", default=".", help="Repo root containing data/ and model/.")
    sub = p.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-panel", help="Build the annual Arkansas demand panel.")
    p_build.add_argument("--max-rows", type=int, default=None,
                         help="Cap panel rows (debugging).")

    p_train = sub.add_parser("train", help="Train next-period demand, risk, and neural models.")
    p_train.add_argument("--max-rows", type=int, default=None)
    p_train.add_argument("--include-periodic-training-features", action="store_true",
                         help="Train on the full feature set including CMS/Medicaid/"
                              "provider periodic-training-only columns (default: "
                              "operational mode excludes them).")

    p_fc = sub.add_parser("forecast", help="Write forecast grid (>=100 rows).")
    p_fc.add_argument("--max-rows", type=int, default=10000,
                      help="Row budget; 50000 scales coverage across more city-drugs.")
    p_fc.add_argument("--forecast-years", type=int, default=2)
    p_fc.add_argument(
        "--allow-training-only-features", action="store_true",
        help="Permit forecasting with a model trained on periodic/historical-only inputs; "
             "without this flag operational mode fails closed.")

    p_qfc = sub.add_parser(
        "forecast-quarterly",
        help="Write state-level Arkansas Medicaid quarterly demand forecast.")
    p_qfc.add_argument("--max-rows", type=int, default=10000)
    p_qfc.add_argument(
        "--live-sdud-url", action="append", default=None,
        help="Optional official CMS SDUD annual CSV URL; repeat for multiple "
             "years. Refreshes the target in memory only; never changes the "
             "historical cache.")

    p_ev = sub.add_parser("evaluate", help="Strict next-period evaluation.")
    p_ev.add_argument("--train-cutoff", type=int, default=2021)
    p_ev.add_argument("--rolling-demand", action="store_true",
                      help="Also run rolling annual demand folds.")
    p_ev.add_argument("--demand-min-train-years", type=int, default=4)
    p_ev.add_argument("--rolling-demand-test-window-years", type=int, default=None,
                      help="If set, limit annual demand folds to this many "
                           "feature years after validation.")
    p_ev.add_argument("--rolling-risk", action="store_true",
                      help="Also run rolling annual shortage-risk folds.")
    p_ev.add_argument("--risk-min-train-years", type=int, default=4)
    p_ev.add_argument("--include-periodic-training-features", action="store_true",
                      help="Evaluate on the full feature set including CMS/"
                           "Medicaid/provider periodic-training-only columns "
                           "(default: operational mode excludes them).")
    p_ev.add_argument("--no-neural", action="store_true",
                      help="Skip the expensive neural blend and evaluate the "
                           "deterministic baseline/ablation models only.")

    p_qev = sub.add_parser(
        "evaluate-quarterly",
        help="Strict next-quarter evaluation on Arkansas Medicaid SDUD data.")
    p_qev.add_argument("--train-cutoff", type=int, default=2021)
    p_qev.add_argument("--rolling", action="store_true",
                       help="Also run rolling-origin strict quarterly folds.")
    p_qev.add_argument("--min-train-years", type=int, default=4)
    p_qev.add_argument("--rolling-test-window-years", type=int, default=None,
                       help="If set, limit each rolling fold to this many "
                           "feature years after validation.")
    p_qev.add_argument(
        "--fluview-path", default=None,
        help="CDC FluView AR/national weekly CSV (default: repo "
             "disease_surveillance_current file when present).")
    p_qev.add_argument(
        "--wastewater-path", default=None,
        help="CDC Arkansas site-level wastewater weekly CSV (default: repo "
             "disease_surveillance_current file when present).")
    p_qev.add_argument(
        "--nadac-path", default=None,
        help="CMS NADAC weekly NDC price CSV (default: local NADAC artifact "
             "when present).")
    p_qev.add_argument(
        "--live-sdud-url", action="append", default=None,
        help="Optional official CMS SDUD annual CSV URL; repeat for multiple "
             "years. Refreshes the target in memory only; never changes "
             "historical artifacts.")

    p_graph = sub.add_parser("build-graph", help="Build canonical entity graph with edge provenance.")
    p_corpus = sub.add_parser("build-news-corpus", help="Restore historical article text and metadata.")
    p_news = sub.add_parser("build-news-signals", help="Build Layer 1 news-state variables from article events.")
    p_news_only = sub.add_parser(
        "build-news-only-features",
        help="Import and validate the separate news-only SLM feature surface.")
    p_news_only.add_argument(
        "--source", type=Path, default=None,
        help="Optional dated news-only CSV; defaults to the configured source.")
    p_uc = sub.add_parser("forecast-universal", help="Write county x drug x supplier forecast contract.")
    p_uc.add_argument("--max-rows", type=int, default=10000)
    p_uc.add_argument("--county-fips", nargs="*", default=None)
    p_uc.add_argument("--region", dest="regions", nargs="*", default=None)
    p_uc.add_argument("--supplier", dest="suppliers", nargs="*", default=None)
    p_uc.add_argument("--live-fda-shortages", action="store_true",
                      help="Refresh supplier shortage context from openFDA before forecasting.")
    p_uc.add_argument("--allow-training-only-features", action="store_true",
                      help="Allow a research-only universal forecast from a model "
                          "trained with periodic or historical-only features.")
    p_weather = sub.add_parser(
        "refresh-weather",
        help="Fetch opt-in NWS hourly weather context and a daily feature view.")
    p_weather.add_argument("--latitude", type=float, required=True,
                           help="Forecast point latitude in decimal degrees.")
    p_weather.add_argument("--longitude", type=float, required=True,
                           help="Forecast point longitude in decimal degrees.")
    p_weather.add_argument(
        "--county-fips", default=None,
        help="Optional verified Arkansas county FIPS tag for this point. The CLI "
             "does not infer county membership from coordinates.")
    p_weather.add_argument("--timeout", type=int, default=30)
    p_weather.add_argument("--output", type=Path, default=None,
                           help="Raw hourly CSV path; defaults to artifacts/live/.")
    p_weather.add_argument("--daily-output", type=Path, default=None,
                           help="Daily feature CSV path; defaults to artifacts/live/.")
    p_county_geo = sub.add_parser(
        "build-weather-geography",
        help="Fetch the authoritative Census Arkansas county internal-point registry.")
    p_county_geo.add_argument("--year", type=int, default=2025)
    p_county_geo.add_argument("--timeout", type=int, default=30)
    p_county_geo.add_argument("--output", type=Path, default=None)
    p_county_weather = sub.add_parser(
        "refresh-county-weather",
        help="Fetch NWS hourly and daily context for every mapped Arkansas county.")
    p_county_weather.add_argument("--registry", type=Path, default=None)
    p_county_weather.add_argument("--timeout", type=int, default=30)
    p_county_weather.add_argument("--output", type=Path, default=None)
    p_county_weather.add_argument("--daily-output", type=Path, default=None)
    p_hist_weather = sub.add_parser(
        "build-historical-weather",
        help="Aggregate local NOAA daily station observations to county weeks/months.")
    p_hist_weather.add_argument("--raw-dir", type=Path, default=None,
                                help="NOAA daily-summary JSON directory.")
    p_hist_weather.add_argument("--registry", type=Path, default=None)
    p_hist_weather.add_argument("--output", type=Path, default=None)
    p_geo = sub.add_parser("build-geography", help="Resolve Arkansas panel cities to counties.")
    p_geo.add_argument("--workers", type=int, default=8)
    p_events = sub.add_parser("build-events", help="Extract structured events from available article text.")
    p_sup = sub.add_parser("build-supplier-hierarchy", help="Build supplier hierarchy coverage artifact.")
    p_audit = sub.add_parser("audit-publishability", help="Run strict final publishability gates.")
    p_ext = sub.add_parser("validate-external-reference",
                           help="Evaluate public epidemiology/news references without training use.")
    p_outcomes = sub.add_parser("build-county-outcomes", help="Aggregate mapped real CMS demand to counties.")
    p_cev = sub.add_parser(
        "evaluate-county-demand",
        help="Strict next-year county x drug demand evaluation on the real "
             "county outcome artifact.")
    p_cev.add_argument("--train-cutoff", type=int, default=2021,
                       help="Validation feature year; train <= cutoff-1, test > cutoff.")
    p_cev.add_argument("--min-train-years", type=int, default=4,
                       help="Minimum train years per rolling-origin fold.")
    p_ann = sub.add_parser("build-annotation-manifest", help="Create an unlabeled expert annotation manifest.")
    p_ann.add_argument("--count", type=int, default=2000)
    sub.add_parser("build-expert-event-gold",
                   help="Build the external expert-labeled event evaluation artifact.")

    p_sse = sub.add_parser(
        "evaluate-supplier-shortage",
        help="Strict next-month FDA supplier-drug shortage event evaluation.")
    p_sse.add_argument("--min-train-months", type=int, default=24,
                       help="Minimum train months per rolling-origin fold.")
    p_sse.add_argument("--test-window-months", type=int, default=4,
                       help="Test window months per rolling-origin fold.")

    p_arcos = sub.add_parser(
        "evaluate-arcos-regional",
        help="Strict next-quarter Arkansas ZIP3/drug ARCOS distribution evaluation.")
    p_arcos.add_argument("--train-end-year", type=int, default=2018)
    p_arcos.add_argument("--rolling-min-train-quarters", type=int, default=12)
    p_arcos.add_argument("--rolling-test-window-quarters", type=int, default=4)
    p_arcos.add_argument("--rolling-validation-quarters", type=int, default=4)
    p_weather_utility = sub.add_parser(
        "evaluate-weather-utility",
        help="Evaluate historical weather as an incremental weekly FluView feature.")
    p_weather_utility.add_argument("--fluview-path", type=Path, default=None)
    p_weather_utility.add_argument("--weather-path", type=Path, default=None)
    p_hospital_weather_utility = sub.add_parser(
        "evaluate-hospital-weather-utility",
        help="Evaluate historical weather as an incremental hospital-influenza feature.")
    p_hospital_weather_utility.add_argument("--hospital-path", type=Path, default=None)
    p_hospital_weather_utility.add_argument("--weather-path", type=Path, default=None)
    p_hospital_disaster_utility = sub.add_parser(
        "evaluate-hospital-disaster-utility",
        help="Evaluate FEMA county declarations as an incremental hospital feature.")
    p_hospital_disaster_utility.add_argument("--hospital-path", type=Path, default=None)
    p_hospital_disaster_utility.add_argument("--disaster-path", type=Path, default=None)

    p_nrel = sub.add_parser(
        "train-news-relevance",
        help="Train weakly supervised news relevance model with chronological validation.")
    p_nrel.add_argument("--feature-dimension", type=int, default=512)
    p_nrel.add_argument("--train-end", default="2022-03-31")
    p_nrel.add_argument("--validation-end", default="2022-05-31")
    p_event_ml = sub.add_parser(
        "train-research-event-state",
        help="Train the article-grouped CIRAD event-state research candidate.")
    p_event_ml.add_argument("--feature-dimension", type=int, default=512)

    for pa in (p_build, p_train, p_fc, p_qfc, p_ev, p_qev, p_graph, p_corpus, p_news,
               p_uc, p_weather, p_news_only, p_county_geo, p_county_weather, p_hist_weather,
               p_geo, p_events, p_sup, p_audit, p_ext, p_outcomes, p_cev, p_ann,
               p_sse, p_arcos, p_weather_utility, p_hospital_weather_utility,
               p_hospital_disaster_utility,
               p_nrel, p_event_ml):
        # Do not let a subparser's default overwrite a parent-level --root.
        # This matters for isolated live-refresh commands and scripted calls.
        pa.add_argument("--root", default=argparse.SUPPRESS,
                         help="Repo root (or set on parent).")
    for pa in (p_build, p_train):
        pa.add_argument("--no-save", action="store_true",
                        help="Return the panel without writing artifacts.")
    return p.parse_args(argv)


def _make_cfg(args: argparse.Namespace) -> Config:
    root = resolve_root(args.root or ".")
    cfg = Config(root=str(root))
    cfg.ensure_dirs()
    return cfg


def _load_panel(cfg: Config, args: argparse.Namespace, cap: bool = True):
    path = cfg.panel_dir / "panel.csv"
    if path.exists():
        panel = io.load_csv(path)
        if cap and getattr(args, "max_rows", None):
            return panel.head(args.max_rows)
        return panel
    panel = build_panel(cfg)
    if cap and getattr(args, "max_rows", None):
        panel = panel.head(args.max_rows)
    io.write_csv(panel, path)
    return panel


def _feature_matrix(view: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return _fill_nan(view[cols].to_numpy(dtype=float))


def cmd_build_panel(cfg: Config, args: argparse.Namespace) -> None:
    panel = build_panel(cfg)
    if args.max_rows:
        panel = panel.head(args.max_rows)
    print(f"panel rows: {len(panel)}")
    print(f"panel years: {int(panel['year'].min())}-{int(panel['year'].max())}")
    print(f"drugs: {panel['drug_key'].nunique()}  cities: {panel['city'].nunique()}")
    if not args.no_save:
        out = cfg.panel_dir / "panel.csv"
        io.write_csv(panel, out)
        io.write_metadata(
            cfg.metadata_dir / "panel.json",
            rows=int(len(panel)),
            years=[int(panel["year"].min()), int(panel["year"].max())],
            drugs=int(panel["drug_key"].nunique()),
            ingredient_map_rate=float(panel["ingredient"].astype(bool).mean()),
            labeler_map_rate=float(panel["labeler"].astype(bool).mean()),
            feature_columns=list(panel.columns),
        )
        print(f"wrote {out}")


def cmd_build_graph(cfg: Config) -> None:
    graph = build_canonical_graph(cfg)
    graph.save(cfg)
    print(f"graph nodes: {len(graph.nodes)} edges: {len(graph.edges)}")


def cmd_build_corpus(cfg: Config) -> None:
    corpus = load_historical_corpus(cfg)
    out = cfg.artifact_path("news/historical_corpus.csv.gz")
    io.write_csv(corpus, out)
    io.write_metadata(cfg.artifact_path("news/historical_corpus.json"), rows=len(corpus),
                      full_text_rows=int(corpus["is_full_text"].sum()) if len(corpus) else 0,
                      metadata_only_rows=int((~corpus["is_full_text"]).sum()) if len(corpus) else 0)
    print(f"corpus rows: {len(corpus)} full_text: {int(corpus['is_full_text'].sum()) if len(corpus) else 0}")


def cmd_forecast_universal(cfg: Config, args: argparse.Namespace) -> None:
    # ``max_rows`` is an output cap for the universal grid.  Capping the
    # source panel first silently reduced the product universe to the first
    # few drugs in file order and made the advertised 50+ drug surface false.
    panel = _load_panel(cfg, args, cap=False)
    crosswalk = load_city_county_crosswalk(cfg)
    events_path = cfg.artifact_path("events/article_events.csv.gz")
    events = io.load_csv(events_path) if events_path.exists() else None
    state_events_path = cfg.data_path(cfg.events)
    state_events = io.load_csv(state_events_path) if state_events_path.exists() else None
    supplier_path = cfg.artifact_path("suppliers/supplier_hierarchy.csv.gz")
    supplier = io.load_csv(supplier_path) if supplier_path.exists() else None
    arcos_path = cfg.data_dir / SOURCE_RELATIVE
    arcos_panel = load_arcos(arcos_path) if arcos_path.exists() else None
    shortage_scores = None
    live_meta = None
    try:
        shortage_records = load_fda_shortage_records(cfg)
        if args.live_fda_shortages:
            live_records, live_meta = fetch_openfda_shortages()
            if not live_records.empty:
                live_records["supplier"] = live_records["supplier"].map(normalize_supplier)
                live_records["drug"] = live_records["drug"].map(normalize_drug)
                shortage_records = pd.concat([
                    shortage_records[["supplier", "drug", "event_date", "month"]],
                    live_records[["supplier", "drug", "event_date", "month"]],
                ], ignore_index=True).drop_duplicates(
                    ["supplier", "drug", "month"])
        shortage_scores = build_supplier_shortage_scores(shortage_records)
    except FileNotFoundError:
        pass
    trained_path = cfg.trained_dir / "models.json"
    trained = io.read_json(trained_path) if trained_path.exists() else None
    contract = forecast_input_contract(trained or {})
    _require_operational_forecast(
        contract, allow_training_only_features=args.allow_training_only_features)
    forecast = build_universal_forecast_grid(panel, cfg, city_to_county=mapping_dict(crosswalk),
                                             max_rows=args.max_rows, events=events,
                                             state_events=state_events,
                                             supplier_lookup=supplier, trained=trained,
                                             county_fips=args.county_fips, regions=args.regions,
                                             suppliers=args.suppliers,
                                             arcos_panel=arcos_panel,
                                             supplier_shortage_scores=shortage_scores,
                                             model_version="universal-v2")
    validate_universal_forecast(forecast)
    out = cfg.forecasts_dir / "universal_forecast.csv"
    io.write_csv(forecast, out)
    io.write_metadata(cfg.metadata_dir / "universal_forecast.json", rows=len(forecast),
                      columns=list(forecast.columns), model_version="universal-v2",
                      county_coverage_rows=int(forecast["county_fips"].fillna("").astype(str).ne("").sum()),
                      unresolved_coverage_rows=int(forecast["evidence_type"].eq("unresolved").sum()),
                      parent_coverage_rows=int(forecast["parent_company"].fillna("").astype(str).ne("").sum()),
                      factory_coverage_rows=int(forecast["factory"].fillna("").astype(str).ne("").sum()),
                      api_source_coverage_rows=int(forecast["api_source"].fillna("").astype(str).ne("").sum()),
                      geography_levels=sorted(forecast["geography_level"].dropna().astype(str).unique().tolist()),
                      region_rows=int(forecast["geography_level"].eq("region").sum()),
                      zip3_rows=int(forecast["geography_level"].eq("zip3").sum()),
                      neighbor_state_rows=int(forecast["geography_level"].eq("neighbor_state").sum()),
                      supplier_drug_rows=int(forecast["geography_level"].eq("supplier_drug").sum()),
                      arkansas_supplier_drug_rows=int(
                          forecast["geography_level"].eq("arkansas_supplier_drug").sum()),
                      live_fda_shortages=bool(args.live_fda_shortages),
                      live_fda_metadata=live_meta,
                      shortage_state_threshold=float((trained or {}).get("shortage_risk", {}).get(
                          "shortage_threshold", 0.5)),
                      input_contract={
                          "operational_feature_count": contract["operational_feature_count"],
                          "periodic_training_only_count": contract["periodic_training_only_count"],
                          "derived_training_variable_count": contract["derived_training_variable_count"],
                          "static_identity_context_count": contract["static_identity_context_count"],
                          "operational_ready": contract["operational_ready"],
                      },
                      allow_training_only_features=bool(args.allow_training_only_features),
                      forecast_mode=("research_training_only" if args.allow_training_only_features
                                     and not contract["operational_ready"] else "operational"),
                      operational_ready=contract["operational_ready"])
    print(f"wrote {out} rows: {len(forecast)}")


def cmd_refresh_weather(cfg: Config, args: argparse.Namespace) -> None:
    """Refresh live NWS context without modifying historical datasets."""
    hourly, source_metadata = fetch_nws_hourly_forecast(
        args.latitude, args.longitude, county_fips=args.county_fips,
        timeout=args.timeout)
    daily = build_nws_daily_weather_features(hourly)
    raw_out = args.output or cfg.artifact_path("live/nws_weather_hourly.csv")
    daily_out = args.daily_output or cfg.artifact_path("live/nws_weather_daily.csv")
    io.write_csv(hourly, raw_out)
    io.write_csv(daily, daily_out)
    io.write_metadata(
        cfg.metadata_dir / "nws_weather_refresh.json",
        **source_metadata,
        raw_hourly_path=str(raw_out), daily_feature_path=str(daily_out),
        hourly_rows=int(len(hourly)), daily_rows=int(len(daily)),
        historical_training_modified=False,
        geography_contract=(
            "caller_supplied_county_fips" if args.county_fips
            else "point_only_no_county_inference"),
        feature_boundary="build_nws_daily_weather_features",
    )
    print(f"wrote {raw_out} rows: {len(hourly)}")
    print(f"wrote {daily_out} rows: {len(daily)}")


def cmd_build_weather_geography(cfg: Config, args: argparse.Namespace) -> None:
    """Build a versioned Census county internal-point weather registry."""
    points, metadata = fetch_census_county_weather_points(
        year=args.year, timeout=args.timeout)
    out = args.output or cfg.geography_dir / "arkansas_county_weather_points.csv"
    io.write_csv(points, out)
    io.write_metadata(cfg.metadata_dir / "arkansas_county_weather_points.json",
                      **metadata, output_path=str(out))
    print(f"wrote {out} rows: {len(points)}")


def cmd_refresh_county_weather(cfg: Config, args: argparse.Namespace) -> None:
    """Refresh complete county-keyed NWS weather context without interpolation."""
    registry = args.registry or cfg.geography_dir / "arkansas_county_weather_points.csv"
    points = io.load_csv(registry)
    hourly, metadata = fetch_nws_county_hourly_forecasts(points, timeout=args.timeout)
    daily = build_nws_daily_weather_features(hourly)
    raw_out = args.output or cfg.artifact_path("live/nws_arkansas_county_weather_hourly.csv")
    daily_out = args.daily_output or cfg.artifact_path("live/nws_arkansas_county_weather_daily.csv")
    io.write_csv(hourly, raw_out)
    io.write_csv(daily, daily_out)
    io.write_metadata(cfg.metadata_dir / "nws_arkansas_county_weather_refresh.json",
                      **metadata, registry_path=str(registry), raw_hourly_path=str(raw_out),
                      daily_feature_path=str(daily_out), historical_training_modified=False,
                      feature_boundary="build_nws_daily_weather_features")
    print(f"wrote {raw_out} rows: {len(hourly)}")
    print(f"wrote {daily_out} rows: {len(daily)}")


def cmd_build_historical_weather(cfg: Config, args: argparse.Namespace) -> None:
    """Build observed county-week/month weather rows for historical joins."""
    raw_dir = args.raw_dir or cfg.data_dir / "final_data/raw_downloads"
    registry = args.registry or cfg.geography_dir / "arkansas_county_weather_points.csv"
    daily = load_noaa_daily_summaries(raw_dir)
    points = io.load_csv(registry)
    panel, metadata = build_county_weather_panel(daily, points)
    out = args.output or cfg.artifact_path("weather/arkansas_county_weather_weekly_monthly.csv.gz")
    io.write_csv(panel, out)
    io.write_metadata(cfg.metadata_dir / "arkansas_county_weather_historical.json",
                      **metadata, raw_dir=str(raw_dir), registry_path=str(registry),
                      output_path=str(out), historical_training_modified=False)
    print(f"wrote {out} rows: {len(panel)}")


def cmd_evaluate_weather_utility(cfg: Config, args: argparse.Namespace) -> None:
    """Run the chronological weather-versus-baseline public utility ablation."""
    flu_path = (args.fluview_path or cfg.data_dir /
                "targeted_additions/disease_surveillance_current/data/"
                "cdc_fluview_ar_national_weekly.csv.gz")
    weather_path = (args.weather_path or cfg.artifact_path(
        "weather/arkansas_county_weather_weekly_monthly.csv.gz"))
    flu = io.load_csv(flu_path)
    weather = io.load_csv(weather_path)
    view = build_fluview_weather_view(flu, weather)
    result = evaluate_weather_incremental_utility(view)
    result.update({"view_rows": int(len(view)), "fluview_path": str(flu_path),
                   "weather_path": str(weather_path),
                   "weather_periods": int(weather["period_start"].nunique())})
    out = cfg.artifact_path("evaluation/weather_fluview_utility.json")
    io.write_json(result, out)
    print(f"wrote {out}")
    print(f"view_rows: {len(view)}")
    print(f"mean_balanced_accuracy_delta: {result.get('mean_balanced_accuracy_delta')}")


def cmd_evaluate_hospital_weather_utility(cfg: Config, args: argparse.Namespace) -> None:
    """Run the weather ablation against the weekly hospital influenza proxy."""
    hospital_path = (args.hospital_path or cfg.data_dir /
                     "targeted_additions/cdc_hospital_respiratory_current/data/"
                     "cdc_hospital_respiratory_weekly.json")
    weather_path = (args.weather_path or cfg.artifact_path(
        "weather/arkansas_county_weather_weekly_monthly.csv.gz"))
    hospital = io.read_json(hospital_path)
    weather = io.load_csv(weather_path)
    view = build_hospital_weather_view(pd.DataFrame(hospital), weather)
    result = evaluate_hospital_weather_incremental_utility(view)
    result.update({"view_rows": int(len(view)), "hospital_path": str(hospital_path),
                   "weather_path": str(weather_path)})
    out = cfg.artifact_path("evaluation/hospital_weather_utility.json")
    io.write_json(result, out)
    print(f"wrote {out}")
    print(f"view_rows: {len(view)}")
    print(f"mean_balanced_accuracy_delta: {result.get('mean_balanced_accuracy_delta')}")


def cmd_evaluate_hospital_disaster_utility(cfg: Config, args: argparse.Namespace) -> None:
    """Run the FEMA declaration ablation against hospital influenza pressure."""
    hospital_path = (args.hospital_path or cfg.data_dir /
                     "targeted_additions/cdc_hospital_respiratory_current/data/"
                     "cdc_hospital_respiratory_weekly.json")
    disaster_path = (args.disaster_path or cfg.data_dir /
                     "final_data/features/disasters/disasters_features.csv.gz")
    hospital = io.read_json(hospital_path)
    disasters = io.load_csv(disaster_path)
    view = build_hospital_disaster_view(pd.DataFrame(hospital), disasters)
    result = evaluate_hospital_disaster_incremental_utility(view)
    result.update({"view_rows": int(len(view)), "hospital_path": str(hospital_path),
                   "disaster_path": str(disaster_path)})
    out = cfg.artifact_path("evaluation/hospital_disaster_utility.json")
    io.write_json(result, out)
    print(f"wrote {out}")
    print(f"view_rows: {len(view)}")
    print(f"mean_balanced_accuracy_delta: {result.get('mean_balanced_accuracy_delta')}")


def cmd_build_geography(cfg: Config, args: argparse.Namespace) -> None:
    panel = _load_panel(cfg, args)
    crosswalk = build_city_county_crosswalk(cfg, panel["city"].dropna().astype(str).unique().tolist(),
                                            workers=args.workers)
    out = cfg.geography_dir / "city_county_crosswalk.csv"
    io.write_csv(crosswalk, out)
    io.write_metadata(cfg.metadata_dir / "city_county_crosswalk.json", rows=len(crosswalk),
                      resolved_rows=int(crosswalk["county_fips"].astype(str).ne("").sum()),
                      unresolved_rows=int(crosswalk["county_fips"].astype(str).eq("").sum()),
                      source="US Census Geocoder")
    print(f"wrote {out}; resolved {int(crosswalk['county_fips'].astype(str).ne('').sum())}/{len(crosswalk)}")


def cmd_build_events(cfg: Config) -> None:
    corpus_path = cfg.artifact_path("news/historical_corpus.csv.gz")
    corpus = io.load_csv(corpus_path)
    dictionary = io.read_json(cfg.data_path(cfg.drug_dictionary))
    names = [x.get("canonical_name", "") for x in dictionary if isinstance(x, dict)]
    crosswalk = load_city_county_crosswalk(cfg)
    events = extract_events(corpus, names, mapping_dict(crosswalk))
    out = cfg.artifact_path("events/article_events.csv.gz")
    io.write_csv(events, out)
    io.write_metadata(cfg.artifact_path("events/article_events.json"), rows=len(events),
                      full_text_articles=int(corpus["is_full_text"].fillna(False).sum()),
                      extraction_method="deterministic article-span rules",
                      event_types=sorted(events["event_type"].unique().tolist()) if len(events) else [])
    print(f"wrote {out}; events: {len(events)}")


def cmd_build_news_signals(cfg: Config) -> None:
    events = io.load_csv(cfg.artifact_path("events/article_events.csv.gz"))
    relevance_path = cfg.artifact_path("news/relevance_scores.csv.gz")
    relevance = io.load_csv(relevance_path) if relevance_path.exists() else None
    signals = build_news_state_features(events, relevance_scores=relevance)
    out = cfg.artifact_path("news/layer1_news_state_features.csv.gz")
    io.write_csv(signals, out)
    io.write_metadata(cfg.artifact_path("news/layer1_news_state_features.json"),
                      rows=len(signals), variable_count=len(signals.columns) - 2,
                      disease_state_count=len(DISEASE_SIGNAL_COLUMNS) // 4,
                      disease_state_fields=["present", "outbreak_risk", "spread_rate", "uncertain_count"],
                      event_types=list(sorted(set(signals.columns) - {"observation_date", "geography_key"})),
                      source="real extracted article events",
                      learned_relevance_scores=bool(relevance is not None))
    print(f"wrote {out}; rows: {len(signals)} variables: {len(signals.columns) - 2}")


def cmd_build_news_only_features(cfg: Config, args: argparse.Namespace) -> None:
    """Materialize the separate SLM's validated 20-column feature output."""
    out, metadata = materialize_news_only_features(cfg, args.source)
    print(f"wrote {out}; rows: {metadata['rows']} signals: {metadata['signal_count']}")


def cmd_build_supplier_hierarchy(cfg: Config) -> None:
    hierarchy = build_supplier_hierarchy(cfg)
    validate_supplier_hierarchy(hierarchy)
    out = cfg.artifact_path("suppliers/supplier_hierarchy.csv.gz")
    io.write_csv(hierarchy, out)
    context = build_establishment_context(cfg)
    context_out = cfg.artifact_path("suppliers/establishment_context.csv.gz")
    io.write_csv(context, context_out)
    io.write_metadata(cfg.artifact_path("suppliers/supplier_hierarchy.json"), rows=len(hierarchy),
                      labeler_rows=int(hierarchy["labeler"].ne("").sum()),
                      parent_rows=int(hierarchy["parent_company"].ne("").sum()),
                      factory_rows=int(hierarchy["factory"].ne("").sum()),
                      api_source_rows=int(hierarchy["api_source"].ne("").sum()),
                      source="FDA NDC listing",
                      establishment_context_suppliers=int(len(context)),
                      establishment_context_path=str(context_out))
    print(f"wrote {out}; rows: {len(hierarchy)}")


def cmd_audit_publishability(cfg: Config) -> None:
    result = run_publishability_audit(cfg)
    print(f"publishable: {result['publishable']}; failed gates: {result['failed_gate_count']}")
    for gate in result["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['name']}: {gate['detail']}")


def cmd_validate_external_reference(cfg: Config) -> None:
    root = Path(cfg.root).resolve()
    result = {"expert_event_gold": evaluate_expert_event_gold(root),
              "eventepi": evaluate_eventepi_reference(root),
              "cirad": evaluate_cirad_reference(root),
              "biocaster": evaluate_biocaster_reference(root),
              "band": evaluate_band_reference(root),
              "daniel": evaluate_daniel_reference(root)}
    io.write_json(result, cfg.evaluation_dir / "external_reference_metrics.json")
    print(result)


def cmd_build_county_outcomes(cfg: Config, args: argparse.Namespace) -> None:
    panel = _load_panel(cfg, args)
    crosswalk = load_city_county_crosswalk(cfg)
    outcomes = build_county_outcomes(panel, crosswalk)
    validate_county_outcomes(outcomes)
    out = cfg.artifact_path("outcomes/county_demand.csv.gz")
    io.write_csv(outcomes, out)
    mapped_rows = int(outcomes["source_row_count"].sum()) if len(outcomes) else 0
    io.write_metadata(cfg.artifact_path("outcomes/county_demand.json"), rows=len(outcomes),
                      mapped_panel_rows=mapped_rows, panel_rows=len(panel),
                      mapped_panel_row_fraction=mapped_rows / max(len(panel), 1),
                      source="CMS Part D city/provider demand")
    print(f"wrote {out}; rows: {len(outcomes)} mapped_panel_rows: {mapped_rows}/{len(panel)}")


def cmd_build_annotation_manifest(cfg: Config, args: argparse.Namespace) -> None:
    corpus = io.load_csv(cfg.artifact_path("news/historical_corpus.csv.gz"))
    manifest = build_annotation_manifest(corpus, n=args.count)
    out = cfg.artifact_path("annotations/news_gold_manifest.csv.gz")
    io.write_csv(manifest, out)
    io.write_metadata(cfg.artifact_path("annotations/news_gold_manifest.json"), rows=len(manifest),
                      labeled_rows=int(manifest["label_status"].eq("labeled").sum()) if len(manifest) else 0,
                      status="unlabeled_requires_expert_annotation")
    print(f"wrote {out}; rows: {len(manifest)}; all labels remain unlabeled")


def cmd_build_expert_event_gold(cfg: Config) -> None:
    frame = build_expert_event_gold(cfg.repo_root)
    out = cfg.artifact_path("events/expert_event_gold.csv.gz")
    io.write_csv(frame, out)
    metadata = expert_event_gold_metadata(frame)
    io.write_json(metadata, cfg.artifact_path("events/expert_event_gold.json"))
    io.write_json(evaluate_expert_event_gold(cfg.repo_root),
                  cfg.evaluation_dir / "expert_event_gold_metrics.json")
    print(f"wrote {out}; rows: {len(frame)}; unique articles: {metadata['unique_articles']}")


def cmd_evaluate_county_demand(cfg: Config, args: argparse.Namespace) -> None:
    outcomes = load_county_demand(cfg)
    results = evaluate_county_demand(outcomes, train_cutoff=args.train_cutoff)
    ldf = results.pop("leaderboard", None)
    results["leaderboard"] = (
        ldf.to_dict("records") if ldf is not None and not ldf.empty else None
    )
    out = cfg.evaluation_dir / "county_metrics.json"
    io.write_json(results, out)
    if ldf is not None and not ldf.empty:
        lboard = cfg.evaluation_dir / "county_leaderboard.csv"
        ldf.to_csv(lboard, index=False)
        print(f"wrote {lboard}")
    print(f"wrote {out}")

    print(f"rows: {results.get('n_rows')}")
    print(f"counties: {results.get('n_counties')}  drugs: {results.get('n_drugs')}")
    print(f"selected model: {results.get('selected_model')} "
          f"(validation wape {results.get('selected_validation_wape', float('nan')):.4f})")
    print(f"test improvement vs previous-year naive: "
          f"{results.get('test_improvement_vs_naive', float('nan')):.4f}")
    print(f"publishable_candidate: {results.get('publishable_candidate')}")
    print(results.get("publishability_reason", ""))
    if ldf is not None and not ldf.empty:
        show = ldf[["model", "family", "validation_wape", "wape", "mae", "rmse"]]
        print(show.to_string(index=False))

    rolling = evaluate_county_demand_rolling(
        outcomes, min_train_years=args.min_train_years)
    folds = rolling.pop("folds", None)
    rolling["folds"] = (
        folds.to_dict("records") if folds is not None and not folds.empty else None
    )
    rout = cfg.evaluation_dir / "county_rolling_metrics.json"
    io.write_json(rolling, rout)
    if folds is not None and not folds.empty:
        fpath = cfg.evaluation_dir / "county_rolling.csv"
        folds.to_csv(fpath, index=False)
        print(f"wrote {fpath}")
    print(f"wrote {rout}")
    print(f"rolling folds: {rolling.get('fold_count')}  "
          f"mean improvement: {rolling.get('mean_improvement_vs_naive', float('nan')):.4f}  "
          f"publishable_rolling_candidate: {rolling.get('publishable_rolling_candidate')}")
    print(rolling.get("publishability_reason", ""))

    regional = evaluate_county_demand_by_region(
        outcomes, min_train_years=args.min_train_years)
    regional_out = cfg.evaluation_dir / "county_region_rolling_metrics.json"
    io.write_json(regional, regional_out)
    print(f"wrote {regional_out}")


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    panel = _load_panel(cfg, args)
    view, spec = build_next_period(panel)
    feature_mode = ("full" if args.include_periodic_training_features
                    else "operational")
    full_cols = feature_columns_for_mode(view, feature_mode)
    excluded_periodic_feature_count = (
        len(ablation_features(view)["full"]) - len(full_cols)
    )
    feature_audit = summarize_dispositions(full_cols)
    X_all = _feature_matrix(view, full_cols)
    w_all = view["y_last"].to_numpy(dtype=float)

    def _fit_log(target: str, alpha: float = 10.0):
        y = view[target].to_numpy(dtype=float)
        return _fit_ridge_log(X_all, y, full_cols, alpha=alpha, sample_weight=w_all)

    claims_model = _fit_log("demand_claims_t1", alpha=0.01)
    cost_model = _fit_log("demand_cost_t1")

    # Shortage-risk logistic on the strict next-period label.
    label = (view["shortage_events_t1"].fillna(0) > 0).astype(float).to_numpy()
    risk_model = None
    if label.sum() >= 5:
        risk_model = LogisticRidge(alpha=10.0, max_iter=30).fit(X_all, label, full_cols)

    # Select the boolean operating point on a held-out final feature year;
    # this threshold is then stored with the refit full-data model.
    shortage_threshold = 0.5
    available_risk_years = set(view["year"].astype(int).unique())
    # Keep the saved operating point aligned with the strict evaluation
    # contract (train through 2020, validate on 2021) when that historical
    # period is available; otherwise use the latest complete held-out year.
    val_year_risk = (2021 if 2021 in available_risk_years
                     else max(available_risk_years))
    risk_fit = view[view["year"] < val_year_risk]
    risk_val = view[view["year"] == val_year_risk]
    if risk_model is not None and len(risk_fit) and len(risk_val):
        fit_x = _feature_matrix(risk_fit, full_cols)
        fit_y = (risk_fit["shortage_events_t1"].fillna(0) > 0).astype(float).to_numpy()
        val_y = (risk_val["shortage_events_t1"].fillna(0) > 0).astype(float).to_numpy()
        if fit_y.sum() >= 5 and val_y.sum() >= 1:
            fold_risk = LogisticRidge(alpha=10.0, max_iter=30).fit(fit_x, fit_y, full_cols)
            shortage_threshold = _select_binary_threshold(
                val_y, fold_risk.predict_proba(_feature_matrix(risk_val, full_cols)))

    # Neural blend (log target, demand-weighted deterministic subsample).
    idx = _neural_subsample(np.arange(len(view)), weights=w_all, cap=120_000)
    y_log = np.log1p(np.clip(view["demand_claims_t1"].to_numpy(dtype=float)[idx], 0, None))
    neural_model = NeuralSignalBlender(epochs=200).fit(
        X_all[idx], y_log.reshape(-1, 1), full_cols)

    # Ensemble weights from a strict last-year validation fold.
    val_year = int(view["year"].max())
    train_fold = view[view["year"] < val_year]
    val_fold = view[view["year"] == val_year]
    X_tr = _feature_matrix(train_fold, full_cols)
    X_va = _feature_matrix(val_fold, full_cols)
    m_fold = _fit_ridge_log(X_tr, train_fold["demand_claims_t1"].to_numpy(dtype=float),
                            full_cols, sample_weight=train_fold["y_last"].to_numpy(dtype=float))
    n_cap = min(len(X_tr), 120_000)
    y_fold = np.log1p(np.clip(
        train_fold["demand_claims_t1"].to_numpy(dtype=float)[:n_cap], 0, None))
    nm_fold = NeuralSignalBlender(epochs=120).fit(
        X_tr[:n_cap], y_fold.reshape(-1, 1), full_cols)
    comp_val = np.column_stack([
        np.log1p(np.clip(val_fold["y_last"].to_numpy(dtype=float), 0, None)),
        m_fold.predict(X_va),
        nm_fold.predict(X_va).reshape(-1),
    ])
    weights = _simplex_weights(
        comp_val, np.log1p(np.clip(val_fold["demand_claims_t1"].to_numpy(dtype=float), 0, None)))

    ridge_val_raw = np.expm1(claims_model.predict(X_va))
    y_val_raw = val_fold["demand_claims_t1"].to_numpy(dtype=float)
    last_val_raw = val_fold["y_last"].to_numpy(dtype=float)
    best_blend = 0.0
    best_blend_wape = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        pred = (1.0 - w) * last_val_raw + w * ridge_val_raw
        wape = float(np.sum(np.abs(y_val_raw - pred)) /
                     max(np.sum(np.abs(y_val_raw)), 1e-9))
        if wape < best_blend_wape:
            best_blend = float(w)
            best_blend_wape = wape

    trained = {
        "demand_claims": {
            "model": claims_model.to_dict(), "family": "ridge_linear",
            "target": "demand_claims_t1", "target_transform": "log1p",
            "feature_cols": full_cols,
            "n_params": int(claims_model.coef.size + 1),
        },
        "demand_cost": {
            "model": cost_model.to_dict(), "family": "ridge_linear",
            "target": "demand_cost_t1", "target_transform": "log1p",
            "feature_cols": full_cols,
            "n_params": int(cost_model.coef.size + 1),
        },
        "shortage_risk": {
            "model": risk_model.to_dict() if risk_model else None,
            "family": "logistic_ridge" if risk_model else "insufficient_labels",
            "feature_cols": full_cols,
            "shortage_threshold": shortage_threshold,
        },
        "neural": {
            "model": neural_model.to_dict(),
            "family": "neural_signal_blender_mlp",
            "n_params": neural_model.n_params,
        },
        "ensemble": {
            "weights": weights.tolist(),
            "members": ["city_drug_last_naive", "ridge_full", "neural_blend"],
            "validation_year": val_year,
            "n_params": int(3 + claims_model.coef.size + 1 + neural_model.n_params),
        },
        "calibrated_blend": {
            "members": ["city_drug_last", "ridge_full_alpha_0.01"],
            "blend_weight": best_blend,
            "validation_wape": best_blend_wape,
            "ridge_alpha": 0.01,
        },
        "encoder_spec": spec,
        "input_contract": feature_audit,
        "feature_mode": feature_mode,
        "excluded_periodic_feature_count": excluded_periodic_feature_count,
    }

    print(f"feature_mode: {feature_mode} "
          f"excluded_periodic_features: {excluded_periodic_feature_count}")
    print(f"trained demand_claims: family=ridge_linear n_features={len(full_cols)}")
    print(f"trained demand_cost: family=ridge_linear")
    print(f"trained shortage_risk: family={trained['shortage_risk']['family']}")
    print(f"trained neural: params={neural_model.n_params}")
    print(f"ensemble weights (naive, ridge, neural): {[round(w, 3) for w in weights]}")
    print(f"calibrated demand blend: ridge_weight={best_blend:.2f} "
          f"validation_wape={best_blend_wape:.3f}")

    if not args.no_save:
        out = cfg.trained_dir / "models.json"
        io.write_json(trained, out)
        io.write_metadata(
            cfg.metadata_dir / "trained.json",
            targets=["demand_claims_t1", "demand_cost_t1"],
            n_features=len(full_cols),
            view_rows=int(len(view)),
            neural_params=int(neural_model.n_params),
            ensemble_weights=weights.tolist(),
            calibrated_blend=trained["calibrated_blend"],
            input_contract={
                "operational_feature_count": feature_audit["operational_feature_count"],
                "periodic_training_only_count": feature_audit["periodic_training_only_count"],
                "derived_training_variable_count": feature_audit["derived_training_variable_count"],
                "static_identity_context_count": feature_audit["static_identity_context_count"],
                "operational_ready": feature_audit["operational_ready"],
            },
            feature_mode=feature_mode,
            excluded_periodic_feature_count=excluded_periodic_feature_count,
        )
        print(f"wrote {out}")


def cmd_forecast(cfg: Config, args: argparse.Namespace) -> None:
    panel = io.load_csv(cfg.panel_dir / "panel.csv")
    trained = io.read_json(cfg.trained_dir / "models.json")
    contract = forecast_input_contract(trained)
    _require_operational_forecast(
        contract, allow_training_only_features=args.allow_training_only_features)
    grid = build_forecast_grid(
        panel, trained, cfg=cfg,
        forecast_years=args.forecast_years,
        max_rows=args.max_rows,
    )
    out = cfg.forecasts_dir / "forecast.csv"
    io.write_csv(grid, out)
    io.write_metadata(
        cfg.metadata_dir / "forecast.json",
        rows=int(len(grid)),
        targets=sorted(grid["target"].unique().tolist()),
        horizons=sorted(grid["horizon_days"].unique().tolist()),
        drugs=int(grid["drug_key"].nunique()),
        cities=int(grid["geography_id"].nunique()),
        output_schema=list(grid.columns),
        input_contract={
            "operational_feature_count": contract["operational_feature_count"],
            "periodic_training_only_count": contract["periodic_training_only_count"],
            "derived_training_variable_count": contract["derived_training_variable_count"],
            "static_identity_context_count": contract["static_identity_context_count"],
            "operational_ready": contract["operational_ready"],
        },
        allow_training_only_features=bool(args.allow_training_only_features),
        forecast_mode=("research_training_only" if args.allow_training_only_features
                       and not contract["operational_ready"] else "operational"),
        operational_ready=contract["operational_ready"],
    )
    print(f"wrote {out}: {len(grid)} rows over {grid['drug_key'].nunique()} drugs "
          f"x {grid['geography_id'].nunique()} cities")


def _require_operational_forecast(
    contract: dict, *, allow_training_only_features: bool = False,
) -> None:
    """Reject non-operational models unless the research override is explicit."""
    if (not contract.get("operational_ready", False)
            and not allow_training_only_features):
        raise RuntimeError(
            "trained model contains periodic or historical-only features; "
            "retrain without --include-periodic-training-features or pass "
            "--allow-training-only-features for an explicitly non-operational forecast")


def _merge_live_sdud_target(
    panel: pd.DataFrame,
    urls: str | list[str] | None,
) -> tuple[pd.DataFrame, dict | None]:
    """Append opt-in CMS target refreshes without mutating the local cache.

    CMS publishes SDUD one year per downloadable file. Multiple URLs are
    accepted, while missing calendar quarters remain missing rather than being
    filled between files.
    """
    if not urls:
        return panel, None
    if isinstance(urls, str):
        urls = [urls]
    frames = []
    source_metadata = []
    for url in urls:
        live_panel, metadata = fetch_medicaid_sdud_csv(url=url)
        source_metadata.append(metadata)
        if not live_panel.empty:
            frames.append(live_panel)
    if not frames:
        return panel, {"source": "CMS State Drug Utilization Data",
                       "sources": source_metadata, "rows": 0}
    combined = pd.concat([panel, *frames], ignore_index=True)
    combined = (combined.groupby(["year", "quarter", "drug"], as_index=False)
                .agg(value=("value", "sum"),
                     ingredient=("ingredient", "first"),
                     ndc=("ndc", "first"),
                     rxnorm_rxcui=("rxnorm_rxcui", "first"),
                     manufacturer=("manufacturer", "first"),
                     mapping_confidence=("mapping_confidence", "first")))
    q_id = ((pd.to_numeric(combined["year"], errors="coerce") - 2012) * 4
            + pd.to_numeric(combined["quarter"], errors="coerce") - 1)
    latest = combined.loc[q_id.idxmax()]
    latest_period = f"{int(latest['year']):04d}-Q{int(latest['quarter'])}"
    period_end = pd.Period(
        f"{int(latest['year'])}Q{int(latest['quarter'])}", freq="Q"
    ).end_time.date()
    next_year = int(latest["year"]) + (int(latest["quarter"]) == 4)
    next_quarter = 1 if int(latest["quarter"]) == 4 else int(latest["quarter"]) + 1
    return combined, {
        "source": "CMS State Drug Utilization Data",
        "sources": source_metadata,
        "source_urls": [m.get("source_url") for m in source_metadata],
        "retrieved_on": max((m.get("retrieved_on", "") for m in source_metadata),
                             default=""),
        "rows": int(sum(m.get("rows", 0) for m in source_metadata)),
        "panel_period_end": latest_period,
        "panel_period_end_date": period_end.isoformat(),
        "panel_age_days": int((date.today() - period_end).days),
        "next_forecast_period": f"{next_year:04d}-Q{next_quarter}",
        "target_semantics": "quarterly Medicaid prescriptions by product name",
        "suppressed_rows_excluded": True,
    }


def cmd_forecast_quarterly(cfg: Config, args: argparse.Namespace) -> None:
    cache = cfg.panel_dir / "medicaid_sdud_quarterly_panel.csv"
    if cache.exists():
        panel = io.load_csv(cache, dtype={"ndc": "string"})
    else:
        combined = cfg.data_dir / "S_D" / "data" / "combined"
        panel = load_medicaid_sdud_panel(combined)
        io.write_csv(panel, cache)
    panel, live_sdud_metadata = _merge_live_sdud_target(
        panel, getattr(args, "live_sdud_url", None))
    grid = build_quarterly_forecast_grid(panel, max_rows=args.max_rows)
    out = cfg.forecasts_dir / "quarterly_forecast.csv"
    io.write_csv(grid, out)
    io.write_metadata(
        cfg.metadata_dir / "quarterly_forecast.json",
        rows=int(len(grid)),
        targets=sorted(grid["target"].unique().tolist()) if not grid.empty else [],
        horizons=sorted(grid["horizon_days"].unique().tolist()) if not grid.empty else [],
        drugs=int(grid["drug_key"].nunique()) if not grid.empty else 0,
        geography_level="state",
        output_schema=list(grid.columns),
        evidence_artifact=(
            "model/artifacts/evaluation/quarterly_rolling_live_target_metrics.json"
            if live_sdud_metadata is not None else
            "model/artifacts/evaluation/quarterly_rolling_1y_metrics.json"),
        target_refresh=live_sdud_metadata,
        panel_period_end=(live_sdud_metadata or {}).get("panel_period_end"),
        panel_period_end_date=(live_sdud_metadata or {}).get("panel_period_end_date"),
        panel_age_days=(live_sdud_metadata or {}).get("panel_age_days"),
        next_forecast_period=(live_sdud_metadata or {}).get("next_forecast_period"),
    )
    print(f"wrote {out}: {len(grid)} rows over "
          f"{grid['drug_key'].nunique() if not grid.empty else 0} Medicaid drugs")


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    panel = io.load_csv(cfg.panel_dir / "panel.csv")
    feature_mode = ("full" if args.include_periodic_training_features
                    else "operational")
    print(f"feature_mode: {feature_mode}")
    results = evaluate_next_period(panel, train_cutoff=args.train_cutoff,
                                   feature_mode=feature_mode,
                                   neural=not args.no_neural)
    ldf = results.pop("leaderboard", None)
    uplift = results.pop("layer_uplift", None)
    results["leaderboard"] = (
        ldf.to_dict("records") if ldf is not None and not ldf.empty else None
    )
    out = cfg.evaluation_dir / "metrics.json"
    io.write_json(results, out)
    if ldf is not None and not ldf.empty:
        lboard = cfg.evaluation_dir / "leaderboard.csv"
        ldf.to_csv(lboard, index=False)
        print(f"wrote {lboard}")
    if uplift is not None and not uplift.empty:
        up = cfg.evaluation_dir / "layer_uplift.csv"
        uplift.to_csv(up, index=False)
        print(f"wrote {up}")
    else:
        print("evaluation returned no leaderboard", file=sys.stderr)
    print(f"wrote {out}")

    print(f"best naive: {results.get('best_naive', {}).get('model')} "
          f"wape={results.get('best_naive', {}).get('wape', float('nan')):.3f}")
    print(f"best model: {results.get('best_model', {}).get('model')} "
          f"wape={results.get('best_model', {}).get('wape', float('nan')):.3f}")
    if ldf is not None and not ldf.empty:
        show = ldf[["model", "family", "wape", "mae", "rmse", "smape", "r2",
                    "dir_acc", "improvement_vs_best_naive"]].head(14)
        print(show.to_string(index=False))
    if "shortage_risk" in results:
        sr = results["shortage_risk"]
        summary = {
            "selected_model": sr.get("selected_model", sr.get("family")),
            "topk_recall": sr.get("topk_recall"),
            "precision_at_k": sr.get("precision_at_k"),
            "lift_at_k": sr.get("lift_at_k"),
            "positives_captured": sr.get("positives_captured"),
            "k": sr.get("k"),
            "publishable_candidate": sr.get("publishable_candidate"),
            "best_model_by_test": sr.get("best_model_by_test"),
        }
        print("shortage risk:", {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in summary.items()})
    if args.rolling_demand:
        rolling = evaluate_demand_rolling(
            panel,
            min_train_years=args.demand_min_train_years,
            test_window_years=args.rolling_demand_test_window_years,
            feature_mode=feature_mode,
        )
        folds = rolling.pop("folds", None)
        rolling["folds"] = (
            folds.to_dict("records") if folds is not None and not folds.empty else None
        )
        suffix = ("" if args.rolling_demand_test_window_years is None
                  else f"_{args.rolling_demand_test_window_years}y")
        out_roll = cfg.evaluation_dir / f"demand_rolling{suffix}_metrics.json"
        io.write_json(rolling, out_roll)
        if folds is not None and not folds.empty:
            fpath = cfg.evaluation_dir / f"demand_rolling{suffix}.csv"
            folds.to_csv(fpath, index=False)
            print(f"wrote {fpath}")
        print(f"wrote {out_roll}")
        print("rolling annual demand:", {
            "fold_count": rolling.get("fold_count"),
            "mean_improvement": round(
                rolling.get("mean_improvement_vs_best_naive", float("nan")), 4),
            "folds_beating_naive": rolling.get("folds_beating_naive"),
            "publishable_rolling_candidate": rolling.get(
                "publishable_rolling_candidate"),
        })

    if args.rolling_risk:
        rolling = evaluate_risk_rolling(
            panel, min_train_years=args.risk_min_train_years,
            feature_mode=feature_mode)
        folds = rolling.pop("folds", None)
        rolling["folds"] = (
            folds.to_dict("records") if folds is not None and not folds.empty else None
        )
        out_roll = cfg.evaluation_dir / "risk_rolling_metrics.json"
        io.write_json(rolling, out_roll)
        if folds is not None and not folds.empty:
            fpath = cfg.evaluation_dir / "risk_rolling.csv"
            folds.to_csv(fpath, index=False)
            print(f"wrote {fpath}")
        print(f"wrote {out_roll}")
        print("rolling shortage risk:", {
            "n_folds": rolling.get("n_folds"),
            "mean_topk_recall": round(rolling.get("mean_topk_recall", float("nan")), 4),
            "median_lift_at_k": round(rolling.get("median_lift_at_k", float("nan")), 4),
            "folds_beating_label_rate": rolling.get("folds_beating_label_rate"),
            "publishable_rolling_candidate": rolling.get("publishable_rolling_candidate"),
        })


def cmd_evaluate_quarterly(cfg: Config, args: argparse.Namespace) -> None:
    combined = cfg.data_dir / "S_D" / "data" / "combined"
    cache = cfg.panel_dir / "medicaid_sdud_quarterly_panel.csv"
    if cache.exists():
        panel = io.load_csv(cache, dtype={"ndc": "string"})
    else:
        panel = load_medicaid_sdud_panel(combined)
        io.write_csv(panel, cache)
    live_sdud_metadata = None
    panel, live_sdud_metadata = _merge_live_sdud_target(
        panel, args.live_sdud_url)
    annual_path = cfg.panel_dir / "panel.csv"
    events_path = cfg.data_dir / "final_data" / "events" / "events.csv.gz"
    news_events_path = cfg.artifact_path("events/article_events.csv.gz")
    arcos_path = (cfg.data_dir / "targeted_additions" / "dea_arcos_arkansas"
                  / "data" / "arcos_arkansas_retail_summary.csv.gz")
    fluview_path = (args.fluview_path
                    or cfg.data_dir / "targeted_additions"
                    / "disease_surveillance_current" / "data"
                    / "cdc_fluview_ar_national_weekly.csv.gz")
    wastewater_path = (args.wastewater_path
                       or cfg.data_dir / "targeted_additions"
                       / "disease_surveillance_current" / "data"
                       / "cdc_wastewater_ar_site_weekly.csv.gz")
    nadac_path = args.nadac_path or (
        cfg.data_dir / "demand_price" / "data" / "nadac_ndc_weekly.csv.gz")
    results = evaluate_quarterly(
        panel,
        train_cutoff=args.train_cutoff,
        annual_panel_path=annual_path,
        events_path=events_path,
        news_events_path=news_events_path,
        arcos_path=arcos_path,
        fluview_path=fluview_path,
        wastewater_path=wastewater_path,
        nadac_path=nadac_path,
    )
    if live_sdud_metadata is not None:
        results["live_target_refresh"] = live_sdud_metadata
    ldf = results.pop("leaderboard", None)
    results["leaderboard"] = (
        ldf.to_dict("records") if ldf is not None and not ldf.empty else None
    )
    artifact_suffix = "_live_target" if live_sdud_metadata is not None else ""
    out = cfg.evaluation_dir / f"quarterly_metrics{artifact_suffix}.json"
    io.write_json(results, out)
    if ldf is not None and not ldf.empty:
        lboard = cfg.evaluation_dir / f"quarterly_leaderboard{artifact_suffix}.csv"
        ldf.to_csv(lboard, index=False)
        print(f"wrote {lboard}")
    print(f"wrote {out}")

    print(f"rows: {results.get('n_rows')}")
    print(f"best naive: {results.get('best_naive')}")
    print(f"best model: {results.get('best_model')}")
    print(f"improvement vs previous_quarter: "
          f"{results.get('improvement_vs_previous_quarter', float('nan')):.4f}")
    print(f"publishable_candidate: {results.get('publishable_candidate')}")
    selected_name = results.get("selected_model", {}).get("model")
    selected_row = (ldf[ldf["model"].eq(selected_name)].iloc[0]
                    if ldf is not None and selected_name in set(ldf["model"]) else None)
    if selected_row is not None:
        print("selected pointwise accuracy:", {
            "within_5pct": float(selected_row.get("within_5pct", float("nan"))),
            "within_10pct": float(selected_row.get("within_10pct", float("nan"))),
            "within_20pct": float(selected_row.get("within_20pct", float("nan"))),
            "demand_state_accuracy": float(
                selected_row.get("demand_state_accuracy", float("nan"))),
            "demand_state_balanced_accuracy": float(
                selected_row.get("demand_state_balanced_accuracy", float("nan"))),
        })
    print(results.get("publishability_reason", ""))
    if ldf is not None and not ldf.empty:
        show = ldf[["model", "family", "wape", "mae", "rmse", "smape", "r2",
                    "improvement_vs_previous_quarter"]]
        print(show.to_string(index=False))
    if args.rolling:
        rolling = evaluate_quarterly_rolling(
            panel,
            annual_panel_path=annual_path,
            events_path=events_path,
            news_events_path=news_events_path,
            arcos_path=arcos_path,
            min_train_years=args.min_train_years,
            test_window_years=args.rolling_test_window_years,
            fluview_path=fluview_path,
            wastewater_path=wastewater_path,
            nadac_path=nadac_path,
        )
        folds = rolling.pop("folds", None)
        rolling["folds"] = (
            folds.to_dict("records") if folds is not None and not folds.empty
            else None
        )
        suffix = ("" if args.rolling_test_window_years is None
                  else f"_{args.rolling_test_window_years}y")
        rout = (cfg.evaluation_dir
                / f"quarterly_rolling{suffix}{artifact_suffix}_metrics.json")
        io.write_json(rolling, rout)
        if folds is not None and not folds.empty:
            fpath = (cfg.evaluation_dir
                     / f"quarterly_rolling{suffix}{artifact_suffix}.csv")
            folds.to_csv(fpath, index=False)
            print(f"wrote {fpath}")
        print(f"wrote {rout}")
        print(f"rolling_candidate: "
              f"{rolling.get('publishable_rolling_candidate')}")
        print(f"promotion_candidate: {rolling.get('promotion_candidate')} "
              f"requested_accuracy_goal_met: "
              f"{rolling.get('requested_accuracy_goal_met')}")
        print(rolling.get("promotion_reason", ""))
        print(rolling.get("publishability_reason", ""))


def cmd_evaluate_supplier_shortage(cfg: Config, args: argparse.Namespace) -> None:
    results = evaluate_supplier_shortage(cfg)
    ldf = results.pop("leaderboard", None)
    results["leaderboard"] = (
        ldf.to_dict("records") if ldf is not None and not ldf.empty else None
    )
    out = cfg.evaluation_dir / "supplier_shortage_metrics.json"
    io.write_json(results, out)
    if ldf is not None and not ldf.empty:
        lboard = cfg.evaluation_dir / "supplier_shortage_leaderboard.csv"
        ldf.to_csv(lboard, index=False)
        print(f"wrote {lboard}")
    print(f"wrote {out}")

    print(f"pairs: {results.get('n_pairs')}  suppliers: {results.get('n_suppliers')}  "
          f"drugs: {results.get('n_drugs')}")
    print(f"rows: {results.get('n_rows')}  events: {results.get('event_counts')}")
    print(f"selected model: {results.get('selected_model')} "
          f"(threshold {results.get('selected_threshold', float('nan')):.3f})")
    if ldf is not None and not ldf.empty:
        show = ldf[["model", "validation_f1", "validation_auprc",
                    "test_accuracy", "test_balanced_accuracy", "test_precision",
                    "test_recall", "test_f1", "test_auroc", "test_auprc",
                    "test_brier"]]
        print(show.to_string(index=False))

    rolling = evaluate_supplier_shortage_rolling(
        cfg, min_train_months=args.min_train_months,
        test_window_months=args.test_window_months)
    folds = rolling.pop("folds", None)
    rolling["folds"] = (
        folds.to_dict("records") if folds is not None and not folds.empty else None
    )
    rout = cfg.evaluation_dir / "supplier_shortage_rolling_metrics.json"
    io.write_json(rolling, rout)
    print(f"wrote {rout}")
    print(f"rolling folds: {rolling.get('fold_count')}")
    if folds is not None and not folds.empty:
        fpath = cfg.evaluation_dir / "supplier_shortage_rolling.csv"
        folds.to_csv(fpath, index=False)
        print(f"wrote {fpath}")


def cmd_evaluate_arcos_regional(cfg: Config, args: argparse.Namespace) -> None:
    panel = load_arcos(cfg.data_dir / SOURCE_RELATIVE)
    view = build_next_quarter_view(panel)
    result = evaluate_arcos_view(view, train_end_year=args.train_end_year)
    ldf = result.pop("leaderboard")
    result["leaderboard"] = ldf.to_dict("records")
    out = cfg.evaluation_dir / "arcos_regional_metrics.json"
    io.write_json(result, out)
    ldf.to_csv(cfg.evaluation_dir / "arcos_regional_leaderboard.csv", index=False)
    rolling = evaluate_arcos_rolling_view(
        view, min_train_quarters=args.rolling_min_train_quarters,
        test_window_quarters=args.rolling_test_window_quarters,
        validation_quarters=args.rolling_validation_quarters)
    folds = rolling.pop("folds", None)
    rolling["folds"] = (folds.to_dict("records")
                         if folds is not None and not folds.empty else [])
    rout = cfg.evaluation_dir / "arcos_regional_rolling_metrics.json"
    io.write_json(rolling, rout)
    if folds is not None and not folds.empty:
        folds.to_csv(cfg.evaluation_dir / "arcos_regional_rolling.csv", index=False)
    print(f"wrote {out}")
    print(f"wrote {rout}")
    print(f"rows: {result['n_rows']} zip3: {result['n_zip3']} drugs: {result['n_drugs']}")
    print(f"selected: {result['selected_model']} test_wape: {result['selected_test_wape']:.6f}")
    print(f"improvement_vs_best_naive: {result['improvement_vs_best_naive']:.3%}")
    print(f"publishable_candidate: {result['publishable_candidate']}")
    print(f"rolling folds: {rolling.get('fold_count', 0)}")
    print(f"rolling_candidate: {rolling.get('publishable_rolling_candidate', False)}")


def cmd_train_news_relevance(cfg: Config, args: argparse.Namespace) -> None:
    result = train_news_relevance(Path(cfg.root), train_end=args.train_end,
                                  validation_end=args.validation_end,
                                  dimension=args.feature_dimension)
    out = cfg.artifact_path("news/news_relevance_model.json")
    io.write_json(result, out)
    corpus_path = cfg.artifact_path("news/historical_corpus.csv.gz")
    if corpus_path.exists():
        scores = score_corpus(io.load_csv(corpus_path), result)
        score_out = cfg.artifact_path("news/relevance_scores.csv.gz")
        io.write_csv(scores, score_out)
        print(f"wrote {score_out}")
    print(f"wrote {out}")
    print(f"rows: {result['split']}")
    print(f"validation: {result['metrics']['validation']}")
    print(f"test: {result['metrics']['test']}")
    print(f"independent_cirad: {result['independent_cirad'].get('metrics')}")


def cmd_train_research_event_state(cfg: Config, args: argparse.Namespace) -> None:
    result = train_cirad_event_state(Path(cfg.root), dimension=args.feature_dimension)
    out = cfg.artifact_path("news/research_event_state_model.json")
    io.write_json(result, out)
    print(f"wrote {out}")
    print(f"split: {result['split']}")
    print(f"metrics: {result['metrics']}")


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cfg = _make_cfg(args)
    try:
        if args.command == "build-panel":
            cmd_build_panel(cfg, args)
        elif args.command == "train":
            cmd_train(cfg, args)
        elif args.command == "forecast":
            cmd_forecast(cfg, args)
        elif args.command == "forecast-quarterly":
            cmd_forecast_quarterly(cfg, args)
        elif args.command == "evaluate":
            cmd_evaluate(cfg, args)
        elif args.command == "evaluate-quarterly":
            cmd_evaluate_quarterly(cfg, args)
        elif args.command == "build-graph":
            cmd_build_graph(cfg)
        elif args.command == "build-news-corpus":
            cmd_build_corpus(cfg)
        elif args.command == "forecast-universal":
            cmd_forecast_universal(cfg, args)
        elif args.command == "refresh-weather":
            cmd_refresh_weather(cfg, args)
        elif args.command == "build-weather-geography":
            cmd_build_weather_geography(cfg, args)
        elif args.command == "refresh-county-weather":
            cmd_refresh_county_weather(cfg, args)
        elif args.command == "build-historical-weather":
            cmd_build_historical_weather(cfg, args)
        elif args.command == "build-geography":
            cmd_build_geography(cfg, args)
        elif args.command == "build-events":
            cmd_build_events(cfg)
        elif args.command == "build-news-signals":
            cmd_build_news_signals(cfg)
        elif args.command == "build-news-only-features":
            cmd_build_news_only_features(cfg, args)
        elif args.command == "build-supplier-hierarchy":
            cmd_build_supplier_hierarchy(cfg)
        elif args.command == "audit-publishability":
            cmd_audit_publishability(cfg)
        elif args.command == "validate-external-reference":
            cmd_validate_external_reference(cfg)
        elif args.command == "build-county-outcomes":
            cmd_build_county_outcomes(cfg, args)
        elif args.command == "evaluate-county-demand":
            cmd_evaluate_county_demand(cfg, args)
        elif args.command == "build-annotation-manifest":
            cmd_build_annotation_manifest(cfg, args)
        elif args.command == "build-expert-event-gold":
            cmd_build_expert_event_gold(cfg)
        elif args.command == "evaluate-supplier-shortage":
            cmd_evaluate_supplier_shortage(cfg, args)
        elif args.command == "evaluate-arcos-regional":
            cmd_evaluate_arcos_regional(cfg, args)
        elif args.command == "evaluate-weather-utility":
            cmd_evaluate_weather_utility(cfg, args)
        elif args.command == "evaluate-hospital-weather-utility":
            cmd_evaluate_hospital_weather_utility(cfg, args)
        elif args.command == "evaluate-hospital-disaster-utility":
            cmd_evaluate_hospital_disaster_utility(cfg, args)
        elif args.command == "train-news-relevance":
            cmd_train_news_relevance(cfg, args)
        elif args.command == "train-research-event-state":
            cmd_train_research_event_state(cfg, args)
        else:
            raise SystemExit(f"unknown command: {args.command}")
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
