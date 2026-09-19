import pandas as pd

from arkansas_pharma_signal.news_signals import (
    DISEASE_SIGNAL_COLUMNS, NEWS_RELEVANCE_COLUMNS, NEWS_SIGNAL_COLUMNS,
    build_news_state_features,
)
from arkansas_pharma_signal.features import build_layer1_news_annual


def _event(**overrides):
    row = {
        "event_id": "e1", "article_id": "a1", "event_type": "shortage",
        "event_subtype": "shortage", "event_start": "2024-01-01",
        "event_end": "", "location": "Little Rock", "county_fips": "05031",
        "disease": "", "variant": "", "drug": "metformin", "therapeutic_class": "",
        "supplier": "", "parent_company": "", "factory": "", "api_source": "",
        "supply_impact": "supply_negative", "demand_impact": "demand_positive",
        "severity": 0.5, "probability": 0.5, "lead_time_days": "",
        "evidence_span": "shortage", "negation_status": "affirmed",
        "uncertainty_status": "uncertain", "causal_status": "unknown", "source": "local",
        "source_timestamp": "2024-01-01T00:00:00Z", "effective_date": "2024-01-01",
        "confidence": 0.7, "evidence_type": "article_span_rule",
    }
    row.update(overrides)
    return row


def test_layer_one_emits_bounded_state_variables_and_respects_time_barrier():
    events = pd.DataFrame([_event(), _event(event_id="e2", article_id="a2",
                                             source_timestamp="2025-01-01T00:00:00Z")])
    result = build_news_state_features(events, as_of="2024-06-01")
    assert len(NEWS_SIGNAL_COLUMNS) == 50
    assert len(result.columns) == (2 + len(NEWS_SIGNAL_COLUMNS)
                                   + len(DISEASE_SIGNAL_COLUMNS)
                                   + len(NEWS_RELEVANCE_COLUMNS))
    assert list(result.columns[:52]) == ["observation_date", "geography_key", *NEWS_SIGNAL_COLUMNS]
    assert len(result) == 1
    row = result.iloc[0]
    assert row["geography_key"] == "05031"
    assert row["news_shortage_present"] == 1
    assert row["news_shortage_count"] == 1
    assert row["news_shortage_uncertain_count"] == 1
    assert row["news_shortage_supply_negative_count"] == 1
    assert row["news_shortage_drug_mention_count"] == 1
    assert row["news_recall_present"] == 0


def test_layer_one_emits_disease_specific_outbreak_states():
    events = pd.DataFrame([_event(
        event_type="disease_outbreak", event_subtype="disease_outbreak",
        disease="influenza", evidence_span="influenza outbreak",
    )])
    result = build_news_state_features(events)
    row = result.iloc[0]
    assert row["news_disease_influenza_present"] == 1
    assert row["news_disease_influenza_outbreak_risk"] == 0.5
    assert row["news_disease_influenza_spread_rate"] == 1.0


def test_layer_one_states_are_aggregated_into_next_period_feature_year():
    events = pd.DataFrame([_event(
        event_type="disease_outbreak", event_subtype="disease_outbreak",
        disease="influenza", evidence_span="influenza outbreak",
    )])
    annual = build_layer1_news_annual(events)
    assert list(annual["year"]) == [2024]
    assert annual.loc[0, "news_disease_influenza_present"] == 1
    assert annual.loc[0, "news_disease_influenza_outbreak_risk"] == 0.5


def test_learned_relevance_score_is_provenance_joined():
    events = pd.DataFrame([_event(article_id="a1")])
    scores = pd.DataFrame([{"article_id": "a1", "relevance_probability": 0.8}])
    result = build_news_state_features(events, relevance_scores=scores)
    assert result.iloc[0]["news_relevance_mean"] == 0.8
    assert result.iloc[0]["news_relevance_scored_count"] == 1
