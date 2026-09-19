from pathlib import Path

from arkansas_pharma_signal.learned_event_state import (
    event_state_probability, load_cirad_event_sentences, train_cirad_event_state,
)


def test_cirad_loader_has_article_grouped_labels():
    root = Path(__file__).resolve().parents[2]
    rows = load_cirad_event_sentences(root)
    assert len(rows) >= 1000
    assert len({row["article_id"] for row in rows}) >= 50
    assert {row["label"] for row in rows} >= {"CE", "RE"}


def test_event_state_training_is_research_only_and_bounded():
    root = Path(__file__).resolve().parents[2]
    artifact = train_cirad_event_state(root, dimension=32)
    assert artifact["training_use"] == "research_only_animal_health_transfer"
    assert artifact["split"]["train_articles"] > 0
    assert artifact["split"]["test_articles"] > 0
    assert artifact["split"]["train_articles"] + artifact["split"]["test_articles"] == artifact["split"]["article_count"]
    assert 0.0 <= event_state_probability("confirmed respiratory outbreak", artifact) <= 1.0
    assert event_state_probability("", artifact) == 0.5

