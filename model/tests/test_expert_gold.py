from pathlib import Path

from arkansas_pharma_signal.expert_gold import (
    build_expert_event_gold,
    normalize_padi_label,
)
from arkansas_pharma_signal.external_validation import evaluate_expert_event_gold


def test_padi_labels_preserve_event_semantics():
    assert normalize_padi_label("Relevant - Event") == ("current_or_risk_event", True)
    assert normalize_padi_label("Relevant (general information)") == (
        "general_information", False)
    assert normalize_padi_label("Irrelevant") == ("irrelevant", False)


def test_external_expert_gold_has_explicit_transfer_unit_types():
    root = Path(__file__).resolve().parents[2]
    frame = build_expert_event_gold(root)
    assert len(frame) >= 2_000
    assert set(frame["unit_type"]) == {"article_event", "sentence_event"}
    assert frame["unit_id"].is_unique
    assert frame["event_positive"].astype(bool).any()
    assert frame["annotation_provenance"].str.contains("animal-health").all()


def test_expert_gold_evaluation_is_scoped_and_reproducible():
    root = Path(__file__).resolve().parents[2]
    result = evaluate_expert_event_gold(root)
    assert result["available"] is True
    assert result["rows"] == 2_670
    assert result["training_use"] == "evaluation_only"
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
