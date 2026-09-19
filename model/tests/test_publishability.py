from pathlib import Path

from arkansas_pharma_signal.config import Config
from arkansas_pharma_signal.publishability import run_publishability_audit


def test_publishability_audit_fails_closed_for_missing_gold_outcomes():
    # Uses the real repo inputs/artifacts when run in the project; the key
    # contract is that missing gold/outcome gates cannot become publishable.
    root = Path(__file__).resolve().parents[2]
    result = run_publishability_audit(Config(root=str(root)))
    assert result["publishable"] is False
    names = {x["name"] for x in result["gates"] if not x["passed"]}
    assert "expert_event_gold_set" not in names
    assert {"rolling_origin_forecast_gate", "modular_rolling_gate"} <= names
    assert "county_level_outcome_labels" not in names
    assert "arcos_regional_pointwise_evaluation" in {
        x["name"] for x in result["gates"]}
    assert "learned_news_relevance_layer" in {
        x["name"] for x in result["gates"]}
    assert "modular_rolling_gate" in {x["name"] for x in result["gates"]}


if __name__ == "__main__":
    test_publishability_audit_fails_closed_for_missing_gold_outcomes()
    print("ok")
