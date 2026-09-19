from pathlib import Path
import json

from arkansas_pharma_signal.project_status import audit_project_status


def test_project_status_does_not_conflate_proxy_gates_with_learned_completion():
    result = audit_project_status(
        Path("model/artifacts/evaluation"),
        Path("model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"))
    assert result["proxy_library_ready"] is True
    assert not any(check["name"] == "target_validity_contract" and not check["passed"]
                   for check in result["checks"])
    assert result["learned_architecture_ready"] is False
    assert result["project_complete"] is False
    assert "learned_end_to_end_accuracy_contract" in result["incomplete_reasons"]
    assert "research_metric_exhaustion" in result["incomplete_reasons"]
    assert "legacy_publishability_diagnostics" in result["diagnostic_failures"]
    assert result["qualified_metric_count"] == 14


def test_research_exhaustion_records_recent_public_source_screening():
    evidence = json.loads(Path(
        "model/artifacts/evaluation/research_exhaustion.json").read_text())
    assert evidence["status"] == "incomplete"
    assert evidence["complete"] is False
    assert evidence["screened_candidate_count_at_review"] == 63
    assert {item["decision"] for item in evidence["latest_screened_sources"]} == {"rejected", "qualified"}
    assert any(item["local_download"] is True
               for item in evidence["latest_screened_sources"])


def test_serialized_project_status_matches_current_metric_audit():
    status = json.loads(Path(
        "model/artifacts/evaluation/project_status.json").read_text())
    audit = json.loads(Path(
        "model/artifacts/evaluation/metric_library_audit.json").read_text())
    assert status["qualified_metric_count"] == audit["qualified_metric_count"]
    assert status["project_complete"] is False
    assert "research_metric_exhaustion" in status["incomplete_reasons"]
