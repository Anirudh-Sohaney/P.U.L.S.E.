from pathlib import Path
import json

from arkansas_pharma_signal.project_status import audit_project_status


def test_project_status_does_not_conflate_proxy_gates_with_learned_completion():
    result = audit_project_status(
        Path("model/artifacts/evaluation"),
        Path("model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"))
    assert result["proxy_library_ready"] is False
    assert not any(check["name"] == "target_validity_contract" and not check["passed"]
                   for check in result["checks"])
    assert result["learned_architecture_ready"] is False
    assert result["project_complete"] is False
    assert "proxy_coverage_categories" in result["incomplete_reasons"]
    assert "operational_metric_surface" in result["incomplete_reasons"]
    assert "learned_end_to_end_accuracy_contract" in result["incomplete_reasons"]
    assert "research_metric_exhaustion" in result["incomplete_reasons"]
    assert "legacy_publishability_diagnostics" in result["diagnostic_failures"]
    assert result["qualified_metric_count"] == 1
    surface = next(check for check in result["checks"]
                   if check["name"] == "operational_metric_surface")
    assert "qualified_targets=1" in surface["detail"]
    assert "targets=14" in surface["detail"]


def test_missing_research_exhaustion_evidence_keeps_project_incomplete():
    evidence_path = Path("model/artifacts/evaluation/research_exhaustion.json")
    assert not evidence_path.exists()
    result = audit_project_status(
        Path("model/artifacts/evaluation"),
        Path("model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"))
    check = next(item for item in result["checks"]
                 if item["name"] == "research_metric_exhaustion")
    assert check["passed"] is False
    assert result["project_complete"] is False


def test_serialized_project_status_matches_current_metric_audit():
    status = json.loads(Path(
        "model/artifacts/evaluation/project_status.json").read_text())
    audit = json.loads(Path(
        "model/artifacts/evaluation/metric_library_audit.json").read_text())
    assert status["qualified_metric_count"] == audit["qualified_metric_count"]
    assert status["project_complete"] is False
    assert "research_metric_exhaustion" in status["incomplete_reasons"]
