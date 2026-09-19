"""Evidence-based project completion and promotion status audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _check(name: str, passed: bool, detail: str, *, blocking: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "blocking": bool(blocking),
            "detail": detail}


def audit_project_status(evaluation_dir: Path, forecast_path: Path) -> dict[str, Any]:
    """Audit proxy readiness separately from learned architecture completion."""
    metric = _read(evaluation_dir / "metric_library_audit.json")
    whole = _read(evaluation_dir / "whole_system_metrics.json")
    publishability = _read(evaluation_dir / "publishability_audit.json")
    exhaustion = _read(evaluation_dir / "research_exhaustion.json")
    qualified = [row for row in metric.get("candidates", [])
                 if row.get("status") == "qualified_proxy"]
    coverage = metric.get("coverage_gates", {})
    target_validity = metric.get("target_validity", {})
    layered = whole.get("layered_demand", {})

    operational_rows = 0
    operational_ok = False
    operational_detail = "forecast file missing"
    if forecast_path.is_file():
        # The serialized JSON-bearing attribution column can make the C parser
        # mis-handle mixed-width chunks; the canonical feature-store reader
        # uses the Python parser-compatible full-row contract.
        frame = pd.read_csv(forecast_path, engine="python", usecols=[
            "forecast_period", "target", "geography_level", "geography_id",
            "county_fips", "drug_key", "therapeutic_class", "labeler", "supplier",
            "target_promotion_status", "uncertainty_status", "calibration_status",
        ])
        operational_rows = int(len(frame))
        keys = ["forecast_period", "target", "geography_level", "geography_id",
                "county_fips", "drug_key", "therapeutic_class", "labeler", "supplier"]
        operational_ok = bool(
            operational_rows > 0
            and frame.duplicated(keys).sum() == 0
            and frame["target_promotion_status"].astype(str).str.startswith("qualified_").all()
            and frame["uncertainty_status"].eq("not_estimated").all()
            and frame["calibration_status"].eq("not_calibrated").all())
        operational_detail = (
            f"rows={operational_rows}; duplicate_keys={int(frame.duplicated(keys).sum())}; "
            f"targets={frame['target'].nunique()}")

    proxy_coverage = all(bool(item.get("passed")) for item in coverage.values())
    proxy_accuracy = bool(qualified) and all(
        not row.get("reasons") for row in qualified)
    learned_raw_contract = bool(
        layered.get("contract_75pct_numeric_accuracy")
        or layered.get("contract_75pct_state_accuracy"))
    learned_event_contract = bool(layered.get("contract_80pct_true_positive"))
    learned_contract = bool(learned_raw_contract or learned_event_contract)
    checks = [
        _check("qualified_proxy_library", bool(qualified),
               f"qualified_metrics={len(qualified)}"),
        _check("qualified_proxy_accuracy_floor", proxy_accuracy,
               "all promoted proxy candidates have no audit rejection reasons"),
        _check("proxy_coverage_categories", proxy_coverage,
               "geography, drug, supplier, and disease/symptom coverage gates"),
        _check("target_validity_contract", bool(target_validity.get("passed")),
               "qualified targets must remain explicitly external proxies, with complete provenance"),
        _check("operational_metric_surface", operational_ok, operational_detail),
        _check("learned_end_to_end_accuracy_contract", learned_contract,
               f"numeric={layered.get('contract_75pct_numeric_accuracy')}; "
               f"state={layered.get('contract_75pct_state_accuracy')}; "
               f"event_route={layered.get('contract_80pct_true_positive')}; "
               f"within5={layered.get('mean_stacked_blend_within_5pct')}; "
               f"state_exact={layered.get('mean_test_state_exact_accuracy')}"),
        _check("research_metric_exhaustion", bool(exhaustion.get("complete"))
               and exhaustion.get("status") == "complete",
               f"status={exhaustion.get('status')}; reviewed_candidates="
               f"{exhaustion.get('screened_candidate_count_at_review')}; "
               f"remaining_questions={len(exhaustion.get('remaining_research_questions', []))}"),
        _check("legacy_publishability_diagnostics", bool(publishability.get("publishable")),
               f"failed_gate_count={publishability.get('failed_gate_count')}; "
               "retained for compatibility diagnostics, not an active completion gate",
               blocking=False),
    ]
    proxy_ready = bool(proxy_accuracy and proxy_coverage and operational_ok)
    complete = bool(proxy_ready and learned_contract and exhaustion.get("complete")
                    and exhaustion.get("status") == "complete")
    reasons = [check["name"] for check in checks
               if not check["passed"] and check.get("blocking", True)]
    diagnostics = [check["name"] for check in checks
                   if not check["passed"] and not check.get("blocking", True)]
    return {
        "protocol": "medi_track_project_status_audit_v1",
        "proxy_library_ready": proxy_ready,
        "learned_architecture_ready": learned_contract,
        "project_complete": complete,
        "qualified_metric_count": len(qualified),
        "checks": checks,
        "incomplete_reasons": reasons,
        "diagnostic_failures": diagnostics,
        "scope_note": "qualified proxies are not direct Arkansas pharmacy inventory observations",
    }
