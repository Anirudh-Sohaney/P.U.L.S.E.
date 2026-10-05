"""The tracked CI input is exactly the approved slice of the research audit."""

import hashlib
import json
from pathlib import Path

from scripts.build_metric_feature_store import build_feature_store_file


FIXTURE = Path(__file__).parent / "fixtures/metric_audit"


def test_checkout_clean_metric_fixture_builds_only_approved_partd_target(tmp_path):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    forecast = FIXTURE / "qualified_partd_forecasts.csv.gz"
    audit = FIXTURE / "metric_library_audit.json"
    for path, key in ((forecast, "fixture_forecast_sha256"),
                      (audit, "fixture_audit_sha256")):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]
    assert manifest["qualified_targets"] == [
        "arkansas_state_annual_partd_demand_five_state"]
    result = build_feature_store_file(forecast, tmp_path / "features.csv.gz",
                                      tmp_path / "features.json", audit)
    assert result["source_rows"] == result["eligible_rows"] == 1212
    assert result["output_rows"] == 1212
    assert result["feature_count"] == 1
    assert result["qualified_targets"] == manifest["qualified_targets"]
    assert result["excluded_targets"] == []
