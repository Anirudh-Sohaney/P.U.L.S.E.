"""Build a grain-preserving regression feature store from qualified forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from arkansas_pharma_signal.metric_feature_store import build_metric_feature_store


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_feature_store_file(
        forecast_path: Path, output_path: Path, metadata_path: Path,
        audit_path: Path, *, qualified_only: bool = False) -> dict:
    """Pivot only targets promoted by the current external metric audit."""
    if not forecast_path.exists():
        raise FileNotFoundError(f"qualified forecast file missing: {forecast_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"metric promotion audit missing: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("target_validity", {}).get("passed") is not True:
        raise ValueError("metric promotion audit has not passed target validity")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("metric promotion audit has no candidates")
    names = [row.get("metric") for row in candidates if isinstance(row, dict)]
    if (len(names) != len(candidates)
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("metric promotion audit has duplicate or malformed candidates")
    approved = {row["metric"] for row in candidates
                if row.get("status") == "qualified_proxy" and not row.get("reasons")}
    if not approved:
        raise ValueError("metric promotion audit has no qualified targets")
    rows = pd.read_csv(forecast_path, compression="infer", low_memory=False)
    source_rows = len(rows)
    targets = set(rows["target"].astype(str))
    unknown = sorted(targets - set(names))
    if unknown:
        raise ValueError(f"forecast targets are absent from metric audit: {unknown}")
    excluded = sorted(targets - approved)
    if excluded and not qualified_only:
        raise ValueError(f"forecast contains targets rejected by metric audit: {excluded}")
    if qualified_only:
        rows = rows[rows["target"].isin(approved)].copy()
    if rows.empty:
        raise ValueError("forecast contains no audit-qualified rows")
    store, metadata = build_metric_feature_store(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.to_csv(output_path, index=False, compression="infer")
    result = {
        **metadata,
        "protocol": "qualified_metric_feature_store_v1",
        "source_path": str(forecast_path),
        "source_sha256": _sha256(forecast_path),
        "source_rows": int(source_rows),
        "eligible_rows": int(len(rows)),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "audit_protocol": audit.get("protocol"),
        "qualified_targets": sorted(set(rows["target"])),
        "excluded_targets": excluded,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "output_rows": int(len(store)),
        "output_columns": list(store.columns),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=Path, default=Path(
        "model/artifacts/forecasts/qualified_metric_forecasts.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path(
        "model/artifacts/forecasts/qualified_metric_feature_store.csv.gz"))
    parser.add_argument("--metadata", type=Path, default=Path(
        "model/artifacts/forecasts/qualified_metric_feature_store.json"))
    parser.add_argument("--audit", type=Path, default=Path(
        "model/artifacts/evaluation/metric_library_audit.json"))
    parser.add_argument("--qualified-only", action="store_true",
                        help="Explicitly omit forecast targets rejected by the current audit")
    args = parser.parse_args()
    result = build_feature_store_file(args.forecast, args.output, args.metadata,
                                      args.audit, qualified_only=args.qualified_only)
    print(json.dumps({
        "source_rows": result["source_rows"],
        "eligible_rows": result["eligible_rows"],
        "output_rows": result["output_rows"],
        "feature_count": result["feature_count"],
        "keys": result["keys"],
        "excluded_targets": result["excluded_targets"],
    }, indent=2))


if __name__ == "__main__":
    main()
