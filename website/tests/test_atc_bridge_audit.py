"""The SDUD bridge audit counts source rows once and never fills suppression."""

import gzip
import hashlib
import json

import pytest

from backend import signal_store
from backend.config import settings
from scripts.audit_atc_bridge import audit, load_mapping


def test_sdud_atc_coverage_preserves_unmapped_and_suppressed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PATH", str(tmp_path / "data"))
    mapping_path = tmp_path / "mapping.csv.gz"
    mapping_path.write_bytes(gzip.compress(
        b"ndc11,class_id\n00000000001,A02AA\n00000000001,C10AA\n00000000002,N06AA\n"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sha256": {"mapping_csv":
        hashlib.sha256(mapping_path.read_bytes()).hexdigest()}}))
    signal_store.initialize()
    with signal_store.connect() as db:
        db.execute("""INSERT INTO sdud_snapshots
            (content_hash, source_year, latest_quarter, retrieved_at, last_seen_at,
             source_url, row_count, suppressed_rows, reported_prescriptions)
            VALUES ('digest', 2026, 1, '2026-07-01T00:00:00+00:00',
                    '2026-10-01T00:00:00+00:00', 'https://data.medicaid.gov/example',
                    4, 1, 30)""")
        db.executemany("""INSERT INTO sdud_rows
            (content_hash, year, quarter, utilization_type, ndc,
             product_name, prescriptions, suppressed) VALUES
            ('digest', 2026, 1, ?, ?, ?, ?, ?)""", [
                ("FFSU", "00000000001", "A", 10, 0),
                ("MCOU", "00000000001", "A", 5, 0),
                ("FFSU", "00000000002", "B", None, 1),
                ("FFSU", "00000000003", "C", 15, 0),
            ])
    result = audit(mapping_path, manifest_path)
    assert result["sdud_rows"] == 4
    assert result["sdud_suppressed_rows"] == 1
    assert result["sdud_unique_ndcs"] == 3
    assert result["mapped_unique_ndcs"] == 2
    assert result["mapped_rows"] == 3
    assert result["reported_prescriptions_lower_bound"] == 30
    assert result["mapped_reported_prescriptions_lower_bound"] == 15
    assert result["qualified_atc_groups_represented"] == ["A02", "C10", "N06"]
    assert result["usable_as_monthly_hhs_model_input"] is False
    mapping_path.write_bytes(mapping_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="do not match"):
        load_mapping(mapping_path, manifest_path)
