import json
from pathlib import Path


def test_publishable_manifest_declares_target_qualification():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "data/targeted_additions/publishable_test_dataset/manifest.json")
        .read_text()
    )
    tables = manifest["tables"]
    assert len(tables) == 4
    assert "arcos_zip3_drug_next_quarter_test.csv.gz" in tables
    assert tables["arcos_zip3_drug_next_quarter_test.csv.gz"][
        "target_qualification"]["promotion_status"] == "qualified_distribution_proxy"
    for table in tables.values():
        qualification = table["target_qualification"]
        assert qualification["cadence"] in {"quarterly", "annual", "monthly"}
        assert qualification["geography"]
        assert qualification["supplier_resolution"]
        assert qualification["direct_observation"] is True
        assert qualification["promotion_status"]


def test_publishable_manifest_declares_authoritative_region_partition():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "data/targeted_additions/publishable_test_dataset/manifest.json")
        .read_text()
    )
    geography = manifest["geography_definition"]
    assert geography["county_count"] == 75
    assert geography["regions"] == [
        "northwest", "northeast", "central", "southwest", "southeast"
    ]
    assert geography["partition_policy"].endswith("exactly once")
