import pandas as pd

from arkansas_pharma_signal.cms_partd_quarterly import (
    add_latest_context_layer,
    load_cms_partd_snapshot,
    snapshot_metadata,
)


def test_cms_snapshot_keeps_overall_rows_and_coverage(tmp_path):
    path = tmp_path / "cms.csv"
    pd.DataFrame([
        {"Brnd_Name": "A", "Gnrc_Name": "alpha", "Tot_Mftr": 2,
         "Mftr_Name": "Overall", "Year": "2025 (Q1-Q3)", "Tot_Benes": 4,
         "Tot_Clms": 8, "Tot_Spndng": "12.50"},
        {"Brnd_Name": "A", "Gnrc_Name": "alpha", "Tot_Mftr": 2,
         "Mftr_Name": "Supplier", "Year": "2025 (Q1-Q3)", "Tot_Benes": 4,
         "Tot_Clms": 8, "Tot_Spndng": "12.50"},
    ]).to_csv(path, index=False)
    out = load_cms_partd_snapshot(path)
    assert len(out) == 1
    assert out.iloc[0]["coverage_end_quarter"] == 3
    assert snapshot_metadata(path)["target_use"].startswith("context input")


def test_cms_snapshot_missing_file_is_explicitly_unavailable(tmp_path):
    assert load_cms_partd_snapshot(tmp_path / "missing.csv").empty
    assert snapshot_metadata(tmp_path / "missing.csv") == {
        "available": False,
        "rows": 0,
    }


def test_cms_context_only_matches_snapshot_endpoint(tmp_path):
    path = tmp_path / "cms.csv"
    pd.DataFrame([{
        "Brnd_Name": "A", "Gnrc_Name": "alpha", "Tot_Mftr": 1,
        "Mftr_Name": "Overall", "Year": "2025 (Q1-Q3)", "Tot_Benes": 4,
        "Tot_Clms": 8, "Tot_Spndng": 12.5,
    }]).to_csv(path, index=False)
    view = pd.DataFrame([
        {"year": 2025, "quarter": 3, "drug": "alpha", "ingredient": "alpha"},
        {"year": 2025, "quarter": 2, "drug": "alpha", "ingredient": "alpha"},
    ])
    out, cols = add_latest_context_layer(view, path)
    assert cols == ["exo_cms_partd_claims", "exo_cms_partd_spending",
                    "exo_cms_partd_beneficiaries"]
    assert out.loc[0, "exo_cms_partd_claims"] == 8
    assert pd.isna(out.loc[1, "exo_cms_partd_claims"])
