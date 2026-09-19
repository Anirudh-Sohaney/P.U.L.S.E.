import pandas as pd

from arkansas_pharma_signal.supplier_hierarchy import (
    ESTABLISHMENT_CONTEXT_COLUMNS, SUPPLIER_COLUMNS, validate_supplier_hierarchy,
)


def test_supplier_hierarchy_contract_preserves_missing_tiers():
    frame = pd.DataFrame([{
        "supplier_key": "labeler:example", "labeler": "Example", "parent_company": "",
        "factory": "", "api_source": "", "supplier_country": "", "drug_key": "x",
        "ndc": "123", "source": "fda", "source_timestamp": "2026-01-01",
        "effective_date": "2020-01-01", "confidence": 0.95, "evidence_type": "direct",
        "coverage_status": "labeler_only_parent_factory_api_missing",
    }])
    validate_supplier_hierarchy(frame)
    assert list(frame.columns) == SUPPLIER_COLUMNS
    assert frame["parent_company"].eq("").all()


def test_supplier_rows_never_claim_unmatched_factory():
    frame = pd.DataFrame([{
        "supplier_key": "labeler:x", "labeler": "X", "parent_company": "",
        "factory": "", "api_source": "", "supplier_country": "", "drug_key": "x",
        "ndc": "1", "source": "fda", "source_timestamp": "", "effective_date": "",
        "confidence": 0.95, "evidence_type": "direct_labeler_only",
        "coverage_status": "labeler_only_parent_factory_api_missing",
    }])
    validate_supplier_hierarchy(frame)
    assert frame.loc[0, "factory"] == ""


def test_labeler_only_rows_reject_unsupported_hierarchy_edges():
    frame = pd.DataFrame([{
        "supplier_key": "labeler:x", "labeler": "X", "parent_company": "Unproven Parent",
        "factory": "Unproven Factory", "api_source": "", "supplier_country": "", "drug_key": "x",
        "ndc": "1", "source": "fda", "source_timestamp": "", "effective_date": "",
        "confidence": 0.95, "evidence_type": "direct_labeler_only",
        "coverage_status": "labeler_only_parent_factory_api_missing",
    }])
    try:
        validate_supplier_hierarchy(frame)
    except ValueError as exc:
        assert "labeler-only" in str(exc)
    else:
        raise AssertionError("unsupported hierarchy edge was accepted")


def test_establishment_match_is_not_labeled_as_direct_labeler_only():
    frame = pd.DataFrame([{
        "supplier_key": "labeler:x", "labeler": "X", "parent_company": "",
        "factory": "X Factory", "api_source": "", "supplier_country": "USA",
        "drug_key": "x", "ndc": "1", "source": "FDA DRLS",
        "source_timestamp": "", "effective_date": "", "confidence": 0.82,
        "evidence_type": "labeler_establishment_name_match",
        "coverage_status": "establishment_match_not_product_specific",
    }], columns=SUPPLIER_COLUMNS)
    validate_supplier_hierarchy(frame)


def test_establishment_context_contract_is_supplier_level_only():
    assert "drug_key" not in ESTABLISHMENT_CONTEXT_COLUMNS
    assert "ndc" not in ESTABLISHMENT_CONTEXT_COLUMNS
if __name__ == "__main__":
    test_supplier_hierarchy_contract_preserves_missing_tiers()
    print("ok")
