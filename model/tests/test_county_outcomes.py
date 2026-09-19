import pandas as pd

from arkansas_pharma_signal.county_outcomes import build_county_outcomes, validate_county_outcomes


def test_county_outcomes_only_uses_mapped_rows():
    panel = pd.DataFrame([
        {"year": 2024, "city": "A", "drug_key": "x", "ingredient": "x", "labeler": "l",
         "demand_claims": 10, "demand_fills": 8, "demand_cost": 20},
        {"year": 2024, "city": "B", "drug_key": "x", "ingredient": "x", "labeler": "l",
         "demand_claims": 99, "demand_fills": 99, "demand_cost": 99},
    ])
    crosswalk = pd.DataFrame([{"city": "A", "county_fips": "05001", "county_name": "A",
                               "arkansas_region": "r", "confidence": 0.95}])
    out = build_county_outcomes(panel, crosswalk)
    validate_county_outcomes(out)
    assert len(out) == 1
    assert out.iloc[0]["demand_claims"] == 10


def test_county_outcomes_derives_region_from_fips_when_crosswalk_region_blank():
    panel = pd.DataFrame([{
        "year": 2024, "city": "A", "drug_key": "x", "ingredient": "x",
        "labeler": "l", "demand_claims": 10, "demand_fills": 8,
        "demand_cost": 20,
    }])
    crosswalk = pd.DataFrame([{
        "city": "A", "county_fips": "05001", "county_name": "Arkansas",
        "arkansas_region": "", "confidence": 0.95,
    }])
    out = build_county_outcomes(panel, crosswalk)
    assert out.iloc[0]["arkansas_region"] == "southeast"


if __name__ == "__main__":
    test_county_outcomes_only_uses_mapped_rows()
    print("ok")
