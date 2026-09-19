"""Tests for data processing modules."""

import numpy as np
import pandas as pd
import pytest

from ml.data import DataCleaner, DataIngester, DataValidator
from ml.data.schema import PHARMACY_RISK_SCHEMA, DataType


@pytest.fixture
def valid_df():
    """Create a valid DataFrame for testing."""
    return pd.DataFrame(
        {
            "medication_id": ["MED001", "MED002", "MED003"],
            "medication_name": ["Aspirin", "Ibuprofen", "Acetaminophen"],
            "medication_category": ["analgesic", "analgesic", "analgesic"],
            "current_inventory": [100.0, 200.0, 150.0],
            "reorder_point": [50.0, 80.0, 60.0],
            "max_capacity": [500.0, 400.0, 300.0],
            "prescription_volume_30d": [30.0, 45.0, 35.0],
            "avg_weekly_demand": [10.0, 15.0, 12.0],
            "demand_trend": [0.1, -0.05, 0.2],
            "supplier_id": ["SUP001", "SUP002", "SUP001"],
            "supplier_lead_time_days": [7.0, 10.0, 5.0],
            "supplier_reliability_score": [0.9, 0.8, 0.95],
            "historical_shortage_count": [2, 1, 0],
            "days_since_last_shortage": [30.0, 60.0, 90.0],
            "avg_stockout_duration_days": [2.0, 1.0, 0.0],
            "risk_label": [0, 1, 0],
        }
    )


@pytest.fixture
def invalid_df():
    """Create an invalid DataFrame for testing."""
    return pd.DataFrame(
        {
            "medication_id": ["MED001", "MED002"],
            "medication_name": ["Aspirin", "Ibuprofen"],
            "medication_category": ["analgesic", "invalid_category"],
            "current_inventory": [-10.0, 200.0],
            "reorder_point": [50.0, 80.0],
            "max_capacity": [500.0, 400.0],
            "prescription_volume_30d": [30.0, 45.0],
            "avg_weekly_demand": [10.0, 15.0],
            "supplier_id": ["SUP001", "SUP002"],
            "supplier_lead_time_days": [7.0, 10.0],
            "supplier_reliability_score": [0.9, 1.5],
            "historical_shortage_count": [2, 1],
        }
    )


class TestDataValidator:
    """Tests for DataValidator."""

    def test_valid_data_passes(self, valid_df):
        validator = DataValidator()
        result = validator.validate(valid_df)
        assert result.is_valid
        assert result.rows_passed == 3
        assert len(result.errors) == 0

    def test_missing_required_column_fails(self):
        df = pd.DataFrame({"medication_id": ["MED001"]})
        validator = DataValidator()
        result = validator.validate(df)
        assert not result.is_valid
        assert any("Missing required column" in e for e in result.errors)

    def test_invalid_category_fails(self, invalid_df):
        validator = DataValidator()
        result = validator.validate(invalid_df)
        assert not result.is_valid
        assert any("invalid values" in e for e in result.errors)

    def test_negative_inventory_fails(self, invalid_df):
        validator = DataValidator()
        result = validator.validate(invalid_df)
        assert not result.is_valid
        assert any("below minimum" in e for e in result.errors)

    def test_out_of_range_reliability_fails(self, invalid_df):
        validator = DataValidator()
        result = validator.validate(invalid_df)
        assert not result.is_valid
        assert any("above maximum" in e for e in result.errors)

    def test_duplicates_detected(self, valid_df):
        df = pd.concat([valid_df, valid_df.iloc[[0]]])
        validator = DataValidator()
        result = validator.validate(df)
        assert result.is_valid
        assert any("duplicate" in w.lower() for w in result.warnings)


class TestDataCleaner:
    """Tests for DataCleaner."""

    def test_fit_transform(self, valid_df):
        cleaner = DataCleaner()
        result = cleaner.fit_transform(valid_df)
        assert len(result) == len(valid_df)
        assert list(result.columns) == list(valid_df.columns)

    def test_handles_missing_values(self):
        df = pd.DataFrame(
            {
                "medication_id": ["MED001", "MED002"],
                "medication_name": ["Aspirin", "Ibuprofen"],
                "medication_category": ["analgesic", "analgesic"],
                "current_inventory": [100.0, 200.0],
                "reorder_point": [50.0, np.nan],
                "max_capacity": [500.0, 400.0],
                "prescription_volume_30d": [30.0, 45.0],
                "avg_weekly_demand": [10.0, 15.0],
                "supplier_id": ["SUP001", "SUP002"],
                "supplier_lead_time_days": [7.0, np.nan],
                "supplier_reliability_score": [0.9, np.nan],
                "historical_shortage_count": [2, 1],
            }
        )
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert not result["reorder_point"].isna().any()
        assert not result["supplier_lead_time_days"].isna().any()
        assert not result["supplier_reliability_score"].isna().any()

    def test_removes_duplicates(self, valid_df):
        df = pd.concat([valid_df, valid_df.iloc[[0]]])
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert len(result) == len(valid_df)

    def test_normalizes_categories(self):
        df = pd.DataFrame(
            {
                "medication_id": ["MED001"],
                "medication_name": ["Aspirin"],
                "medication_category": ["  ANALGESIC  "],
                "current_inventory": [100.0],
                "reorder_point": [50.0],
                "max_capacity": [500.0],
                "prescription_volume_30d": [30.0],
                "avg_weekly_demand": [10.0],
                "supplier_id": ["SUP001"],
                "supplier_lead_time_days": [7.0],
                "supplier_reliability_score": [0.9],
                "historical_shortage_count": [2],
            }
        )
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result["medication_category"].iloc[0] == "analgesic"


class TestDataIngester:
    """Tests for DataIngester."""

    def test_ingest_dataframe(self, valid_df):
        ingester = DataIngester()
        result = ingester.ingest_dataframe(valid_df)
        assert result.success
        assert result.data is not None
        assert len(result.data) == len(valid_df)

    def test_ingest_invalid_dataframe_fails(self, invalid_df):
        ingester = DataIngester()
        result = ingester.ingest_dataframe(invalid_df)
        assert not result.success
        assert result.error_message == "Validation failed"

    def test_get_schema_info(self):
        ingester = DataIngester()
        info = ingester.get_schema_info()
        assert "medication_id" in info
        assert "current_inventory" in info
        assert info["medication_id"]["required"] is True
