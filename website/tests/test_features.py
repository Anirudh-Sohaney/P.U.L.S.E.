"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from ml.features import FeatureEngineer


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for feature engineering."""
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
        }
    )


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_fit_transform(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_df)

    def test_creates_expected_features(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)

        expected_features = [
            "inventory_ratio",
            "inventory_deficit",
            "inventory_coverage_days",
            "demand_volatility",
            "demand_per_capacity",
            "supply_risk_score",
            "days_of_supply_remaining",
            "shortage_frequency",
            "recency_weighted_shortage",
            "demand_supply_mismatch",
            "reliability_inventory_interaction",
            "below_reorder",
            "low_supplier_reliability",
            "high_shortage_history",
        ]

        for feature in expected_features:
            assert feature in features.columns

    def test_no_nan_values(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)
        assert not features.isna().any().any()

    def test_inventory_ratio_calculation(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)

        expected = sample_df["current_inventory"] / sample_df["max_capacity"]
        pd.testing.assert_series_equal(
            features["inventory_ratio"], expected, check_names=False
        )

    def test_supply_risk_score(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)

        expected = (1 - sample_df["supplier_reliability_score"]) * sample_df[
            "supplier_lead_time_days"
        ]
        pd.testing.assert_series_equal(
            features["supply_risk_score"], expected, check_names=False
        )

    def test_binary_indicators(self, sample_df):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_df)

        assert set(features["below_reorder"].unique()).issubset({0, 1})
        assert set(features["low_supplier_reliability"].unique()).issubset({0, 1})
        assert set(features["high_shortage_history"].unique()).issubset({0, 1})

    def test_feature_names(self, sample_df):
        engineer = FeatureEngineer()
        engineer.fit_transform(sample_df)
        names = engineer.get_feature_names()
        assert len(names) > 0
        assert "inventory_ratio" in names

    def test_feature_descriptions(self, sample_df):
        engineer = FeatureEngineer()
        descriptions = engineer.get_feature_importance_names()
        assert "inventory_ratio" in descriptions
        assert "supply_risk_score" in descriptions

    def test_transform_without_fit_raises(self):
        engineer = FeatureEngineer()
        df = pd.DataFrame({"current_inventory": [100]})
        with pytest.raises(RuntimeError):
            engineer.transform(df)
