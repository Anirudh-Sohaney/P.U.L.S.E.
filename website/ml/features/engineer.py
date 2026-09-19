"""Feature engineering module for the Pharmacy Risk Prediction Platform."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates model features from pharmacy data."""

    def __init__(self):
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit feature engineer on training data."""
        self._fitted = True
        self._feature_names = self._get_feature_names(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data into model features."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")

        features = pd.DataFrame()

        # Inventory features
        features["inventory_ratio"] = df["current_inventory"] / df["max_capacity"].replace(
            0, np.nan
        )
        features["inventory_deficit"] = (
            df["current_inventory"] - df["reorder_point"]
        ) / df["reorder_point"].replace(0, np.nan)
        features["inventory_coverage_days"] = df["current_inventory"] / df[
            "avg_weekly_demand"
        ].replace(0, np.nan) * 7

        # Demand features
        features["demand_volatility"] = df["demand_trend"].abs()
        features["demand_per_capacity"] = df["avg_weekly_demand"] / df["max_capacity"].replace(
            0, np.nan
        )

        # Supply features
        features["supply_risk_score"] = (
            1 - df["supplier_reliability_score"]
        ) * df["supplier_lead_time_days"]
        features["days_of_supply_remaining"] = df["current_inventory"] / df[
            "avg_weekly_demand"
        ].replace(0, np.nan) * 7

        # Historical features
        features["shortage_frequency"] = df["historical_shortage_count"] / 12  # Monthly rate
        features["recency_weighted_shortage"] = df["historical_shortage_count"] / (
            df["days_since_last_shortage"].replace(0, 1) + 1
        )

        # Interaction features
        features["demand_supply_mismatch"] = (
            df["avg_weekly_demand"] * df["supplier_lead_time_days"]
        ) / df["current_inventory"].replace(0, np.nan)
        features["reliability_inventory_interaction"] = (
            df["supplier_reliability_score"] * df["current_inventory"]
        ) / df["max_capacity"].replace(0, np.nan)

        # Risk indicators
        features["below_reorder"] = (df["current_inventory"] < df["reorder_point"]).astype(int)
        features["low_supplier_reliability"] = (df["supplier_reliability_score"] < 0.7).astype(
            int
        )
        features["high_shortage_history"] = (df["historical_shortage_count"] > 3).astype(int)

        # Fill any NaN values created by division
        features = features.fillna(0)

        self._feature_names = list(features.columns)
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get feature names after transformation."""
        # This will be populated after fit_transform
        return []

    def get_feature_names(self) -> list[str]:
        """Return feature names."""
        return self._feature_names

    def get_feature_importance_names(self) -> dict[str, str]:
        """Return human-readable feature descriptions."""
        return {
            "inventory_ratio": "Current inventory as % of capacity",
            "inventory_deficit": "Inventory relative to reorder point",
            "inventory_coverage_days": "Days of inventory at current demand",
            "demand_volatility": "Absolute demand trend (volatility indicator)",
            "demand_per_capacity": "Weekly demand as % of capacity",
            "supply_risk_score": "Supplier risk (low reliability + long lead time)",
            "days_of_supply_remaining": "Days until stockout at current rate",
            "shortage_frequency": "Monthly shortage event rate",
            "recency_weighted_shortage": "Shortage weighted by recency",
            "demand_supply_mismatch": "Demand vs supply capacity ratio",
            "reliability_inventory_interaction": "Supplier reliability weighted by inventory",
            "below_reorder": "Currently below reorder point",
            "low_supplier_reliability": "Supplier reliability below threshold",
            "high_shortage_history": "High historical shortage count",
        }
