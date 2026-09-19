"""Prediction explanation module for the Pharmacy Risk Prediction Platform."""

from typing import Any

import numpy as np
import pandas as pd


class PredictionExplainer:
    """Explains individual predictions using feature contributions."""

    def __init__(self, model):
        self.model = model
        self._feature_descriptions = self._get_feature_descriptions()

    def explain(self, features: pd.Series) -> dict[str, Any]:
        """Explain a single prediction."""
        # Get feature importance from the model
        importance_df = self.model.get_feature_importance()
        importance_dict = dict(
            zip(importance_df["feature"], importance_df["importance"])
        )

        # Calculate contribution scores
        contributions = {}
        for feature_name, importance in importance_dict.items():
            if feature_name in features.index:
                value = features[feature_name]
                # Simple contribution: importance * normalized value
                contribution = importance * value
                contributions[feature_name] = {
                    "value": float(value),
                    "importance": float(importance),
                    "contribution": float(contribution),
                    "description": self._feature_descriptions.get(
                        feature_name, feature_name
                    ),
                }

        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(
                contributions.items(),
                key=lambda x: abs(x[1]["contribution"]),
                reverse=True,
            )
        )

        # Get top risk factors
        top_risk_factors = []
        for name, contrib in list(sorted_contributions.items())[:5]:
            direction = "increases" if contrib["contribution"] > 0 else "decreases"
            top_risk_factors.append(
                {
                    "factor": name,
                    "description": contrib["description"],
                    "direction": direction,
                    "impact": abs(contrib["contribution"]),
                }
            )

        return {
            "top_risk_factors": top_risk_factors,
            "contributions": sorted_contributions,
            "feature_count": len(contributions),
        }

    def _get_feature_descriptions(self) -> dict[str, str]:
        """Get human-readable feature descriptions."""
        return {
            "inventory_ratio": "Current inventory level relative to maximum capacity",
            "inventory_deficit": "Inventory level compared to reorder point",
            "inventory_coverage_days": "Number of days inventory will last at current demand",
            "demand_volatility": "Variation in demand over time",
            "demand_per_capacity": "Weekly demand as percentage of storage capacity",
            "supply_risk_score": "Combined supplier reliability and lead time risk",
            "days_of_supply_remaining": "Estimated days until stockout",
            "shortage_frequency": "How often shortages occur (monthly rate)",
            "recency_weighted_shortage": "Historical shortages weighted by how recent they were",
            "demand_supply_mismatch": "Ratio of demand to available supply",
            "reliability_inventory_interaction": "Supplier reliability weighted by inventory level",
            "below_reorder": "Whether inventory is currently below reorder point",
            "low_supplier_reliability": "Whether supplier reliability is below threshold",
            "high_shortage_history": "Whether medication has high historical shortage count",
        }
