"""Tests for prediction pipeline and explanation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.features import FeatureEngineer
from ml.model import XGBoostTrainer
from ml.prediction import PredictionPipeline, PredictionExplainer


@pytest.fixture
def sample_data():
    """Create sample data for prediction testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        random_state=42,
    )
    feature_names = [
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
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    # Create raw data for pipeline
    raw_data = pd.DataFrame(
        {
            "medication_id": [f"MED{i:03d}" for i in range(100)],
            "medication_name": [f"Medication {i}" for i in range(100)],
            "medication_category": ["analgesic"] * 100,
            "current_inventory": np.random.uniform(50, 500, 100),
            "reorder_point": np.random.uniform(20, 100, 100),
            "max_capacity": np.random.uniform(200, 1000, 100),
            "prescription_volume_30d": np.random.uniform(10, 100, 100),
            "avg_weekly_demand": np.random.uniform(5, 50, 100),
            "demand_trend": np.random.uniform(-0.5, 0.5, 100),
            "supplier_id": [f"SUP{i % 5:03d}" for i in range(100)],
            "supplier_lead_time_days": np.random.uniform(3, 14, 100),
            "supplier_reliability_score": np.random.uniform(0.6, 1.0, 100),
            "historical_shortage_count": np.random.randint(0, 5, 100),
            "days_since_last_shortage": np.random.uniform(10, 100, 100),
            "avg_stockout_duration_days": np.random.uniform(0, 3, 100),
            "risk_label": y_series,
        }
    )

    return X_df, y_series, raw_data


@pytest.fixture
def trained_pipeline(sample_data):
    """Create a trained prediction pipeline."""
    X, y, raw_data = sample_data

    # Train model
    trainer = XGBoostTrainer()
    trainer.fit(X.iloc[:80], y.iloc[:80])

    # Create feature engineer
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(raw_data.iloc[:80])

    # Create pipeline
    pipeline = PredictionPipeline(
        model=trainer,
        feature_engineer=feature_engineer,
        threshold=0.5,
    )

    return pipeline, raw_data.iloc[80:]


class TestPredictionPipeline:
    """Tests for PredictionPipeline."""

    def test_predict_batch(self, trained_pipeline):
        pipeline, test_data = trained_pipeline
        results = pipeline.predict(test_data)
        assert len(results) == len(test_data)
        for result in results:
            assert 0 <= result.risk_score <= 1
            assert result.risk_label in ["high", "low"]
            assert 0 <= result.confidence <= 1
            assert "top_risk_factors" in result.explanation

    def test_predict_single(self, trained_pipeline):
        pipeline, test_data = trained_pipeline
        record = test_data.iloc[0].to_dict()
        result = pipeline.predict_single(record)
        assert 0 <= result.risk_score <= 1
        assert result.risk_label in ["high", "low"]

    def test_set_threshold(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        pipeline.set_threshold(0.7)
        assert pipeline.threshold == 0.7

    def test_invalid_threshold(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        with pytest.raises(ValueError):
            pipeline.set_threshold(1.5)
        with pytest.raises(ValueError):
            pipeline.set_threshold(-0.1)

    def test_prediction_result_structure(self, trained_pipeline):
        pipeline, test_data = trained_pipeline
        result = pipeline.predict_single(test_data.iloc[0].to_dict())
        assert hasattr(result, "risk_score")
        assert hasattr(result, "risk_label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "explanation")
        assert hasattr(result, "feature_values")


class TestPredictionExplainer:
    """Tests for PredictionExplainer."""

    def test_explain(self, trained_pipeline):
        pipeline, test_data = trained_pipeline
        explainer = PredictionExplainer(pipeline.model)
        features = pipeline.feature_engineer.transform(test_data).iloc[0]
        explanation = explainer.explain(features)

        assert "top_risk_factors" in explanation
        assert "contributions" in explanation
        assert "feature_count" in explanation
        assert explanation["feature_count"] > 0

    def test_top_risk_factors(self, trained_pipeline):
        pipeline, test_data = trained_pipeline
        explainer = PredictionExplainer(pipeline.model)
        features = pipeline.feature_engineer.transform(test_data).iloc[0]
        explanation = explainer.explain(features)

        top_factors = explanation["top_risk_factors"]
        assert len(top_factors) <= 5
        for factor in top_factors:
            assert "factor" in factor
            assert "description" in factor
            assert "direction" in factor
            assert factor["direction"] in ["increases", "decreases"]
            assert "impact" in factor

    def test_feature_descriptions(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        explainer = PredictionExplainer(pipeline.model)
        descriptions = explainer._get_feature_descriptions()
        assert "inventory_ratio" in descriptions
        assert "supply_risk_score" in descriptions
