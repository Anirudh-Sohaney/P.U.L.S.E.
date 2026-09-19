"""Tests for model training and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.model import XGBoostTrainer, ModelMetrics


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    X, y = make_classification(
        n_samples=200,
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
    return X_df, y_series


@pytest.fixture
def trained_model(classification_data):
    """Create a trained model for testing."""
    X, y = classification_data
    trainer = XGBoostTrainer()
    trainer.fit(X.iloc[:150], y.iloc[:150])
    return trainer, X.iloc[150:], y.iloc[150:]


class TestXGBoostTrainer:
    """Tests for XGBoostTrainer."""

    def test_fit(self, classification_data):
        X, y = classification_data
        trainer = XGBoostTrainer()
        trainer.fit(X, y)
        assert trainer._is_fitted

    def test_predict(self, trained_model):
        model, X_test, _ = trained_model
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, trained_model):
        model, X_test, _ = trained_model
        probabilities = model.predict_proba(X_test)
        assert len(probabilities) == len(X_test)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_evaluate(self, trained_model):
        model, X_test, y_test = trained_model
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.roc_auc <= 1

    def test_cross_validate(self, classification_data):
        X, y = classification_data
        trainer = XGBoostTrainer()
        trainer.fit(X, y)
        cv_results = trainer.cross_validate(X, y, n_folds=3)
        assert "mean_auc" in cv_results
        assert "std_auc" in cv_results
        assert 0 <= cv_results["mean_auc"] <= 1

    def test_get_feature_importance(self, trained_model):
        model, _, _ = trained_model
        importance = model.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == 14

    def test_save_and_load(self, trained_model, tmp_path):
        model, X_test, y_test = trained_model
        model_dir = tmp_path / "model"
        model.save(model_dir)

        loaded_model = XGBoostTrainer.load(model_dir)
        assert loaded_model._is_fitted

        original_metrics = model.evaluate(X_test, y_test)
        loaded_metrics = loaded_model.evaluate(X_test, y_test)
        assert abs(original_metrics.accuracy - loaded_metrics.accuracy) < 0.01

    def test_predict_without_fit_raises(self):
        trainer = XGBoostTrainer()
        X = pd.DataFrame({"feature": [1, 2, 3]})
        with pytest.raises(RuntimeError):
            trainer.predict(X)

    def test_model_params(self, classification_data):
        X, y = classification_data
        custom_params = {
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 50,
        }
        trainer = XGBoostTrainer(model_params=custom_params)
        trainer.fit(X, y)
        assert trainer.model_params["max_depth"] == 4


class TestModelMetrics:
    """Tests for ModelMetrics."""

    def test_to_dict(self):
        metrics = ModelMetrics(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.86,
            roc_auc=0.92,
            true_positives=50,
            true_negatives=40,
            false_positives=5,
            false_negatives=5,
        )
        d = metrics.to_dict()
        assert d["accuracy"] == 0.9
        assert d["true_positives"] == 50

    def test_summary(self):
        metrics = ModelMetrics(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.86,
            roc_auc=0.92,
            true_positives=50,
            true_negatives=40,
            false_positives=5,
            false_negatives=5,
        )
        summary = metrics.summary()
        assert "Accuracy" in summary
        assert "Precision" in summary
