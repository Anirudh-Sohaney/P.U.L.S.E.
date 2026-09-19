"""Prediction pipeline for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data.cleaning import DataCleaner
from ..data.validation import DataValidator
from ..features.engineer import FeatureEngineer
from ..model.trainer import XGBoostTrainer
from .explanation import PredictionExplainer


@dataclass
class PredictionResult:
    risk_score: float
    risk_label: str
    confidence: float
    explanation: dict[str, Any]
    feature_values: dict[str, float]


class PredictionPipeline:
    """End-to-end prediction pipeline."""

    def __init__(
        self,
        model: XGBoostTrainer,
        feature_engineer: FeatureEngineer,
        threshold: float = 0.5,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.threshold = threshold
        self.explainer = PredictionExplainer(model)

    def predict(self, df: pd.DataFrame) -> list[PredictionResult]:
        """Make predictions on new data."""
        # Validate
        validation_result = self.validator.validate(df)
        if not validation_result.is_valid:
            raise ValueError(f"Validation failed: {validation_result.errors}")

        # Clean
        df_clean = self.cleaner.transform(df)

        # Engineer features
        features = self.feature_engineer.transform(df_clean)

        # Get predictions
        probabilities = self.model.predict_proba(features)
        labels = self.model.predict(features)

        # Build results
        results = []
        for i, (prob, label) in enumerate(zip(probabilities, labels)):
            explanation = self.explainer.explain(features.iloc[i])
            feature_values = features.iloc[i].to_dict()

            risk_label = "high" if prob >= self.threshold else "low"

            results.append(
                PredictionResult(
                    risk_score=float(prob),
                    risk_label=risk_label,
                    confidence=float(max(prob, 1 - prob)),
                    explanation=explanation,
                    feature_values=feature_values,
                )
            )

        return results

    def predict_single(self, record: dict[str, Any]) -> PredictionResult:
        """Make prediction on a single record."""
        df = pd.DataFrame([record])
        results = self.predict(df)
        return results[0]

    def set_threshold(self, threshold: float) -> None:
        """Update the risk classification threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
