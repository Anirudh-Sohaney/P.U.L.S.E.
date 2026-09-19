"""XGBoost training pipeline for the Pharmacy Risk Prediction Platform."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .metrics import ModelMetrics


class XGBoostTrainer:
    """XGBoost training pipeline for pharmacy risk prediction."""

    def __init__(self, model_params: dict[str, Any] | None = None):
        self.model_params = model_params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
            "random_state": 42,
        }
        self.model: xgb.XGBClassifier | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_metadata: dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostTrainer":
        """Train the XGBoost model."""
        self._feature_names = list(X_train.columns)

        # Initialize and train model
        self.model = xgb.XGBClassifier(**self.model_params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model, cv=3, method="isotonic"
        )
        self.calibrated_model.fit(X_train, y_train)

        self._is_fitted = True

        # Store training metadata
        self._training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
            "model_params": self.model_params,
            "feature_names": self._feature_names,
        }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk labels."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk probabilities using calibrated model."""
        self._check_fitted()
        return self.calibrated_model.predict_proba(X)[:, 1]

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> ModelMetrics:
        """Evaluate model performance."""
        self._check_fitted()

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            classification_report=report,
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """Perform stratified k-fold cross-validation."""
        self._check_fitted()

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        cv_scores = cross_val_score(
            self.model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1
        )

        return {
            "mean_auc": float(cv_scores.mean()),
            "std_auc": float(cv_scores.std()),
            "fold_scores": cv_scores.tolist(),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        self._check_fitted()

        importance = self.model.feature_importances_
        feature_names = self._feature_names

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save(self, model_dir: str | Path) -> Path:
        """Save model and metadata."""
        self._check_fitted()

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "xgboost_model.joblib"
        joblib.dump(self.model, model_path)

        # Save calibrated model
        calibrated_path = model_dir / "calibrated_model.joblib"
        joblib.dump(self.calibrated_model, calibrated_path)

        # Save metadata
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._training_metadata, f, indent=2)

        # Save feature importance
        importance_path = model_dir / "feature_importance.csv"
        importance_df = self.get_feature_importance()
        importance_df.to_csv(importance_path, index=False)

        return model_dir

    @classmethod
    def load(cls, model_dir: str | Path) -> "XGBoostTrainer":
        """Load a saved model."""
        model_dir = Path(model_dir)

        trainer = cls()

        # Load model
        model_path = model_dir / "xgboost_model.joblib"
        trainer.model = joblib.load(model_path)

        # Load calibrated model
        calibrated_path = model_dir / "calibrated_model.joblib"
        trainer.calibrated_model = joblib.load(calibrated_path)

        # Load metadata
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path) as f:
            trainer._training_metadata = json.load(f)

        trainer._feature_names = trainer._training_metadata.get("feature_names", [])
        trainer._is_fitted = True

        return trainer

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
