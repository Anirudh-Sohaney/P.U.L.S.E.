"""Reproduce the exact backend code path."""
import sys
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from ml.data import DataIngester
from ml.features import FeatureEngineer
from ml.model import XGBoostTrainer
from backend.config import settings

# Simulate what the ingest endpoint does
ingester = DataIngester()
df = pd.read_csv("data/pharmacy_risk_data.csv")
print("1. Raw data shape:", df.shape)

result = ingester.ingest_dataframe(df)
print("2. Ingestion success:", result.success)
print("3. Cleaned data shape:", result.data.shape)

if not result.success:
    print("Errors:", result.validation_result.errors)
    sys.exit(1)

# Save cleaned data
data_dir = Path(settings.DATA_PATH)
data_dir.mkdir(parents=True, exist_ok=True)
result.data.to_csv(data_dir / "cleaned_data.csv", index=False)
print("4. Saved cleaned data")

# Now try _train_model
df = result.data
feature_cols = [c for c in df.columns if c not in ("medication_id", "medication_name", "risk_label")]
X = df[feature_cols].copy()
y = df["risk_label"].astype(int)

print("5. X shape:", X.shape, "y shape:", y.shape)
print("6. y value_counts:", y.value_counts().to_dict())

fe = FeatureEngineer()
X_features = fe.fit_transform(X)
print("7. X_features shape:", X_features.shape)

trainer = XGBoostTrainer()
trainer.fit(X_features, y)
print("8. Model trained")

# This is where it fails - predict_proba
print("9. Calling predict_proba...")
try:
    proba = trainer.predict_proba(X_features)
    print("10. proba shape:", proba.shape)
except Exception as e:
    print(f"ERROR at predict_proba: {e}")
    import traceback
    traceback.print_exc()

# Also try evaluate
try:
    metrics = trainer.evaluate(X_features, y)
    print("11. Metrics:", metrics.accuracy, metrics.roc_auc)
except Exception as e:
    print(f"ERROR at evaluate: {e}")
    import traceback
    traceback.print_exc()

# Save model
trainer.save(Path(settings.MODEL_PATH))
print("12. Model saved")

# Generate predictions
try:
    probabilities = trainer.predict_proba(X_features)
    print("13. Predictions generated, shape:", probabilities.shape)
except Exception as e:
    print(f"ERROR generating predictions: {e}")
    import traceback
    traceback.print_exc()
