"""Debug the training error in detail."""
import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from pathlib import Path

# Load the data
df = pd.read_csv("data/pharmacy_risk_data.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Risk label distribution:")
print(df["risk_label"].value_counts())
print()

# Simulate what data.py does
feature_cols = [c for c in df.columns if c not in ("medication_id", "medication_name", "risk_label")]
print("Feature cols:", feature_cols)
print("Num feature cols:", len(feature_cols))

X = df[feature_cols].copy()
y = df["risk_label"].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("y unique:", y.unique())

# Try feature engineering
from ml.features import FeatureEngineer
fe = FeatureEngineer()
X_features = fe.fit_transform(X)
print("X_features shape:", X_features.shape)
print("X_features columns:", list(X_features.columns))

# Try training
from ml.model import XGBoostTrainer
trainer = XGBoostTrainer()
trainer.fit(X_features, y)

# Try predict
print("Trying predict_proba...")
proba = trainer.predict_proba(X_features)
print("proba shape:", proba.shape)
print("proba[:5]:", proba[:5])
