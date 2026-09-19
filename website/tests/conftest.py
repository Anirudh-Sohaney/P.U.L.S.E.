"""Shared test fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_pharmacy_data():
    """Create sample pharmacy data for testing."""
    np.random.seed(42)
    n_samples = 50

    return pd.DataFrame(
        {
            "medication_id": [f"MED{i:03d}" for i in range(n_samples)],
            "medication_name": [f"Medication {i}" for i in range(n_samples)],
            "medication_category": np.random.choice(
                ["analgesic", "antibiotic", "cardiovascular", "diabetes"],
                n_samples,
            ),
            "current_inventory": np.random.uniform(50, 500, n_samples),
            "reorder_point": np.random.uniform(20, 100, n_samples),
            "max_capacity": np.random.uniform(200, 1000, n_samples),
            "prescription_volume_30d": np.random.uniform(10, 100, n_samples),
            "avg_weekly_demand": np.random.uniform(5, 50, n_samples),
            "demand_trend": np.random.uniform(-0.5, 0.5, n_samples),
            "supplier_id": [f"SUP{i % 5:03d}" for i in range(n_samples)],
            "supplier_lead_time_days": np.random.uniform(3, 14, n_samples),
            "supplier_reliability_score": np.random.uniform(0.6, 1.0, n_samples),
            "historical_shortage_count": np.random.randint(0, 5, n_samples),
            "days_since_last_shortage": np.random.uniform(10, 100, n_samples),
            "avg_stockout_duration_days": np.random.uniform(0, 3, n_samples),
            "risk_label": np.random.choice([0, 1], n_samples),
        }
    )
