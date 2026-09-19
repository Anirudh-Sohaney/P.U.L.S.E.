"""Data schema definitions for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DataType(Enum):
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORY = "category"


@dataclass
class FieldDefinition:
    name: str
    data_type: DataType
    description: str
    unit: Optional[str]
    required: bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list] = None
    missing_policy: str = "warn"  # "warn", "error", "fill_mean", "fill_zero", "drop"


# Core schema for pharmacy risk prediction
PHARMACY_RISK_SCHEMA = {
    # Medication identifiers
    "medication_id": FieldDefinition(
        name="medication_id",
        data_type=DataType.STRING,
        description="Unique identifier for the medication",
        unit=None,
        required=True,
        missing_policy="error",
    ),
    "medication_name": FieldDefinition(
        name="medication_name",
        data_type=DataType.STRING,
        description="Name of the medication",
        unit=None,
        required=True,
        missing_policy="error",
    ),
    "medication_category": FieldDefinition(
        name="medication_category",
        data_type=DataType.CATEGORY,
        description="Category/classification of the medication",
        unit=None,
        required=True,
        allowed_values=[
            "analgesic",
            "antibiotic",
            "cardiovascular",
            "diabetes",
            "respiratory",
            "gastrointestinal",
            "neurological",
            "psychiatric",
            "oncology",
            "immunological",
            "other",
        ],
        missing_policy="error",
    ),
    # Inventory features
    "current_inventory": FieldDefinition(
        name="current_inventory",
        data_type=DataType.FLOAT,
        description="Current inventory quantity in units",
        unit="units",
        required=True,
        min_value=0,
        missing_policy="error",
    ),
    "reorder_point": FieldDefinition(
        name="reorder_point",
        data_type=DataType.FLOAT,
        description="Inventory level that triggers reorder",
        unit="units",
        required=True,
        min_value=0,
        missing_policy="fill_zero",
    ),
    "max_capacity": FieldDefinition(
        name="max_capacity",
        data_type=DataType.FLOAT,
        description="Maximum inventory storage capacity",
        unit="units",
        required=True,
        min_value=0,
        missing_policy="error",
    ),
    # Demand features
    "prescription_volume_30d": FieldDefinition(
        name="prescription_volume_30d",
        data_type=DataType.FLOAT,
        description="Number of prescriptions in last 30 days",
        unit="prescriptions",
        required=True,
        min_value=0,
        missing_policy="fill_zero",
    ),
    "avg_weekly_demand": FieldDefinition(
        name="avg_weekly_demand",
        data_type=DataType.FLOAT,
        description="Average weekly demand in units",
        unit="units/week",
        required=True,
        min_value=0,
        missing_policy="fill_zero",
    ),
    "demand_trend": FieldDefinition(
        name="demand_trend",
        data_type=DataType.FLOAT,
        description="Demand trend coefficient (positive = increasing)",
        unit="coefficient",
        required=False,
        missing_policy="fill_zero",
    ),
    # Supplier features
    "supplier_id": FieldDefinition(
        name="supplier_id",
        data_type=DataType.STRING,
        description="Unique identifier for the supplier",
        unit=None,
        required=True,
        missing_policy="error",
    ),
    "supplier_lead_time_days": FieldDefinition(
        name="supplier_lead_time_days",
        data_type=DataType.FLOAT,
        description="Average supplier lead time in days",
        unit="days",
        required=True,
        min_value=0,
        missing_policy="fill_mean",
    ),
    "supplier_reliability_score": FieldDefinition(
        name="supplier_reliability_score",
        data_type=DataType.FLOAT,
        description="Supplier reliability score (0-1)",
        unit="score",
        required=True,
        min_value=0,
        max_value=1,
        missing_policy="fill_mean",
    ),
    # Historical features
    "historical_shortage_count": FieldDefinition(
        name="historical_shortage_count",
        data_type=DataType.INTEGER,
        description="Number of shortage events in last 12 months",
        unit="count",
        required=True,
        min_value=0,
        missing_policy="fill_zero",
    ),
    "days_since_last_shortage": FieldDefinition(
        name="days_since_last_shortage",
        data_type=DataType.FLOAT,
        description="Days since last shortage event",
        unit="days",
        required=False,
        missing_policy="fill_zero",
    ),
    "avg_stockout_duration_days": FieldDefinition(
        name="avg_stockout_duration_days",
        data_type=DataType.FLOAT,
        description="Average stockout duration in days",
        unit="days",
        required=False,
        min_value=0,
        missing_policy="fill_zero",
    ),
    # Risk target (for training)
    "risk_label": FieldDefinition(
        name="risk_label",
        data_type=DataType.INTEGER,
        description="Binary risk label (1 = high risk, 0 = low risk)",
        unit=None,
        required=False,  # Not required for prediction
        allowed_values=[0, 1],
        missing_policy="drop",
    ),
}


def get_required_fields() -> list[str]:
    """Return list of required field names."""
    return [
        name
        for name, field in PHARMACY_RISK_SCHEMA.items()
        if field.required and name != "risk_label"
    ]


def get_feature_fields() -> list[str]:
    """Return list of feature field names (excluding target)."""
    return [name for name in PHARMACY_RISK_SCHEMA if name != "risk_label"]
