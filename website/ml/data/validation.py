"""Data validation module for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .schema import PHARMACY_RISK_SCHEMA, DataType, FieldDefinition


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows_validated: int = 0
    rows_passed: int = 0
    rows_failed: int = 0


class DataValidator:
    """Validates incoming data against the pharmacy risk schema."""

    def __init__(self, schema: dict[str, FieldDefinition] | None = None):
        self.schema = schema or PHARMACY_RISK_SCHEMA

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a DataFrame against the schema."""
        result = ValidationResult(is_valid=True, rows_validated=len(df))

        # Check required columns
        self._validate_required_columns(df, result)

        # Validate each column
        for col_name, field_def in self.schema.items():
            if col_name not in df.columns:
                continue
            self._validate_column(df, col_name, field_def, result)

        # Check for duplicates
        self._validate_duplicates(df, result)

        # Update pass/fail counts
        if result.errors:
            result.is_valid = False
            result.rows_failed = result.rows_validated
        else:
            result.rows_passed = result.rows_validated

        return result

    def _validate_required_columns(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check that all required columns are present."""
        for col_name, field_def in self.schema.items():
            if field_def.required and col_name not in df.columns:
                result.errors.append(f"Missing required column: {col_name}")

    def _validate_column(
        self,
        df: pd.DataFrame,
        col_name: str,
        field_def: FieldDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate a single column."""
        series = df[col_name]

        # Check missing values
        missing_count = series.isna().sum()
        if missing_count > 0:
            if field_def.missing_policy == "error":
                result.errors.append(
                    f"Column '{col_name}' has {missing_count} missing values (policy: error)"
                )
            elif field_def.missing_policy == "warn":
                result.warnings.append(
                    f"Column '{col_name}' has {missing_count} missing values"
                )

        # Validate data types
        if field_def.data_type == DataType.FLOAT:
            self._validate_numeric_range(df, col_name, field_def, result)
        elif field_def.data_type == DataType.INTEGER:
            self._validate_integer(df, col_name, field_def, result)
        elif field_def.data_type == DataType.CATEGORY:
            self._validate_category(df, col_name, field_def, result)

    def _validate_numeric_range(
        self,
        df: pd.DataFrame,
        col_name: str,
        field_def: FieldDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate numeric values are within range."""
        series = df[col_name].dropna()

        if not pd.api.types.is_numeric_dtype(series):
            result.errors.append(f"Column '{col_name}' should be numeric")
            return

        if field_def.min_value is not None:
            below_min = (series < field_def.min_value).sum()
            if below_min > 0:
                result.errors.append(
                    f"Column '{col_name}' has {below_min} values below minimum ({field_def.min_value})"
                )

        if field_def.max_value is not None:
            above_max = (series > field_def.max_value).sum()
            if above_max > 0:
                result.errors.append(
                    f"Column '{col_name}' has {above_max} values above maximum ({field_def.max_value})"
                )

    def _validate_integer(
        self,
        df: pd.DataFrame,
        col_name: str,
        field_def: FieldDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate integer values."""
        series = df[col_name].dropna()

        if not pd.api.types.is_integer_dtype(series):
            # Check if values are close to integers
            non_integer = (series % 1 != 0).sum()
            if non_integer > 0:
                result.warnings.append(
                    f"Column '{col_name}' has {non_integer} non-integer values"
                )

        self._validate_numeric_range(df, col_name, field_def, result)

    def _validate_category(
        self,
        df: pd.DataFrame,
        col_name: str,
        field_def: FieldDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate categorical values."""
        if field_def.allowed_values is None:
            return

        series = df[col_name].dropna()
        invalid_values = set(series.unique()) - set(field_def.allowed_values)
        if invalid_values:
            result.errors.append(
                f"Column '{col_name}' has invalid values: {invalid_values}"
            )

    def _validate_duplicates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate records."""
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            result.warnings.append(f"Dataset has {duplicate_count} duplicate rows")
