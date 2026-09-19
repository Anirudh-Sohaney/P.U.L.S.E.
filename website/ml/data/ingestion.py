"""Data ingestion module for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaning import CleaningConfig, DataCleaner
from .schema import PHARMACY_RISK_SCHEMA
from .validation import DataValidator, ValidationResult


@dataclass
class IngestionResult:
    success: bool
    data: pd.DataFrame | None = None
    validation_result: ValidationResult | None = None
    error_message: str | None = None
    source_info: dict[str, Any] | None = None


class DataIngester:
    """Handles data ingestion from various sources."""

    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

    def ingest_csv(
        self,
        file_path: str | Path,
        clean: bool = True,
    ) -> IngestionResult:
        """Ingest data from a CSV file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return IngestionResult(
                success=False,
                error_message=f"File not found: {file_path}",
            )

        try:
            df = pd.read_csv(file_path)
            return self._process_data(df, file_path, clean)
        except Exception as e:
            return IngestionResult(
                success=False,
                error_message=f"Error reading CSV: {str(e)}",
            )

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        source_name: str = "dataframe",
        clean: bool = True,
    ) -> IngestionResult:
        """Ingest data from a pandas DataFrame."""
        return self._process_data(df, source_name, clean)

    def _process_data(
        self,
        df: pd.DataFrame,
        source: Any,
        clean: bool,
    ) -> IngestionResult:
        """Process ingested data through validation and cleaning."""
        source_info = {
            "source": str(source),
            "rows": len(df),
            "columns": list(df.columns),
        }

        # Validate
        validation_result = self.validator.validate(df)

        if not validation_result.is_valid:
            return IngestionResult(
                success=False,
                validation_result=validation_result,
                error_message="Validation failed",
                source_info=source_info,
            )

        # Clean if requested
        if clean:
            df = self.cleaner.fit_transform(df)

        return IngestionResult(
            success=True,
            data=df,
            validation_result=validation_result,
            source_info=source_info,
        )

    def get_schema_info(self) -> dict[str, Any]:
        """Return schema information for documentation."""
        return {
            name: {
                "type": field.data_type.value,
                "required": field.required,
                "description": field.description,
                "unit": field.unit,
                "min_value": field.min_value,
                "max_value": field.max_value,
            }
            for name, field in PHARMACY_RISK_SCHEMA.items()
        }
