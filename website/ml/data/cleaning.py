"""Data cleaning module for the Pharmacy Risk Prediction Platform."""

from dataclasses import dataclass

import pandas as pd

from .schema import PHARMACY_RISK_SCHEMA, DataType


@dataclass
class CleaningConfig:
    fill_numeric_missing: str = "mean"  # "mean", "median", "zero"
    drop_duplicates: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore"
    outlier_threshold: float = 3.0


class DataCleaner:
    """Cleans data for the Pharmacy Risk Prediction Platform."""

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()
        self._fitted_stats: dict = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn statistics from training data for reproducible cleaning."""
        self._fitted_stats = {}

        for col in df.columns:
            if col not in PHARMACY_RISK_SCHEMA:
                continue

            field_def = PHARMACY_RISK_SCHEMA[col]
            if field_def.data_type in (DataType.FLOAT, DataType.INTEGER):
                if self.config.fill_numeric_missing == "mean":
                    self._fitted_stats[col] = {"fill_value": df[col].mean()}
                elif self.config.fill_numeric_missing == "median":
                    self._fitted_stats[col] = {"fill_value": df[col].median()}
                else:
                    self._fitted_stats[col] = {"fill_value": 0}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations to the data."""
        df_clean = df.copy()

        # Type conversions
        df_clean = self._convert_types(df_clean)

        # Handle missing values
        df_clean = self._handle_missing(df_clean)

        # Remove duplicates
        if self.config.drop_duplicates:
            df_clean = df_clean.drop_duplicates()

        # Normalize categories
        df_clean = self._normalize_categories(df_clean)

        # Handle outliers (only for numeric columns)
        df_clean = self._handle_outliers(df_clean)

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column types to match schema."""
        for col_name, field_def in PHARMACY_RISK_SCHEMA.items():
            if col_name not in df.columns:
                continue

            if field_def.data_type == DataType.FLOAT:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            elif field_def.data_type == DataType.INTEGER:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(
                    "Int64"
                )
            elif field_def.data_type == DataType.CATEGORY:
                df[col_name] = df[col_name].astype(str).str.lower().str.strip()

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on schema policy."""
        for col_name, field_def in PHARMACY_RISK_SCHEMA.items():
            if col_name not in df.columns:
                continue

            missing_count = df[col_name].isna().sum()
            if missing_count == 0:
                continue

            if field_def.missing_policy == "drop":
                df = df.dropna(subset=[col_name])
            elif field_def.missing_policy == "fill_zero":
                df[col_name] = df[col_name].fillna(0)
            elif field_def.missing_policy == "fill_mean":
                if col_name in self._fitted_stats:
                    df[col_name] = df[col_name].fillna(
                        self._fitted_stats[col_name]["fill_value"]
                    )
                else:
                    df[col_name] = df[col_name].fillna(df[col_name].mean())

        return df

    def _normalize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical values."""
        for col_name, field_def in PHARMACY_RISK_SCHEMA.items():
            if col_name not in df.columns:
                continue
            if field_def.data_type != DataType.CATEGORY:
                continue

            # Lowercase and strip whitespace
            df[col_name] = df[col_name].astype(str).str.lower().str.strip()

            # Map to allowed values if defined
            if field_def.allowed_values:
                df[col_name] = df[col_name].apply(
                    lambda x: x if x in field_def.allowed_values else "other"
                )

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR or Z-score method."""
        for col_name, field_def in PHARMACY_RISK_SCHEMA.items():
            if col_name not in df.columns:
                continue
            if field_def.data_type not in (DataType.FLOAT, DataType.INTEGER):
                continue
            # Skip columns with fixed allowed values (e.g. binary labels)
            if field_def.allowed_values is not None:
                continue

            series = df[col_name].dropna()
            if len(series) < 10:
                continue

            if self.config.outlier_method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                # Integer extension columns cannot accept fractional IQR bounds.
                # Use a float series while clipping so mixed numeric uploads work
                # consistently on current pandas versions.
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(float).clip(
                    lower=lower, upper=upper
                )
            elif self.config.outlier_method == "zscore":
                from scipy import stats

                z_scores = stats.zscore(series)
                mask = abs(z_scores) <= self.config.outlier_threshold
                df.loc[series.index[~mask], col_name] = None

        return df
