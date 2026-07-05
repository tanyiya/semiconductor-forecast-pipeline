"""
kaggle_bronze_silver.py

Bronze -> Silver transformer for the Kaggle "Global AI Chip Supply
Chain" dataset.

Because the Kaggle dataset's exact columns are dataset-dependent, the
target schema is declared explicitly in ``config.KAGGLE_SILVER_CONFIG``
(not inferred or hardcoded here) and passed in via ``config`` at
construction time. Update that mapping to match your actual CSV's
columns after downloading it.

Responsibilities
----------------
    - Standardise column names (lowercase, trim, spaces -> underscores)
    - Cast every column strictly to its declared type
    - Standardise string label values (lowercase, trim)
    - Cast the primary date column to DateType (YYYY-MM-DD)
    - Flag and cap statistical outliers (mean +/- N*stddev) in configured
      numeric columns, computed dynamically via Spark summary stats
    - Write to data/silver/kaggle/
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, LongType, StringType

from config.config import BRONZE_KAGGLE_DIR, KAGGLE_SILVER_CONFIG, SILVER_KAGGLE_DIR
from processing.base_transformer import BaseTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

_TYPE_MAP = {
    "string": StringType(),
    "double": DoubleType(),
    "long": LongType(),
    "date": DateType(),
}


class KaggleTransformer(BaseTransformer):
    """Cleans, casts, and outlier-flags the Kaggle Bronze dataset into Silver."""

    source_name = "kaggle"
    bronze_dir = BRONZE_KAGGLE_DIR
    silver_dir = SILVER_KAGGLE_DIR

    def __init__(self, spark, config=None):
        merged_config = {
            "date_column": KAGGLE_SILVER_CONFIG.date_column,
            "column_schema": dict(KAGGLE_SILVER_CONFIG.column_schema),
            "outlier_columns": list(KAGGLE_SILVER_CONFIG.outlier_columns),
            "outlier_std_threshold": KAGGLE_SILVER_CONFIG.outlier_std_threshold,
        }
        if config:
            merged_config.update(config)
        super().__init__(spark, merged_config)

    def transform(self, df: DataFrame) -> DataFrame:
        df = self._standardise_column_names(df)
        df = self._cast_columns(df)
        df = self._standardise_string_values(df)
        df = self._flag_and_cap_outliers(df)
        return df

    # ------------------------------------------------------------------
    def _standardise_column_names(self, df: DataFrame) -> DataFrame:
        return super()._standardise_column_names(df)

    def _cast_columns(self, df: DataFrame) -> DataFrame:
        """
        Cast every column present in ``column_schema`` to its declared
        type. Columns in the schema but missing from the data are
        logged and skipped (not fatal - upstream CSV may have changed).
        Columns present in the data but absent from the schema are left
        as-is and logged, so nothing is silently dropped.
        """
        schema: Dict[str, str] = self.config["column_schema"]
        date_column: str = self.config["date_column"]

        missing = [c for c in schema if c not in df.columns]
        if missing:
            logger.warning(
                "[kaggle] Configured schema columns not found in data (skipped): %s",
                missing,
            )

        unmapped = [c for c in df.columns if c not in schema]
        if unmapped:
            logger.info(
                "[kaggle] Columns present but not in configured schema (left as-is): %s",
                unmapped,
            )

        for column, type_name in schema.items():
            if column not in df.columns:
                continue
            spark_type = _TYPE_MAP.get(type_name)
            if spark_type is None:
                logger.warning(
                    "[kaggle] Unknown type '%s' for column '%s' - skipping cast",
                    type_name,
                    column,
                )
                continue
            if isinstance(spark_type, DateType):
                df = df.withColumn(column, F.to_date(F.col(column)))
            else:
                df = df.withColumn(column, F.col(column).cast(spark_type))

        # Ensure the primary date column is explicitly a DateType, even if
        # it wasn't listed in column_schema under that exact name.
        if date_column in df.columns:
            df = df.withColumn(date_column, F.to_date(F.col(date_column)))

        return df

    @staticmethod
    def _standardise_string_values(df: DataFrame) -> DataFrame:
        """Lowercase and trim every StringType column's values."""
        string_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        for column in string_columns:
            df = df.withColumn(column, F.lower(F.trim(F.col(column))))
        if string_columns:
            logger.info("[kaggle] Standardised string values in columns: %s", string_columns)
        return df

    def _flag_and_cap_outliers(self, df: DataFrame) -> DataFrame:
        """
        For each configured numeric column, compute mean/stddev dynamically
        via Spark aggregate stats, flag rows outside
        [mean - k*stddev, mean + k*stddev] in a new "<col>_is_outlier"
        boolean column, and cap the value itself to those bounds.
        """
        outlier_columns: List[str] = self.config["outlier_columns"]
        k: float = self.config["outlier_std_threshold"]

        for column in outlier_columns:
            if column not in df.columns:
                logger.warning(
                    "[kaggle] Outlier column '%s' not present in data - skipping", column
                )
                continue

            stats = df.select(
                F.mean(F.col(column)).alias("mean"), F.stddev(F.col(column)).alias("stddev")
            ).collect()[0]
            mean_val, stddev_val = stats["mean"], stats["stddev"]

            if mean_val is None or stddev_val is None or stddev_val == 0:
                logger.info(
                    "[kaggle] Skipping outlier check for '%s' (insufficient variance/data)",
                    column,
                )
                continue

            lower_bound = mean_val - k * stddev_val
            upper_bound = mean_val + k * stddev_val

            flag_col = f"{column}_is_outlier"
            df = df.withColumn(
                flag_col,
                (F.col(column) < lower_bound) | (F.col(column) > upper_bound),
            )

            outlier_count = df.filter(F.col(flag_col)).count()
            if outlier_count:
                logger.info(
                    "[kaggle] Column '%s': flagged and capped %d outlier row(s) "
                    "(bounds=[%.2f, %.2f])",
                    column,
                    outlier_count,
                    lower_bound,
                    upper_bound,
                )

            df = df.withColumn(
                column,
                F.when(F.col(column) < lower_bound, lower_bound)
                .when(F.col(column) > upper_bound, upper_bound)
                .otherwise(F.col(column)),
            )

        return df

    def apply_quality_filters(self, df: DataFrame) -> DataFrame:
        """Beyond the base all-null filter, drop strict duplicate rows."""
        df = super().apply_quality_filters(df)
        before = df.count()
        deduped = df.dropDuplicates()
        removed = before - deduped.count()
        if removed:
            logger.info("[kaggle] Removed %d strict duplicate row(s)", removed)
        return deduped


def run_kaggle_silver_transform(spark) -> DataFrame:
    """Convenience function used by main.py to run this transformer."""
    transformer = KaggleTransformer(spark)
    return transformer.run()