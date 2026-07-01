"""
kaggle_loader.py

Loads the locally-downloaded "Global AI Chip Supply Chain" Kaggle CSV
dataset(s) using Spark, profiles the data, validates it, and persists a
cleaned copy to the Bronze layer in Parquet format.

This module does NOT perform business-logic transformations (that is
reserved for the Silver-layer ETL stage). It only ingests, profiles, and
validates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from config.config import BRONZE_KAGGLE_DIR, KAGGLE_CSV_GLOB, RAW_KAGGLE_DIR
from ingestion.validator import validate_dataframe
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)


class KaggleLoader:
    """Encapsulates the Kaggle CSV ingestion workflow."""

    def __init__(self, spark: SparkSession, raw_dir: Path = RAW_KAGGLE_DIR):
        self.spark = spark
        self.raw_dir = raw_dir

    def _resolve_input_path(self) -> str:
        """
        Build the glob path used by Spark's CSV reader. Spark accepts a
        directory + glob pattern directly, so no manual file listing is
        required.
        """
        pattern = str(self.raw_dir / KAGGLE_CSV_GLOB)
        logger.info("Reading Kaggle CSV(s) matching: %s", pattern)
        return pattern

    @log_execution_time("Kaggle CSV Read")
    def read_csv(self) -> DataFrame:
        """Read the Kaggle CSV(s) into a Spark DataFrame with inferred schema."""
        input_path = self._resolve_input_path()

        matching_files = list(self.raw_dir.glob(KAGGLE_CSV_GLOB))
        if not matching_files:
            raise FileNotFoundError(
                f"No CSV files found in '{self.raw_dir}' matching pattern "
                f"'{KAGGLE_CSV_GLOB}'. Place the Kaggle dataset CSV(s) there "
                f"before running this loader."
            )

        df = (
            self.spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(input_path)
        )
        logger.info("Loaded %d file(s) from %s", len(matching_files), self.raw_dir)
        return df

    @staticmethod
    def describe(df: DataFrame) -> None:
        """Log schema, row count, and summary statistics for inspection."""
        logger.info("Schema:")
        for line in df._jdf.schema().treeString().splitlines():
            logger.info("  %s", line)

        row_count = df.count()
        logger.info("Row count: %d", row_count)

        logger.info("Summary statistics:")
        # describe() only covers numeric/string columns meaningfully;
        # still useful for a quick profile.
        summary_df = df.summary()
        for row in summary_df.collect():
            logger.info("  %s", row.asDict())

    @staticmethod
    def deduplicate(df: DataFrame) -> DataFrame:
        """Drop fully duplicated rows, logging how many were removed."""
        before = df.count()
        deduped = df.dropDuplicates()
        after = deduped.count()
        removed = before - after
        if removed:
            logger.info("Removed %d duplicate row(s)", removed)
        return deduped

    @log_execution_time("Kaggle Bronze Write")
    def write_bronze(self, df: DataFrame, output_dir: Path = BRONZE_KAGGLE_DIR) -> None:
        """Persist the cleaned DataFrame to the Bronze layer as Parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(output_dir))
        )
        logger.info("Kaggle Bronze data written to %s", output_dir)

    @log_execution_time("Kaggle Ingestion Pipeline")
    def run(self, expected_date_column: Optional[str] = None) -> DataFrame:
        """
        Execute the full Kaggle ingestion workflow:
        read -> profile -> validate -> dedupe -> write to Bronze.

        Returns the cleaned DataFrame for optional further use by the caller.
        """
        df = self.read_csv()
        self.describe(df)

        validate_dataframe(
            df,
            source_name="kaggle",
            date_column=expected_date_column,
        )

        cleaned_df = self.deduplicate(df)
        self.write_bronze(cleaned_df)
        return cleaned_df


def run_kaggle_ingestion(
    spark: SparkSession, expected_date_column: Optional[str] = None
) -> DataFrame:
    """Convenience function used by main.py to run this stage."""
    loader = KaggleLoader(spark)
    return loader.run(expected_date_column=expected_date_column)
