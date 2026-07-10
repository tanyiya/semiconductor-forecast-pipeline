"""
kaggle_loader.py

Loads the locally-downloaded "Global AI Chip Supply Chain" Kaggle CSV
dataset(s) using Spark, profiles the data, validates it, and persists a
1:1 copy of each dataset to the Bronze layer in Parquet format.

This module does NOT perform business-logic transformations or row-level
changes (that is reserved for the Silver-layer ETL stage). It only
ingests, profiles, and validates -- each CSV is converted to Parquet
as-is (schema inferred, no dedup/filtering/mutation), and each output
folder is named after its source CSV file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

from config.config import BRONZE_KAGGLE_DIR, RAW_KAGGLE_DIR
from ingestion.validator import validate_dataframe
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)


# The six source datasets that make up the "Global AI Chip Supply Chain"
# Kaggle dataset. Each entry is read independently and written to its own
# Bronze parquet folder, named after the CSV (minus the .csv extension).
DATASET_FILENAMES: List[str] = [
    "1_semiconductor_production.csv",
    "2_ai_hardware_demand.csv",
    "3_semiconductor_trade_supply_chain.csv",
    "4_technology_node_innovation.csv",
    "5_geopolitical_risk_sanctions.csv",
    "6_supply_chain_disruption.csv",
    "7_semiconductor_market_economics.csv",
]

# Optional: date column per dataset used for validation, if applicable.
# All of these datasets use a "Year" column rather than a full date, so
# this is left as None by default -- adjust per-file here if needed.
DATE_COLUMNS_BY_FILE: Dict[str, Optional[str]] = {
    name: None for name in DATASET_FILENAMES
}


class KaggleLoader:
    """Encapsulates the Kaggle CSV ingestion workflow for multiple named files."""

    def __init__(self, spark: SparkSession, raw_dir: Path = RAW_KAGGLE_DIR):
        self.spark = spark
        self.raw_dir = raw_dir

    def _resolve_input_path(self, filename: str) -> Path:
        """Build the full path to a specific source CSV file."""
        input_path = self.raw_dir / filename
        logger.info("Reading Kaggle CSV: %s", input_path)
        return input_path

    @log_execution_time("Kaggle CSV Read")
    def read_csv(self, filename: str) -> DataFrame:
        """Read a single Kaggle CSV file into a Spark DataFrame with inferred schema."""
        input_path = self._resolve_input_path(filename)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Expected Kaggle CSV file '{filename}' not found at "
                f"'{input_path}'. Place the Kaggle dataset CSV(s) there "
                f"before running this loader."
            )

        df = (
            self.spark.read.option("header", "true")
            .option("inferSchema", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .csv(str(input_path))
        )
        logger.info("Loaded file: %s", input_path)
        return df

    @staticmethod
    def describe(df: DataFrame, filename: str) -> None:
        """Log schema, row count, and summary statistics for inspection only.

        This is purely informational -- it does not modify the DataFrame.
        """
        logger.info("[%s] Schema:", filename)
        for line in df._jdf.schema().treeString().splitlines():
            logger.info("  %s", line)

        row_count = df.count()
        logger.info("[%s] Row count: %d", filename, row_count)

        logger.info("[%s] Summary statistics:", filename)
        summary_df = df.summary()
        for row in summary_df.collect():
            logger.info("  %s", row.asDict())

    @log_execution_time("Kaggle Bronze Write")
    def write_bronze(self, df: DataFrame, dataset_name: str, output_dir: Path = BRONZE_KAGGLE_DIR) -> None:
        """Persist the DataFrame to the Bronze layer as Parquet, unmodified.

        The output subfolder is named after the source CSV (without the
        .csv extension), e.g. 'semiconductor_production.csv' ->
        '<BRONZE_KAGGLE_DIR>/semiconductor_production/'.
        """
        target_dir = output_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(target_dir))
        )
        logger.info("Kaggle Bronze data written to %s", target_dir)

    @log_execution_time("Kaggle Ingestion Pipeline")
    def run(self) -> Dict[str, DataFrame]:
        """
        Execute the full Kaggle ingestion workflow for every dataset in
        DATASET_FILENAMES:

            read -> profile -> validate (advisory only) -> write to Bronze

        Each dataset is written as-is (no dedup, no row/column changes) so
        the Bronze copy is a faithful Parquet conversion of the original
        CSV. Returns a dict mapping dataset name (without extension) to
        its DataFrame for optional further use by the caller.
        """
        results: Dict[str, DataFrame] = {}

        for filename in DATASET_FILENAMES:
            dataset_name = Path(filename).stem  # e.g. "semiconductor_production"

            df = self.read_csv(filename)
            self.describe(df, filename)

            # Validation is advisory only here: it logs/reports issues but
            # must not alter or drop any rows, since the goal is a 1:1
            # Parquet conversion of the original CSV.
            try:
                validate_dataframe(
                    df,
                    source_name=dataset_name,
                    date_column=DATE_COLUMNS_BY_FILE.get(filename),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Validation reported an issue for %s (continuing without "
                    "modifying data): %s",
                    filename,
                    exc,
                )

            # NOTE: no deduplication step -- the file is imported as-is.
            self.write_bronze(df, dataset_name)
            results[dataset_name] = df

        return results


def run_kaggle_ingestion(spark: SparkSession) -> Dict[str, DataFrame]:
    """Convenience function used by main.py to run this stage."""
    loader = KaggleLoader(spark)
    return loader.run()