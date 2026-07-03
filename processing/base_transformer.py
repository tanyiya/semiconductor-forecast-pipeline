"""
base_transformer.py

Abstract base class enforcing a uniform Bronze -> Silver lifecycle for
every source-specific transformer (Kaggle, Yahoo Finance, TrendForce).

Lifecycle
---------
    read_bronze()  -> DataFrame   (read Parquet from the Bronze layer)
    transform(df)  -> DataFrame   (source-specific cleaning/enrichment; ABSTRACT)
    apply_quality_filters(df) -> DataFrame  (drop clearly-bad rows post-transform)
    write_silver(df)              (write Parquet to the Silver layer)

``run()`` chains all four steps, logs row counts before/after, times the
whole operation via ``utils.timing.log_execution_time``, and is resilient
to missing Bronze data (raises a clear, catchable error rather than a
cryptic Spark stack trace).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)


class BaseTransformer(ABC):
    """
    Abstract base class for all Bronze -> Silver transformers.

    Subclasses must set the class attributes ``source_name``,
    ``bronze_dir``, and ``silver_dir``, and implement ``transform()``.
    """

    #: Human-readable source name used in logs (e.g. "kaggle", "yahoo").
    source_name: str = "unknown"

    #: Bronze-layer read path. Set by subclasses (typically from config.py).
    bronze_dir: Path

    #: Silver-layer write path. Set by subclasses (typically from config.py).
    silver_dir: Path

    #: Optional column(s) to partition the Silver Parquet output by.
    partition_by: Optional[List[str]] = None

    def __init__(self, spark: SparkSession, config: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        spark : SparkSession
            Shared Spark session (from spark.spark_session.get_spark_session()).
        config : dict, optional
            Source-specific configuration (schema mappings, thresholds,
            feature flags, etc.). Stored as ``self.config``.
        """
        self.spark = spark
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------
    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply source-specific cleaning, casting, deduplication, and
        feature engineering. Must be implemented by every subclass.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared lifecycle steps
    # ------------------------------------------------------------------
    def read_bronze(self) -> DataFrame:
        """Read the Bronze Parquet dataset for this source."""
        if not self.bronze_dir.exists() or not any(self.bronze_dir.iterdir()):
            raise FileNotFoundError(
                f"Bronze data not found for source '{self.source_name}' at "
                f"'{self.bronze_dir}'. Run the ingestion stage for this "
                f"source before running Silver processing."
            )
        logger.info("[%s] Reading Bronze data from %s", self.source_name, self.bronze_dir)
        df = self.spark.read.parquet(str(self.bronze_dir))
        return df

    def apply_quality_filters(self, df: DataFrame) -> DataFrame:
        """
        Default post-transform quality filter: drop rows that are
        entirely null. Subclasses may override/extend this for
        source-specific quality rules (e.g. dropping rows with an
        unparseable price).
        """
        return df.dropna(how="all")

    def write_silver(self, df: DataFrame) -> None:
        """Persist the transformed DataFrame to the Silver layer as Parquet."""
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        writer = df.write.mode("overwrite").option("compression", "snappy")
        if self.partition_by:
            writer = writer.partitionBy(*self.partition_by)
        writer.parquet(str(self.silver_dir))
        logger.info("[%s] Silver data written to %s", self.source_name, self.silver_dir)

    def run(self) -> DataFrame:
        """
        Execute the full Bronze -> Silver pipeline for this source:
        read -> transform -> quality-filter -> write, with row-count
        logging and execution timing.
        """

        @log_execution_time(f"{self.source_name.title()} Bronze->Silver Transform")
        def _execute() -> DataFrame:
            bronze_df = self.read_bronze()
            before_count = bronze_df.count()
            logger.info("[%s] Bronze row count: %d", self.source_name, before_count)

            transformed_df = self.transform(bronze_df)
            filtered_df = self.apply_quality_filters(transformed_df)
            filtered_df = filtered_df.cache()

            after_count = filtered_df.count()
            delta = after_count - before_count
            if delta < 0:
                logger.info(
                    "[%s] Silver row count: %d (%d row(s) net removed during cleaning)",
                    self.source_name,
                    after_count,
                    -delta,
                )
            elif delta > 0:
                logger.info(
                    "[%s] Silver row count: %d (%d row(s) net added, e.g. via "
                    "temporal explode/feature joins)",
                    self.source_name,
                    after_count,
                    delta,
                )
            else:
                logger.info("[%s] Silver row count: %d (unchanged)", self.source_name, after_count)

            self.write_silver(filtered_df)
            return filtered_df

        return _execute()