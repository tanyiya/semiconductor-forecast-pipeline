"""
yahoo_bronze_silver.py

Bronze -> Silver transformer for Yahoo Finance stock data.

Responsibilities
----------------
    - Strict deduplication, sorted by Ticker, Date
    - Forward-fill gaps/zeros in Close and Volume using a per-Ticker
      window (last(col, ignorenulls=True))
    - Technical feature engineering: Daily_Return, Volatility_Range,
      MA_5, MA_20
    - Write to data/silver/yahoo/, partitioned by Ticker

This establishes a clean daily time-series per ticker, which is the
standard time dimension the Gold layer will later join against.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.config import BRONZE_YAHOO_DIR, SILVER_YAHOO_DIR
from processing.base_transformer import BaseTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceTransformer(BaseTransformer):
    """Cleans and enriches Yahoo Finance Bronze data into Silver."""

    source_name = "yahoo"
    bronze_dir = BRONZE_YAHOO_DIR
    silver_dir = SILVER_YAHOO_DIR
    partition_by = ["Ticker"]

    def transform(self, df: DataFrame) -> DataFrame:
        df = self._cast_and_sort(df)
        df = self._deduplicate(df)
        df = self._forward_fill_gaps(df)
        df = self._add_technical_features(df)
        df = self._standardise_column_names(df)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _cast_and_sort(df: DataFrame) -> DataFrame:
        """Cast Date to DateType and sort by Ticker, Date."""
        df = df.withColumn("Date", F.to_date(F.col("Date")))
        return df.orderBy("Ticker", "Date")

    @staticmethod
    def _deduplicate(df: DataFrame) -> DataFrame:
        """Drop strict (fully identical) duplicate rows."""
        before = df.count()
        deduped = df.dropDuplicates()
        after = deduped.count()
        removed = before - after
        if removed:
            logger.info("[yahoo] Removed %d strict duplicate row(s)", removed)
        return deduped

    @staticmethod
    def _forward_fill_gaps(df: DataFrame) -> DataFrame:
        """
        Forward-fill null or zero values in Close and Volume using the
        most recent valid prior value per Ticker, ordered by Date.
        """
        window = (
            Window.partitionBy("Ticker")
            .orderBy("Date")
            .rowsBetween(Window.unboundedPreceding, 0)
        )

        # Treat 0 as missing for Close/Volume (a price or volume of
        # exactly zero is not a realistic trading value) before filling.
        df = df.withColumn(
            "Close", F.when(F.col("Close") == 0, None).otherwise(F.col("Close"))
        ).withColumn(
            "Volume", F.when(F.col("Volume") == 0, None).otherwise(F.col("Volume"))
        )

        filled_close = F.last("Close", ignorenulls=True).over(window)
        filled_volume = F.last("Volume", ignorenulls=True).over(window)

        before_nulls = df.filter(F.col("Close").isNull() | F.col("Volume").isNull()).count()

        df = df.withColumn("Close", filled_close).withColumn("Volume", filled_volume)

        if before_nulls:
            logger.info(
                "[yahoo] Forward-filled %d row(s) with missing/zero Close or Volume",
                before_nulls,
            )
        return df

    @staticmethod
    def _add_technical_features(df: DataFrame) -> DataFrame:
        """Add Daily_Return, Volatility_Range, MA_5, and MA_20 columns."""
        df = df.withColumn(
            "Daily_Return",
            F.when(F.col("Open") != 0, (F.col("Close") - F.col("Open")) / F.col("Open")),
        ).withColumn(
            "Volatility_Range",
            F.when(F.col("Close") != 0, (F.col("High") - F.col("Low")) / F.col("Close")),
        )

        ma5_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-4, 0)
        ma20_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-19, 0)

        df = df.withColumn("MA_5", F.avg("Close").over(ma5_window)).withColumn(
            "MA_20", F.avg("Close").over(ma20_window)
        )

        return df

    def apply_quality_filters(self, df: DataFrame) -> DataFrame:
        """
        Beyond the base all-null filter, drop rows missing a Ticker or
        Date - those can't be joined or windowed meaningfully downstream.
        """
        df = super().apply_quality_filters(df)
        return df.filter(F.col("Ticker").isNotNull() & F.col("Date").isNotNull())


def run_yahoo_silver_transform(spark) -> DataFrame:
    """Convenience function used by main.py to run this transformer."""
    transformer = YahooFinanceTransformer(spark)
    return transformer.run()