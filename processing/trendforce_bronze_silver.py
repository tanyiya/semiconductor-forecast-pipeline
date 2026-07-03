"""
trendforce_bronze_silver.py

Bronze -> Silver transformer for scraped TrendForce pricing data.

Responsibilities
----------------
    - Normalise Product and Category text (trim, collapse whitespace,
      lowercase for consistent grouping/joins downstream)
    - Parse the raw string Price into a clean FloatType
    - Standardise the Unit column (e.g. "Gb"/"GB"/"gb" -> "Gb",
      "GiB"/"gib" -> "GiB", "per unit"/"/unit" -> "unit", currency
      variants -> "USD")
    - Temporal alignment: since TrendForce reports prices weekly/
      irregularly rather than daily, each (Product, Category) price
      observation is given an explicit SCD Type 2 style validity
      interval - ``Price_Effective_Date`` and ``Price_Expiration_Date``
      (the day before the next observed price for that same product).
      By default the interval is then exploded into one row per
      calendar day (``explode_daily=True`` in config), which is
      functionally a forward-fill of the price across every day it was
      in effect - giving a clean daily grain that joins directly against
      daily Yahoo Finance data in the Gold layer. Set
      ``explode_daily=False`` to keep the compact interval table instead.
    - Write to data/silver/trendforce/
"""

from __future__ import annotations

from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.config import BRONZE_TRENDFORCE_DIR, SILVER_TRENDFORCE_DIR
from processing.base_transformer import BaseTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

# Known raw Unit variants -> standardised form. Matching is done on a
# lowercased, trimmed copy of the raw value, so keys here must be lowercase.
_UNIT_STANDARDISATION_MAP: Dict[str, str] = {
    "gb": "Gb",
    "gbit": "Gb",
    "gigabit": "Gb",
    "gib": "GiB",
    "gigabyte": "GiB",
    "mb": "Mb",
    "per unit": "unit",
    "/unit": "unit",
    "unit": "unit",
    "usd": "USD",
    "us$": "USD",
    "$": "USD",
    "u.s. dollar": "USD",
}


class TrendForceTransformer(BaseTransformer):
    """Cleans, normalises, and temporally aligns TrendForce Bronze data into Silver."""

    source_name = "trendforce"
    bronze_dir = BRONZE_TRENDFORCE_DIR
    silver_dir = SILVER_TRENDFORCE_DIR

    def __init__(self, spark, config=None):
        merged_config = {"explode_daily": True}
        if config:
            merged_config.update(config)
        super().__init__(spark, merged_config)

    def transform(self, df: DataFrame) -> DataFrame:
        df = self._normalise_text_columns(df)
        df = self._parse_price(df)
        df = self._standardise_units(df)
        df = self._add_temporal_validity(df)
        if self.config.get("explode_daily", True):
            df = self._explode_to_daily(df)
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_text_columns(df: DataFrame) -> DataFrame:
        """Trim, collapse internal whitespace, and lowercase Product/Category."""
        for column in ("Product", "Category"):
            df = df.withColumn(
                column,
                F.lower(F.trim(F.regexp_replace(F.col(column), r"\s+", " "))),
            )
        logger.info("[trendforce] Normalised Product and Category text")
        return df

    @staticmethod
    def _parse_price(df: DataFrame) -> DataFrame:
        """
        Strip currency symbols/commas/whitespace from the raw string Price
        and cast to FloatType. Unparseable values (including empty
        strings after stripping) become null rather than raising - Spark
        runs with ANSI mode by default in this environment, so a plain
        ``.cast("float")`` throws on malformed input; ``try_cast`` is used
        instead to tolerate it. Null prices are dropped later in
        ``apply_quality_filters``.
        """
        cleaned = F.regexp_replace(F.col("Price"), r"[^0-9.\-]", "")
        df = df.withColumn("_price_cleaned", cleaned)
        df = df.withColumn(
            "Price",
            F.when(
                (F.col("_price_cleaned") == "") | F.col("_price_cleaned").isNull(),
                F.lit(None),
            ).otherwise(F.expr("try_cast(_price_cleaned AS FLOAT)")),
        ).drop("_price_cleaned")
        return df

    @staticmethod
    def _standardise_units(df: DataFrame) -> DataFrame:
        """
        Standardise Unit values (Gb/GiB/USD/unit/etc.) using a native
        Spark expression chain rather than a Python UDF. A plain-Python
        UDF spins up a separate Python worker subprocess per partition;
        on some platforms (notably Windows) that worker process can die
        silently under normal conditions, surfacing as an opaque
        "Python worker exited unexpectedly" / EOFException crash deep in
        Spark's shuffle code. Building this as a pure column expression
        keeps everything inside the JVM, avoiding that failure mode
        entirely (and is faster, since there's no per-row Python
        round-trip).
        """
        trimmed = F.trim(F.col("Unit"))
        lower_key = F.lower(trimmed)

        # Fall back to the trimmed original value when it isn't one of
        # the known variants, so nothing is silently discarded.
        standardised = trimmed
        for raw_variant, canonical in _UNIT_STANDARDISATION_MAP.items():
            standardised = F.when(lower_key == raw_variant, F.lit(canonical)).otherwise(
                standardised
            )

        df = df.withColumn(
            "Unit",
            F.when(F.col("Unit").isNull(), F.lit(None))
            .when(trimmed == "", F.lit(None))
            .otherwise(standardised),
        )
        return df

    @staticmethod
    def _add_temporal_validity(df: DataFrame) -> DataFrame:
        """
        Cast the raw report Date, then compute an SCD Type 2 style validity
        interval per (Product, Category): Price_Effective_Date is the
        observation date itself; Price_Expiration_Date is the day before
        the NEXT observed price for that same product/category, or null
        if this is the most recent observation (i.e. still in effect).
        """
        df = df.withColumnRenamed("Date", "Report_Date")
        df = df.withColumn("Report_Date", F.to_date(F.col("Report_Date")))

        window = Window.partitionBy("Product", "Category").orderBy("Report_Date")
        next_report_date = F.lead("Report_Date").over(window)

        df = df.withColumn("Price_Effective_Date", F.col("Report_Date")).withColumn(
            "Price_Expiration_Date",
            F.when(next_report_date.isNotNull(), F.date_sub(next_report_date, 1)),
        )
        logger.info(
            "[trendforce] Computed SCD2 Price_Effective_Date / Price_Expiration_Date "
            "per (Product, Category)"
        )
        return df

    @staticmethod
    def _explode_to_daily(df: DataFrame) -> DataFrame:
        """
        Explode each validity interval into one row per calendar day
        between Price_Effective_Date and Price_Expiration_Date inclusive.
        Intervals still open (null expiration - the latest known price)
        are capped at the maximum Report_Date seen anywhere in the
        dataset, so exploding never runs unbounded into the future.

        The result carries a daily "Date" column suitable for direct
        joining against daily stock data, while Report_Date and the
        original SCD2 bounds are retained for lineage.
        """
        max_date_row = df.agg(F.max("Report_Date").alias("max_date")).collect()[0]
        max_date = max_date_row["max_date"]

        expiration_filled = F.coalesce(F.col("Price_Expiration_Date"), F.lit(max_date))

        df = df.withColumn(
            "Date",
            F.explode(
                F.sequence(
                    F.col("Price_Effective_Date"),
                    expiration_filled,
                    F.expr("interval 1 day"),
                )
            ),
        )
        logger.info(
            "[trendforce] Exploded validity intervals to daily grain (open intervals "
            "capped at %s)",
            max_date,
        )
        return df

    def apply_quality_filters(self, df: DataFrame) -> DataFrame:
        """
        Beyond the base all-null filter, drop rows where Price failed to
        parse (null after casting) - these carry no usable signal.
        """
        df = super().apply_quality_filters(df)
        before = df.count()
        filtered = df.filter(F.col("Price").isNotNull())
        dropped = before - filtered.count()
        if dropped:
            logger.info("[trendforce] Dropped %d row(s) with unparseable Price", dropped)
        return filtered


def run_trendforce_silver_transform(spark) -> DataFrame:
    """Convenience function used by main.py to run this transformer."""
    transformer = TrendForceTransformer(spark)
    return transformer.run()