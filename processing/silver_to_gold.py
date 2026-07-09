"""
silver_to_gold.py

Builds the Gold-layer galaxy schema from the three Silver datasets
(Kaggle, Yahoo Finance, TrendForce): one shared Dim_Date, three
source-specific dimensions (Dim_Product, Dim_Company, Dim_Country), and
three fact tables at their specified grains:

    Fact_MarketPrice  (TrendForce) - one product price observation for a
                                      product on a reporting date.
    Fact_StockMarket  (Yahoo)      - one ticker for one trading day.
    Fact_Production   (Kaggle)     - one company for one calendar year.

All table and column names are snake_case, matching the Silver layer's
naming convention.

Conformed dimensions
---------------------
Dim_Date is shared by all three facts (Fact_Production references it at
year grain via ``year_key``, which is simply the integer year - joinable
against ``dim_date.year`` - since a yearly-grain fact can't meaningfully
hold a single-day date_key).

Dim_Company is shared by Fact_StockMarket (identified by ticker) and
Fact_Production (identified by company name). These are reconciled via
``config.TICKER_TO_COMPANY_NAME``, a small conformed mapping between the
7 tracked tickers and their company names.

Known limitation - Kaggle column names
----------------------------------------
The real Kaggle "Global AI Chip Supply Chain" dataset's exact Silver
column names could not be verified from this environment. Target metric
columns (production_capacity, fab_count, ai_chip_production,
foundry_revenue, global_market_share) and the country column are
declared in ``config.KAGGLE_GOLD_CONFIG`` - update that config if your
actual Silver Kaggle schema differs. Any configured column not found in
Silver at build time is filled with NULL and logged as a warning rather
than crashing the run, so Fact_Production can still be inspected and the
mapping adjusted iteratively.

Rebuild strategy
-----------------
Every table here is a full drop-and-rebuild from Silver on each run (no
incremental Type-1/2 merge logic) - appropriate for this project's scope.
Dimension surrogate keys are generated via ``dense_rank()`` over a
deterministic ordering of the natural key, so they are stable across
rebuilds as long as the same distinct values are present.
"""

from __future__ import annotations

from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config.config import (
    GOLD_DIM_COMPANY_DIR,
    GOLD_DIM_COUNTRY_DIR,
    GOLD_DIM_DATE_DIR,
    GOLD_DIM_PRODUCT_DIR,
    GOLD_FACT_MARKET_PRICE_DIR,
    GOLD_FACT_PRODUCTION_DIR,
    GOLD_FACT_STOCK_MARKET_DIR,
    KAGGLE_GOLD_CONFIG,
    SILVER_KAGGLE_DIR,
    SILVER_TRENDFORCE_DIR,
    SILVER_YAHOO_DIR,
    TICKER_TO_COMPANY_NAME,
    ensure_directories,
)
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)


# ==========================================================================
# Shared helpers
# ==========================================================================
def _read_silver(spark: SparkSession, path, source_name: str) -> Optional[DataFrame]:
    """
    Read a Silver Parquet dataset. Returns None (with a warning) instead
    of raising if it doesn't exist yet, so the Gold build degrades
    gracefully - e.g. you can build Dim_Date/Fact_StockMarket from Yahoo
    alone before the Kaggle or TrendForce Silver stages have been run.
    """
    if not path.exists() or not any(path.iterdir()):
        logger.warning(
            "[gold] Silver data not found for '%s' at %s - skipping anything "
            "that depends on it",
            source_name,
            path,
        )
        return None
    df = spark.read.parquet(str(path))
    logger.info("[gold] Read Silver '%s': %d row(s)", source_name, df.count())
    return df


def _write_gold(df: DataFrame, path, table_name: str) -> None:
    """Persist a Gold dimension/fact table to Parquet (full overwrite)."""
    path.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").option("compression", "snappy").parquet(str(path))
    logger.info("[gold] Wrote %s to %s", table_name, path)


def _create_map_expr(mapping: Dict[str, str]):
    """Build a Spark map() literal expression from a small Python dict."""
    if not mapping:
        return F.create_map()
    literals = []
    for key, value in mapping.items():
        literals.extend([F.lit(key), F.lit(value)])
    return F.create_map(*literals)


# ==========================================================================
# Dim_Date
# ==========================================================================
@log_execution_time("Build Dim_Date")
def build_dim_date(
    yahoo_df: Optional[DataFrame],
    trendforce_df: Optional[DataFrame],
    kaggle_df: Optional[DataFrame],
    spark: SparkSession,
) -> Optional[DataFrame]:
    """
    Build a full daily calendar spanning the min-to-max date found across
    every available Silver source, with date_key as an integer YYYYMMDD
    (the standard Kimball convention) plus day/month/quarter/year/week
    attributes.
    """
    date_columns = []
    if yahoo_df is not None:
        date_columns.append(yahoo_df.select(F.col("date")))
    if trendforce_df is not None:
        date_columns.append(trendforce_df.select(F.col("date")))
    if kaggle_df is not None and KAGGLE_GOLD_CONFIG.date_column in kaggle_df.columns:
        date_columns.append(
            kaggle_df.select(F.col(KAGGLE_GOLD_CONFIG.date_column).alias("date"))
        )

    if not date_columns:
        logger.warning("[gold] No Silver source available to derive Dim_Date - skipping")
        return None

    all_dates = date_columns[0]
    for extra in date_columns[1:]:
        all_dates = all_dates.unionByName(extra)
    all_dates = all_dates.dropna(subset=["date"])

    if all_dates.limit(1).count() == 0:
        logger.warning(
            "[gold] No valid (non-null) dates found across Silver sources - skipping Dim_Date"
        )
        return None

    # Build the min-to-max day sequence entirely inside Spark's own
    # execution graph (F.sequence over an aggregate) rather than
    # collecting the bounds to the driver and calling
    # spark.createDataFrame() on local Python data. The latter can route
    # through a separate Python worker subprocess for serialisation,
    # which has proven unreliable on some platforms (see spark_session.py
    # for the matching Arrow-disable note) - this version avoids that
    # entirely for calendar generation.
    calendar_df = (
        all_dates.agg(F.sequence(F.min("date"), F.max("date"), F.expr("interval 1 day")).alias("date_seq"))
        .withColumn("full_date", F.explode("date_seq"))
        .select("full_date")
    )

    dim_date = (
        calendar_df.withColumn("date_key", F.date_format(F.col("full_date"), "yyyyMMdd").cast("int"))
        .withColumn("day", F.dayofmonth("full_date"))
        .withColumn("month", F.month("full_date"))
        .withColumn("quarter", F.quarter("full_date"))
        .withColumn("year", F.year("full_date"))
        .withColumn("week", F.weekofyear("full_date"))
        .select("date_key", "full_date", "day", "month", "quarter", "year", "week")
        .orderBy("date_key")
    )

    date_range = dim_date.agg(F.min("full_date").alias("min_date"), F.max("full_date").alias("max_date")).collect()[0]
    logger.info(
        "[gold] Dim_Date spans %s to %s (%d day(s))",
        date_range["min_date"],
        date_range["max_date"],
        dim_date.count(),
    )
    return dim_date


# ==========================================================================
# Dim_Product (TrendForce)
# ==========================================================================
@log_execution_time("Build Dim_Product")
def build_dim_product(trendforce_df: Optional[DataFrame]) -> Optional[DataFrame]:
    """Distinct (product, category, unit) combinations from TrendForce Silver."""
    if trendforce_df is None:
        logger.warning("[gold] TrendForce Silver unavailable - skipping Dim_Product")
        return None

    distinct_products = trendforce_df.select("product", "category", "unit").distinct()

    # Small dimension table - a global (unpartitioned) window is fine here.
    window = Window.orderBy("category", "product", "unit")
    dim_product = distinct_products.withColumn("product_key", F.dense_rank().over(window)).select(
        "product_key", "product", "category", "unit"
    )

    logger.info("[gold] Dim_Product: %d distinct product(s)", dim_product.count())
    return dim_product

# ==========================================================================
# Dim_Company (conformed: Yahoo tickers + Kaggle company names)
# ==========================================================================
@log_execution_time("Build Dim_Company")
def build_dim_company(
    yahoo_df: Optional[DataFrame], kaggle_df: Optional[DataFrame]
) -> Optional[DataFrame]:
    """
    Build the conformed Dim_Company shared by Fact_StockMarket (Yahoo,
    keyed by ticker) and Fact_Production (Kaggle, keyed by company name),
    reconciled via config.TICKER_TO_COMPANY_NAME.
    """
    ticker_to_company = _create_map_expr(TICKER_TO_COMPANY_NAME)
    company_to_ticker = _create_map_expr({v: k for k, v in TICKER_TO_COMPANY_NAME.items()})

    sides = []

    if yahoo_df is not None:
        yahoo_companies = (
            yahoo_df.select(F.col("ticker").alias("ticker"))
            .dropna(subset=["ticker"])
            .distinct()
            .withColumn(
                "company",
                F.coalesce(ticker_to_company.getItem(F.col("ticker")), F.lower(F.col("ticker"))),
            )
            .select("company", "ticker")
        )
        sides.append(yahoo_companies)

    if kaggle_df is not None and KAGGLE_GOLD_CONFIG.company_column in kaggle_df.columns:
        kaggle_companies = (
            kaggle_df.select(F.col(KAGGLE_GOLD_CONFIG.company_column).alias("company"))
            .dropna(subset=["company"])
            .filter(F.expr("TRY_CAST(company AS DOUBLE)").isNull())  # Remove float contamination
            .distinct()
            .withColumn("ticker", company_to_ticker.getItem(F.col("company")))
            .select("company", "ticker")
        )
        sides.append(kaggle_companies)

    if not sides:
        logger.warning("[gold] Neither Yahoo nor Kaggle Silver available - skipping Dim_Company")
        return None

    combined = sides[0]
    for extra in sides[1:]:
        combined = combined.union(extra)  # Use position-based union to avoid case-sensitivity bugs

    # A company may appear on both sides (with/without a ticker) - collapse
    # to one row per company, keeping the first non-null ticker seen.
    dim_company = combined.groupBy("company").agg(
        F.first(F.col("ticker"), ignorenulls=True).alias("ticker")
    )

    window = Window.orderBy("company")
    dim_company = dim_company.withColumn("company_key", F.dense_rank().over(window)).select(
        "company_key", "company", "ticker"
    )

    logger.info("[gold] Dim_Company: %d distinct compan(y/ies)", dim_company.count())
    return dim_company


# ==========================================================================
# Dim_Country (Kaggle)
# ==========================================================================
@log_execution_time("Build Dim_Country")
def build_dim_country(kaggle_df: Optional[DataFrame]) -> Optional[DataFrame]:
    """Distinct countries/regions from Kaggle Silver."""
    country_column = KAGGLE_GOLD_CONFIG.country_column

    if kaggle_df is None:
        logger.warning("[gold] Kaggle Silver unavailable - skipping Dim_Country")
        return None

    if country_column not in kaggle_df.columns:
        logger.warning(
            "[gold] Configured country column '%s' not found in Kaggle Silver "
            "(columns present: %s) - Dim_Country will be empty. Update "
            "config.KAGGLE_GOLD_CONFIG.country_column if your dataset names "
            "this differently.",
            country_column,
            kaggle_df.columns,
        )
        empty_schema = kaggle_df.sparkSession.createDataFrame(
            [], "country_key int, country string"
        )
        return empty_schema

    countries = (
        kaggle_df.select(F.col(country_column).alias("country"))
        .distinct()
        .dropna(subset=["country"])
        .filter(F.expr("TRY_CAST(country AS DOUBLE)").isNull())  # Remove float contamination
    )

    # Remove companies that leaked into the country column by anti-joining valid companies
    if KAGGLE_GOLD_CONFIG.company_column in kaggle_df.columns:
        valid_companies_df = (
            kaggle_df.select(F.col(KAGGLE_GOLD_CONFIG.company_column).alias("company_name"))
            .filter(F.expr("TRY_CAST(company_name AS DOUBLE)").isNull())
            .dropna(subset=["company_name"])
            .distinct()
        )
        countries = countries.join(
            valid_companies_df,
            countries.country == valid_companies_df.company_name,
            "left_anti"
        )

    window = Window.orderBy("country")
    dim_country = countries.withColumn("country_key", F.dense_rank().over(window)).select(
        "country_key", "country"
    )

    logger.info("[gold] Dim_Country: %d distinct countr(y/ies)", dim_country.count())
    return dim_country


# ==========================================================================
# Fact_MarketPrice (TrendForce)
# ==========================================================================
@log_execution_time("Build Fact_MarketPrice")
def build_fact_market_price(
    trendforce_df: Optional[DataFrame],
    dim_date: Optional[DataFrame],
    dim_product: Optional[DataFrame],
) -> Optional[DataFrame]:
    """
    Grain: one product price observation for a product on a reporting
    date. The Silver TrendForce table is exploded to daily grain (one row
    per calendar day a price was in effect); this collapses it back down
    to one row per original (product, category, report_date) observation,
    which is exactly the SCD2 interval already computed in Silver.
    """
    if trendforce_df is None or dim_date is None or dim_product is None:
        logger.warning(
            "[gold] Missing TrendForce Silver, Dim_Date, or Dim_Product - "
            "skipping Fact_MarketPrice"
        )
        return None

    observations = trendforce_df.dropDuplicates(
        ["product", "category", "report_date"]
    ).select(
        "product",
        "category",
        "unit",
        "price",
        "report_date",
        "price_effective_date",
        "price_expiration_date",
    )

    observations = observations.join(
        dim_product, on=["product", "category", "unit"], how="left"
    )

    date_for_report = dim_date.select(
        F.col("date_key").alias("date_key"), F.col("full_date").alias("report_date")
    )
    date_for_effective = dim_date.select(
        F.col("date_key").alias("effective_date_key"),
        F.col("full_date").alias("price_effective_date"),
    )
    date_for_expiration = dim_date.select(
        F.col("date_key").alias("expiration_date_key"),
        F.col("full_date").alias("price_expiration_date"),
    )

    observations = (
        observations.join(date_for_report, on="report_date", how="left")
        .join(date_for_effective, on="price_effective_date", how="left")
        .join(date_for_expiration, on="price_expiration_date", how="left")
    )

    fact = observations.withColumn(
        "price_fact_key", F.monotonically_increasing_id()
    ).select(
        "price_fact_key",
        "date_key",
        "product_key",
        "price",
        "effective_date_key",
        "expiration_date_key",
    )

    logger.info("[gold] Fact_MarketPrice: %d row(s)", fact.count())
    return fact


# ==========================================================================
# Fact_StockMarket (Yahoo)
# ==========================================================================
@log_execution_time("Build Fact_StockMarket")
def build_fact_stock_market(
    yahoo_df: Optional[DataFrame],
    dim_date: Optional[DataFrame],
    dim_company: Optional[DataFrame],
) -> Optional[DataFrame]:
    """Grain: one ticker for one trading day."""
    if yahoo_df is None or dim_date is None or dim_company is None:
        logger.warning(
            "[gold] Missing Yahoo Silver, Dim_Date, or Dim_Company - "
            "skipping Fact_StockMarket"
        )
        return None

    date_lookup = dim_date.select(F.col("date_key"), F.col("full_date").alias("date"))
    company_lookup = dim_company.select("company_key", "ticker")

    stock = yahoo_df.join(date_lookup, on="date", how="left").join(
        company_lookup, on="ticker", how="left"
    )

    fact = stock.withColumn("stock_fact_key", F.monotonically_increasing_id()).select(
        "stock_fact_key",
        "date_key",
        "company_key",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "daily_return",
        "ma_5",
        "ma_20",
    )

    logger.info("[gold] Fact_StockMarket: %d row(s)", fact.count())
    return fact


# ==========================================================================
# Fact_Production (Kaggle)
# ==========================================================================
@log_execution_time("Build Fact_Production")
def build_fact_production(
    kaggle_df: Optional[DataFrame],
    dim_company: Optional[DataFrame],
    dim_country: Optional[DataFrame],
) -> Optional[DataFrame]:
    """
    Grain: one company for one calendar year. Rolls up (potentially
    finer-grained) Kaggle Silver rows to company+year using the
    aggregation function configured per metric column in
    config.KAGGLE_GOLD_CONFIG.metric_aggregations. year_key is the plain
    integer year (joinable against dim_date.year - a yearly-grain fact
    has no single date_key to reference).
    """
    if kaggle_df is None or dim_company is None or dim_country is None:
        logger.warning(
            "[gold] Missing Kaggle Silver, Dim_Company, or Dim_Country - "
            "skipping Fact_Production"
        )
        return None

    company_col = KAGGLE_GOLD_CONFIG.company_column
    country_col = KAGGLE_GOLD_CONFIG.country_column
    date_col = KAGGLE_GOLD_CONFIG.date_column
    metric_aggregations = KAGGLE_GOLD_CONFIG.metric_aggregations

    df = kaggle_df

    if date_col in df.columns:
        df = df.withColumn("year", F.year(F.col(date_col)))
    elif "year" in df.columns:
        logger.info(
            "[gold] Kaggle Silver has no configured date column '%s'; using existing 'year' column instead",
            date_col,
        )
        df = df.withColumn("year", F.col("year").cast("int"))
    else:
        logger.error(
            "[gold] Configured Kaggle date column '%s' not found (columns: %s) - "
            "cannot derive calendar year, skipping Fact_Production",
            date_col,
            df.columns,
        )
        return None

    missing_metrics = [c for c in metric_aggregations if c not in df.columns]
    if missing_metrics:
        logger.warning(
            "[gold] Metric column(s) not found in Kaggle Silver (filled with NULL): "
            "%s. Update config.KAGGLE_GOLD_CONFIG.metric_aggregations if your "
            "dataset names these differently.",
            missing_metrics,
        )
        for col in missing_metrics:
            df = df.withColumn(col, F.lit(None).cast("double"))

    if country_col not in df.columns:
        logger.warning(
            "[gold] Country column '%s' not found in Kaggle Silver - Fact_Production "
            "rows will have a null country_key",
            country_col,
        )
        df = df.withColumn(country_col, F.lit(None).cast("string"))

    if company_col not in df.columns:
        logger.error(
            "[gold] Company column '%s' not found in Kaggle Silver - cannot build "
            "Fact_Production",
            company_col,
        )
        return None

    for metric_column in metric_aggregations:
        if metric_column in df.columns:
            df = df.withColumn(
                metric_column,
                F.expr(f"try_cast({metric_column} AS DOUBLE)"),
            )

    agg_func_map = {"sum": F.sum, "avg": F.avg, "max": F.max, "min": F.min}
    agg_exprs = []
    for column, func_name in metric_aggregations.items():
        agg_func = agg_func_map.get(func_name)
        if agg_func is None:
            logger.warning(
                "[gold] Unknown aggregation '%s' for column '%s' - defaulting to 'avg'",
                func_name,
                column,
            )
            agg_func = F.avg
        agg_exprs.append(agg_func(F.col(column)).alias(column))

    rolled_up = df.groupBy(F.col(company_col).alias("company"), "year", F.col(country_col).alias("country")).agg(
        *agg_exprs
    )

    rolled_up = rolled_up.join(dim_company.select("company_key", "company"), on="company", how="left")
    rolled_up = rolled_up.join(dim_country.select("country_key", "country"), on="country", how="left")

    fact = rolled_up.withColumnRenamed("year", "year_key").withColumn(
        "production_fact_key", F.monotonically_increasing_id()
    )

    final_columns = ["production_fact_key", "year_key", "company_key", "country_key"] + list(
        metric_aggregations.keys()
    )
    fact = fact.select(*final_columns)

    logger.info("[gold] Fact_Production: %d row(s)", fact.count())
    return fact


# ==========================================================================
# Orchestration
# ==========================================================================
@log_execution_time("Full Silver -> Gold Build")
def run_silver_to_gold(spark: SparkSession) -> Dict[str, DataFrame]:
    """
    Read every available Silver source, build the full galaxy schema
    (Dim_Date, Dim_Product, Dim_Company, Dim_Country, and the three fact
    tables), write each to the Gold layer, and return whatever was
    successfully built (as a dict keyed by table name) for inspection or
    testing.
    """
    ensure_directories()

    yahoo_df = _read_silver(spark, SILVER_YAHOO_DIR, "yahoo")
    trendforce_df = _read_silver(spark, SILVER_TRENDFORCE_DIR, "trendforce")
    kaggle_df = _read_silver(spark, SILVER_KAGGLE_DIR, "kaggle")

    built: Dict[str, DataFrame] = {}

    dim_date = build_dim_date(yahoo_df, trendforce_df, kaggle_df, spark)
    if dim_date is not None:
        _write_gold(dim_date, GOLD_DIM_DATE_DIR, "Dim_Date")
        built["dim_date"] = dim_date

    dim_product = build_dim_product(trendforce_df)
    if dim_product is not None:
        _write_gold(dim_product, GOLD_DIM_PRODUCT_DIR, "Dim_Product")
        built["dim_product"] = dim_product

    dim_company = build_dim_company(yahoo_df, kaggle_df)
    if dim_company is not None:
        _write_gold(dim_company, GOLD_DIM_COMPANY_DIR, "Dim_Company")
        built["dim_company"] = dim_company

    dim_country = build_dim_country(kaggle_df)
    if dim_country is not None:
        _write_gold(dim_country, GOLD_DIM_COUNTRY_DIR, "Dim_Country")
        built["dim_country"] = dim_country

    fact_market_price = build_fact_market_price(trendforce_df, dim_date, dim_product)
    if fact_market_price is not None:
        _write_gold(fact_market_price, GOLD_FACT_MARKET_PRICE_DIR, "Fact_MarketPrice")
        built["fact_market_price"] = fact_market_price

    fact_stock_market = build_fact_stock_market(yahoo_df, dim_date, dim_company)
    if fact_stock_market is not None:
        _write_gold(fact_stock_market, GOLD_FACT_STOCK_MARKET_DIR, "Fact_StockMarket")
        built["fact_stock_market"] = fact_stock_market

    fact_production = build_fact_production(kaggle_df, dim_company, dim_country)
    if fact_production is not None:
        _write_gold(fact_production, GOLD_FACT_PRODUCTION_DIR, "Fact_Production")
        built["fact_production"] = fact_production

    logger.info("[gold] Gold build complete. Tables written: %s", list(built.keys()))
    return built


def main() -> None:
    from spark.spark_session import get_spark_session, stop_spark_session

    spark = get_spark_session()
    try:
        run_silver_to_gold(spark)
    finally:
        stop_spark_session()


if __name__ == "__main__":
    main()