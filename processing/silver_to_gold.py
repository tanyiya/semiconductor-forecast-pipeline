"""
scripts/silver_to_gold.py

Lakehouse Pipeline: Silver to Gold Transformation
Transforms cleaned Silver parquet data into a dimensional Gold star schema.
"""

import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType
from pyspark.sql.utils import AnalysisException

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
SILVER_BASE_PATH = "data/silver"
GOLD_BASE_PATH = "data/gold"

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def create_spark_session() -> SparkSession:
    """Initialize and return a SparkSession."""
    logger.info("Initializing SparkSession...")
    return SparkSession.builder \
        .appName("SilverToGold_Transformation") \
        .getOrCreate()

def read_silver_table(spark: SparkSession, path: str, recursive: bool = False):
    """Read a parquet file/folder from the Silver layer."""
    try:
        if recursive:
            return spark.read.option("recursiveFileLookup", "true").parquet(path)
        return spark.read.parquet(path)
    except AnalysisException as e:
        logger.error(f"Failed to read path {path}. Error: {e}")
        sys.exit(1)

def write_gold_table(df, path: str, table_name: str):
    """Write DataFrame to Gold layer, print schema and row count for quality checks."""
    logger.info(f"Writing {table_name} to {path}...")
    
    # Save as parquet
    df.write.mode("overwrite").parquet(path)
    
    # Quality Checks
    row_count = df.count()
    logger.info(f"=======================================")
    logger.info(f"QUALITY CHECK: {table_name}")
    logger.info(f"Rows: {row_count}")
    logger.info("Schema:")
    df.printSchema()
    logger.info(f"=======================================\n")

def map_company_name(col):
    """Apply mapping logic for company names to standardized tickers where applicable."""
    return F.when(F.upper(col).contains("NVIDIA"), "NVDA") \
            .when(F.upper(col).contains("ADVANCED MICRO"), "AMD") \
            .when(F.upper(col).contains("INTEL"), "INTC") \
            .when(F.upper(col).contains("BROADCOM"), "AVGO") \
            .when(F.upper(col).contains("MICRON"), "MU") \
            .when(F.upper(col).contains("QUALCOMM"), "QCOM") \
            .when(F.upper(col).contains("TAIWAN SEMI"), "TSM") \
            .otherwise(col)

def standardize_to_date(col):
    """Convert year integer or string date to yyyy-MM-dd date type."""
    return F.when(
        F.length(col.cast("string")) == 4, 
        F.to_date(F.concat(col.cast("string"), F.lit("-01-01")), "yyyy-MM-dd")
    ).otherwise(F.to_date(col))

# ---------------------------------------------------------
# Transformation Logic
# ---------------------------------------------------------

def main():
    spark = create_spark_session()
    
    logger.info("Starting Silver -> Gold transformation...")

    # 1. Read Silver Data
    logger.info("Reading Silver data...")
    df_prod = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/1_semiconductor_production", recursive=True)
    df_demand = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/2_ai_hardware_demand", recursive=True)
    df_trade = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/3_semiconductor_trade_supply_chain", recursive=True)
    df_tech = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/4_technology_node_innovation", recursive=True)
    df_geo = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/5_geopolitical_risk_sanctions", recursive=True)
    df_disruption = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/6_supply_chain_disruption", recursive=True)
    df_market = read_silver_table(spark, f"{SILVER_BASE_PATH}/kaggle/7_semiconductor_market_economics", recursive=True)
    
    df_trendforce = read_silver_table(spark, f"{SILVER_BASE_PATH}/trendforce", recursive=True)
    # Read Yahoo data without recursiveFileLookup to preserve Ticker partition column
    df_yahoo = read_silver_table(spark, f"{SILVER_BASE_PATH}/yahoo", recursive=False)


    # ==========================================
    # BUILD DIMENSIONS
    # ==========================================
    logger.info("Building Dimension Tables...")

    # --- dim_company ---
    logger.info("Creating dim_company...")
    companies_prod = df_prod.select(F.col("company").alias("company_name"))
    companies_tech = df_tech.select(F.col("company").alias("company_name"))
    companies_yahoo = df_yahoo.select(F.col("Ticker").alias("company_name"))
    
    dim_company = companies_prod.union(companies_tech).union(companies_yahoo) \
        .dropna(subset=["company_name"]) \
        .dropDuplicates() \
        .withColumn("ticker", map_company_name(F.col("company_name"))) \
        .withColumn("company_key", F.monotonically_increasing_id()) \
        .select("company_key", "company_name", "ticker")
    
    write_gold_table(dim_company, f"{GOLD_BASE_PATH}/dimensions/dim_company", "dim_company")


    # --- dim_country ---
    logger.info("Creating dim_country...")
    countries = df_prod.select(F.col("country").alias("country_name")) \
        .union(df_demand.select(F.col("country").alias("country_name"))) \
        .union(df_geo.select(F.col("country").alias("country_name"))) \
        .union(df_trade.select(F.col("exporting_country").alias("country_name"))) \
        .union(df_trade.select(F.col("importing_country").alias("country_name")))
    
    dim_country = countries.dropna(subset=["country_name"]) \
        .dropDuplicates() \
        .withColumn("country_key", F.monotonically_increasing_id()) \
        .select("country_key", "country_name")
        
    write_gold_table(dim_country, f"{GOLD_BASE_PATH}/dimensions/dim_country", "dim_country")


    # --- dim_region ---
    logger.info("Creating dim_region...")
    dim_region = df_disruption.select(F.col("region").alias("region_name")) \
        .dropna(subset=["region_name"]) \
        .dropDuplicates() \
        .withColumn("region_key", F.monotonically_increasing_id()) \
        .select("region_key", "region_name")
        
    write_gold_table(dim_region, f"{GOLD_BASE_PATH}/dimensions/dim_region", "dim_region")


    # --- dim_product ---
    logger.info("Creating dim_product...")
    dim_product = df_trendforce.select(
        F.col("product").alias("product_name"), 
        "category", 
        "unit"
    ).dropna(subset=["product_name"]) \
     .dropDuplicates() \
     .withColumn("product_key", F.monotonically_increasing_id()) \
     .select("product_key", "product_name", "category", "unit")
     
    write_gold_table(dim_product, f"{GOLD_BASE_PATH}/dimensions/dim_product", "dim_product")


    # --- dim_technology ---
    logger.info("Creating dim_technology...")
    dim_technology = df_tech.select("node_size_nm") \
        .union(df_prod.select(F.col("technology_node_nm").alias("node_size_nm"))) \
        .dropna(subset=["node_size_nm"]) \
        .dropDuplicates() \
        .withColumn("technology_key", F.monotonically_increasing_id()) \
        .select("technology_key", "node_size_nm")
        
    write_gold_table(dim_technology, f"{GOLD_BASE_PATH}/dimensions/dim_technology", "dim_technology")


# --- dim_date ---
    logger.info("Creating dim_date...")
    
    # Cast all incoming date/year columns to string before unioning to prevent INCOMPATIBLE_COLUMN_TYPE errors
    dates = df_prod.select(F.col("year").cast("string").alias("raw_date")) \
        .union(df_demand.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_trade.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_tech.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_geo.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_disruption.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_market.select(F.col("year").cast("string").alias("raw_date"))) \
        .union(df_trendforce.select(F.col("date").cast("string").alias("raw_date"))) \
        .union(df_yahoo.select(F.col("date").cast("string").alias("raw_date")))

    dim_date = dates.dropna(subset=["raw_date"]) \
        .select(standardize_to_date(F.col("raw_date")).alias("full_date")) \
        .dropDuplicates() \
        .withColumn("date_key", F.monotonically_increasing_id()) \
        .withColumn("day", F.dayofmonth("full_date")) \
        .withColumn("month", F.month("full_date")) \
        .withColumn("quarter", F.quarter("full_date")) \
        .withColumn("year", F.year("full_date")) \
        .withColumn("week_number", F.weekofyear("full_date")) \
        .select("date_key", "full_date", "day", "month", "quarter", "year", "week_number")

    write_gold_table(dim_date, f"{GOLD_BASE_PATH}/dimensions/dim_date", "dim_date")


    # ==========================================
    # BUILD FACT TABLES
    # ==========================================
    logger.info("Building Fact Tables...")

    # Prep DataFrames for Fact Join Mapping
    # Creating broadcast lookups where practical
    dim_date_lkp = dim_date.select("full_date", "date_key")
    dim_company_lkp = dim_company.select("company_name", "company_key")
    dim_company_ticker_lkp = dim_company.select("ticker", "company_key").filter(F.col("ticker").isNotNull())
    dim_country_lkp = dim_country.select("country_name", "country_key")
    dim_region_lkp = dim_region.select("region_name", "region_key")
    dim_product_lkp = dim_product.select("product_name", "product_key")
    dim_tech_lkp = dim_technology.select("node_size_nm", "technology_key")


    # --- FACT 1: fact_semiconductor_market ---
    logger.info("Creating fact_semiconductor_market...")
    
    # Standardize dates and map keys for all 4 sources
    f1_demand = df_demand.withColumn("full_date", standardize_to_date(F.col("year"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_country_lkp, df_demand.country == dim_country_lkp.country_name, "left") \
        .withColumn("product_key", F.lit(None).cast("long")) \
        .withColumn("company_key", F.lit(None).cast("long")) \
        .select("date_key", "country_key", "product_key", "company_key",
                "ai_gpu_demand", "data_center_count", "ai_compute_power",
                "cloud_ai_investment", "training_compute_flops", "ai_model_count")

    f1_market = df_market.withColumn("full_date", standardize_to_date(F.col("year"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .withColumn("country_key", F.lit(None).cast("long")) \
        .withColumn("product_key", F.lit(None).cast("long")) \
        .withColumn("company_key", F.lit(None).cast("long")) \
        .select("date_key", "country_key", "product_key", "company_key",
                "global_semiconductor_revenue", "ai_chip_revenue", 
                "consumer_electronics_demand", "automotive_chip_demand",
                "chip_price_index", "market_growth_rate")

    f1_trendforce = df_trendforce.withColumn("full_date", standardize_to_date(F.col("date"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_product_lkp, df_trendforce.product == dim_product_lkp.product_name, "left") \
        .withColumn("country_key", F.lit(None).cast("long")) \
        .withColumn("company_key", F.lit(None).cast("long")) \
        .select("date_key", "country_key", "product_key", "company_key", "price")

    f1_yahoo = df_yahoo.withColumn("full_date", standardize_to_date(F.col("date"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_company_ticker_lkp, df_yahoo.Ticker == dim_company_ticker_lkp.ticker, "left") \
        .withColumn("country_key", F.lit(None).cast("long")) \
        .withColumn("product_key", F.lit(None).cast("long")) \
        .select("date_key", "country_key", "product_key", "company_key",
                "open", "high", "low", "close", "adj_close", "volume", 
                "daily_return", "volatility_range", "ma_5", "ma_20")

    # Union by column to create a sparse fact table as requested
    fact_market = f1_demand.unionByName(f1_market, allowMissingColumns=True) \
        .unionByName(f1_trendforce, allowMissingColumns=True) \
        .unionByName(f1_yahoo, allowMissingColumns=True) \
        .withColumn("market_fact_key", F.monotonically_increasing_id())

    # Reorder columns to place key at the front
    cols_market = ["market_fact_key", "date_key", "country_key", "product_key", "company_key"] + \
        [c for c in fact_market.columns if c not in ["market_fact_key", "date_key", "country_key", "product_key", "company_key"]]
    fact_market = fact_market.select(*cols_market)
    
    write_gold_table(fact_market, f"{GOLD_BASE_PATH}/facts/fact_semiconductor_market", "fact_semiconductor_market")


# --- FACT 2: fact_semiconductor_production ---
    logger.info("Creating fact_semiconductor_production...")
    
    # Outer join production and tech on shared granularity (year, company, node_size)
    prod_prep = df_prod.withColumn("join_node", F.col("technology_node_nm"))
    tech_prep = df_tech.withColumn("join_node", F.col("node_size_nm"))
    
    fact_prod_raw = prod_prep.join(
        tech_prep, 
        on=["year", "company", "join_node"],
        how="outer"
    ).withColumn("full_date", standardize_to_date(F.col("year")))
    
    # Alias DataFrames to break PySpark lineage ambiguity
    f_raw = fact_prod_raw.alias("f_raw")
    d_tech = dim_tech_lkp.alias("d_tech")
    d_comp = dim_company_lkp.alias("d_comp")
    d_count = dim_country_lkp.alias("d_count")
    d_date = dim_date_lkp.alias("d_date")

    fact_prod = f_raw \
        .join(d_date, F.col("f_raw.full_date") == F.col("d_date.full_date"), "left") \
        .join(d_comp, F.col("f_raw.company") == F.col("d_comp.company_name"), "left") \
        .join(d_count, F.col("f_raw.country") == F.col("d_count.country_name"), "left") \
        .join(d_tech, F.col("f_raw.join_node") == F.col("d_tech.node_size_nm"), "left") \
        .withColumn("production_fact_key", F.monotonically_increasing_id()) \
        .select("production_fact_key", "date_key", "company_key", "country_key", "technology_key",
                "production_capacity_wafers", "fab_count", "ai_chip_production", 
                "foundry_revenue_usd", "global_market_share", "transistor_density", 
                "rd_spending_usd", "patent_count", "ai_chip_performance", "energy_efficiency")
        
    write_gold_table(fact_prod, f"{GOLD_BASE_PATH}/facts/fact_semiconductor_production", "fact_semiconductor_production")

# --- FACT 3: fact_semiconductor_supply_risk ---
    logger.info("Creating fact_semiconductor_supply_risk...")

    f3_trade = df_trade.withColumn("full_date", standardize_to_date(F.col("year"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_country_lkp.alias("exp"), F.col("exporting_country") == F.col("exp.country_name"), "left") \
        .join(dim_country_lkp.alias("imp"), F.col("importing_country") == F.col("imp.country_name"), "left") \
        .select(
            "date_key", 
            F.lit(None).cast("long").alias("country_key"), 
            F.lit(None).cast("long").alias("region_key"), 
            F.col("exp.country_key").alias("export_country_key"), 
            F.col("imp.country_key").alias("import_country_key"),
            "chip_export_value_usd", "chip_import_value_usd", "trade_balance", 
            "logistics_route", "supply_chain_dependency"
        )

    f3_geo = df_geo.withColumn("full_date", standardize_to_date(F.col("year"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_country_lkp, df_geo.country == dim_country_lkp.country_name, "left") \
        .select(
            "date_key", "country_key", 
            F.lit(None).cast("long").alias("region_key"), 
            F.lit(None).cast("long").alias("export_country_key"), 
            F.lit(None).cast("long").alias("import_country_key"),
            "export_control_level", "sanctions_index", "trade_tension_level", 
            "military_tech_influence", "semiconductor_security_risk"
        )

    f3_supply = df_disruption.withColumn("full_date", standardize_to_date(F.col("year"))) \
        .join(dim_date_lkp, "full_date", "left") \
        .join(dim_region_lkp, df_disruption.region == dim_region_lkp.region_name, "left") \
        .select(
            "date_key", 
            F.lit(None).cast("long").alias("country_key"), 
            "region_key", 
            F.lit(None).cast("long").alias("export_country_key"), 
            F.lit(None).cast("long").alias("import_country_key"),
            "natural_disaster_risk", "energy_supply_risk", "water_shortage_risk", 
            "factory_shutdown_risk", "supply_disruption_index"
        )

    # Union by column for sparse risk fact
    fact_risk = f3_trade.unionByName(f3_geo, allowMissingColumns=True) \
        .unionByName(f3_supply, allowMissingColumns=True) \
        .withColumn("risk_fact_key", F.monotonically_increasing_id())

    # Reorder columns to put keys first
    cols_risk = ["risk_fact_key", "date_key", "country_key", "region_key", "export_country_key", "import_country_key"] + \
        [c for c in fact_risk.columns if c not in ["risk_fact_key", "date_key", "country_key", "region_key", "export_country_key", "import_country_key"]]
    fact_risk = fact_risk.select(*cols_risk)

    write_gold_table(fact_risk, f"{GOLD_BASE_PATH}/facts/fact_semiconductor_supply_risk", "fact_semiconductor_supply_risk")
    
    logger.info("Silver -> Gold transformation completed successfully.")

if __name__ == "__main__":
    main()