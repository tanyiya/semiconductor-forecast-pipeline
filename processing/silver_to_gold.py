"""
scripts/silver_to_gold.py

Lakehouse Pipeline: Silver to Gold Transformation
Transforms cleaned Silver parquet data into a dimensional Gold star schema.
Updated to enforce single-grain fact tables with deterministic FK mappings.
Utilizes centralized spark_session factory.

REFACTOR NOTE:
fact_semiconductor_production and fact_semiconductor_supply_risk are now built
"natural keys first": related Silver datasets are integrated on normalized natural
keys BEFORE dimension tables are joined. Surrogate keys are retrieved via those 
natural-key joins on fully assembled business entities, eliminating disconnected 
unioned row groups and post-hoc patching.
"""

import os
import sys
import logging
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

# Import centralized SparkSession manager
from spark.spark_session import get_spark_session, stop_spark_session

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

def read_silver_table(spark, path: str, recursive: bool = False):
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
    df.write.mode("overwrite").parquet(path)

    row_count = df.count()
    logger.info(f"=======================================")
    logger.info(f"QUALITY CHECK: {table_name}")
    logger.info(f"Rows: {row_count}")
    logger.info("Schema:")
    df.printSchema()
    logger.info(f"=======================================\n")

def validate_null_fks(df, table_name: str, fk_columns):
    """
    Validate foreign key completeness on a finished Gold fact table.
    Prints total rows, rows with any NULL FK among fk_columns, and the NULL %.
    """
    total_rows = df.count()

    null_condition = None
    for c in fk_columns:
        cond = F.col(c).isNull()
        null_condition = cond if null_condition is None else (null_condition | cond)

    null_rows_df = df.filter(null_condition) if null_condition is not None else df.limit(0)
    null_row_count = null_rows_df.count()
    null_pct = (null_row_count / total_rows * 100) if total_rows > 0 else 0.0

    logger.info(f"=======================================")
    logger.info(f"FK VALIDATION: {table_name}")
    logger.info(f"Checked columns: {fk_columns}")
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"NULL foreign key rows: {null_row_count}")
    logger.info(f"NULL percentage: {null_pct:.2f}%")
    if null_row_count > 0:
        logger.info("Sample rows with NULL foreign key(s):")
        for row in null_rows_df.take(5):
            logger.info(str(row))
    logger.info(f"=======================================\n")

    return {"total_rows": total_rows, "null_fk_rows": null_row_count, "null_pct": null_pct}

def validate_grain_and_sample(df, table_name: str, grain_cols: list):
    """
    Validate uniqueness on the intended business grain, print total row count,
    schema, duplicate count, and display 20 sample rows showing integrated records.
    """
    total_rows = df.count()
    distinct_grain_rows = df.dropDuplicates(grain_cols).count()
    duplicate_count = total_rows - distinct_grain_rows

    logger.info(f"=======================================")
    logger.info(f"GRAIN VALIDATION & INSPECTION: {table_name}")
    logger.info(f"Intended Business Grain: {grain_cols}")
    logger.info(f"Total rows: {total_rows}")
    logger.info(f"Duplicate rows on grain: {duplicate_count}")
    logger.info("Schema:")
    df.printSchema()
    logger.info("Sample 20 integrated records (Business columns & Surrogate keys):")
    for row in df.limit(20).collect():
        logger.info(str(row.asDict()))
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

def dict_to_spark_expr(col, mapping_dict):
    """
    Converts a Python dictionary into a JVM-native PySpark CASE WHEN expression.
    This bypasses spark.createDataFrame() and prevents Python worker EOF crashes on Windows.
    Expects `col` to already be normalized (lower/trim) by the caller, but lower-trims
    defensively here as well.
    """
    mapping_expr = F.lit(None).cast("string")
    for k, v in mapping_dict.items():
        mapping_expr = F.when(F.lower(F.trim(col)) == k.lower(), v.lower()).otherwise(mapping_expr)
    return mapping_expr

# ---------------------------------------------------------
# Dimension Builders
# ---------------------------------------------------------

def create_dim_company(df_prod, df_tech, df_yahoo):
    logger.info("Creating dim_company...")
    companies_prod = df_prod.select(F.col("company").alias("company_name"))
    companies_tech = df_tech.select(F.col("company").alias("company_name"))
    companies_yahoo = df_yahoo.select(F.col("Ticker").alias("company_name"))

    dim = companies_prod.unionByName(companies_tech).unionByName(companies_yahoo) \
        .dropna(subset=["company_name"]) \
        .dropDuplicates() \
        .withColumn("ticker", map_company_name(F.col("company_name"))) \
        .withColumn("company_key", F.monotonically_increasing_id()) \
        .select("company_key", "company_name", "ticker")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_company", "dim_company")
    return dim

def create_dim_country(df_prod, df_demand, df_geo, df_trade):
    logger.info("Creating dim_country...")
    countries = df_prod.select(F.col("country").alias("country_name")) \
        .unionByName(df_demand.select(F.col("country").alias("country_name"))) \
        .unionByName(df_geo.select(F.col("country").alias("country_name"))) \
        .unionByName(df_trade.select(F.col("exporting_country").alias("country_name"))) \
        .unionByName(df_trade.select(F.col("importing_country").alias("country_name")))

    dim = countries.dropna(subset=["country_name"]) \
        .dropDuplicates() \
        .withColumn("country_key", F.monotonically_increasing_id()) \
        .select("country_key", "country_name")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_country", "dim_country")
    return dim

def create_dim_region(df_disruption):
    logger.info("Creating dim_region...")
    dim = df_disruption.select(F.col("region").alias("region_name")) \
        .dropna(subset=["region_name"]) \
        .dropDuplicates() \
        .withColumn("region_key", F.monotonically_increasing_id()) \
        .select("region_key", "region_name")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_region", "dim_region")
    return dim

def create_dim_product(df_trendforce):
    logger.info("Creating dim_product...")
    dim = df_trendforce.select(
        F.col("product").alias("product_name"), "category", "unit"
    ).dropna(subset=["product_name"]) \
     .dropDuplicates() \
     .withColumn("product_key", F.monotonically_increasing_id()) \
     .select("product_key", "product_name", "category", "unit")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_product", "dim_product")
    return dim

def create_dim_technology(df_tech, df_prod):
    logger.info("Creating dim_technology...")
    dim = df_tech.select("node_size_nm") \
        .unionByName(df_prod.select(F.col("technology_node_nm").alias("node_size_nm"))) \
        .dropna(subset=["node_size_nm"]) \
        .dropDuplicates() \
        .withColumn("technology_key", F.monotonically_increasing_id()) \
        .select("technology_key", "node_size_nm")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_technology", "dim_technology")
    return dim

def create_dim_date(dfs_with_year, dfs_with_date):
    logger.info("Creating dim_date...")

    # Cast all to string before union to avoid INCOMPATIBLE_COLUMN_TYPE
    dates_from_years = [df.select(F.col("year").cast("string").alias("raw_date")) for df in dfs_with_year]
    dates_from_dates = [df.select(F.col("date").cast("string").alias("raw_date")) for df in dfs_with_date]

    all_dates = dates_from_years[0]
    for df in dates_from_years[1:] + dates_from_dates:
        all_dates = all_dates.unionByName(df)

    dim = all_dates.dropna(subset=["raw_date"]) \
        .select(standardize_to_date(F.col("raw_date")).alias("full_date")) \
        .dropDuplicates() \
        .withColumn("date_key", F.monotonically_increasing_id()) \
        .withColumn("day", F.dayofmonth("full_date")) \
        .withColumn("month", F.month("full_date")) \
        .withColumn("quarter", F.quarter("full_date")) \
        .withColumn("year", F.year("full_date")) \
        .withColumn("week_number", F.weekofyear("full_date")) \
        .select("date_key", "full_date", "day", "month", "quarter", "year", "week_number")

    write_gold_table(dim, f"{GOLD_BASE_PATH}/dimensions/dim_date", "dim_date")
    return dim

# ---------------------------------------------------------
# Fact Builders
# ---------------------------------------------------------

def create_fact_ai_demand(df_demand, df_market, dims):
    logger.info("Creating fact_ai_demand...")
    dim_date, dim_country = dims['date'], dims['country']

    fact_raw = df_demand.join(df_market, on="year", how="outer") \
        .withColumn("full_date", standardize_to_date(F.col("year")))

    f_raw = fact_raw.alias("f_raw")
    d_date = dim_date.alias("d_date")
    d_count = dim_country.alias("d_count")

    fact = f_raw.join(d_date, F.col("f_raw.full_date") == F.col("d_date.full_date"), "left") \
        .join(d_count, F.col("f_raw.country") == F.col("d_count.country_name"), "left") \
        .withColumn("demand_fact_key", F.monotonically_increasing_id()) \
        .select("demand_fact_key", "date_key", "country_key",
                "ai_gpu_demand", "data_center_count", "ai_compute_power", "cloud_ai_investment",
                "training_compute_flops", "ai_model_count", "global_semiconductor_revenue",
                "ai_chip_revenue", "consumer_electronics_demand", "automotive_chip_demand",
                "chip_price_index", "market_growth_rate")

    write_gold_table(fact, f"{GOLD_BASE_PATH}/facts/fact_ai_demand", "fact_ai_demand")


def create_fact_market_price(df_trendforce, dims):
    logger.info("Creating fact_chip_market_price...")
    dim_date, dim_product = dims['date'], dims['product']

    fact_raw = df_trendforce.withColumn("full_date", standardize_to_date(F.col("date")))

    f_raw = fact_raw.alias("f_raw")
    d_date = dim_date.alias("d_date")
    d_prod = dim_product.alias("d_prod")

    fact = f_raw.join(d_date, F.col("f_raw.full_date") == F.col("d_date.full_date"), "left") \
        .join(d_prod, F.col("f_raw.product") == F.col("d_prod.product_name"), "left") \
        .withColumn("price_fact_key", F.monotonically_increasing_id()) \
        .select("price_fact_key", "date_key", "product_key", "price")

    write_gold_table(fact, f"{GOLD_BASE_PATH}/facts/fact_chip_market_price", "fact_chip_market_price")


def create_fact_stock_market(df_yahoo, dims):
    logger.info("Creating fact_stock_market...")
    dim_date, dim_comp = dims['date'], dims['company']

    fact_raw = df_yahoo.withColumn("full_date", standardize_to_date(F.col("date")))

    f_raw = fact_raw.alias("f_raw")
    d_date = dim_date.alias("d_date")
    d_comp = dim_comp.filter(F.col("ticker").isNotNull()).alias("d_comp")

    fact = f_raw.join(d_date, F.col("f_raw.full_date") == F.col("d_date.full_date"), "left") \
        .join(d_comp, F.col("f_raw.Ticker") == F.col("d_comp.ticker"), "left") \
        .withColumn("stock_fact_key", F.monotonically_increasing_id()) \
        .select("stock_fact_key", "date_key", "company_key",
                "open", "high", "low", "close", "adj_close", "volume",
                "daily_return", "volatility_range", "ma_5", "ma_20")

    write_gold_table(fact, f"{GOLD_BASE_PATH}/facts/fact_stock_market", "fact_stock_market")


def create_fact_production(spark, df_prod, df_tech, dims):
    """
    fact_semiconductor_production -- built natural-keys-first.
    Grain: company + country + technology node + year.

    Pipeline:
        1. Base dataset: 1_semiconductor_production (defines grain).
        2. Normalize natural keys (company, country, node_size) in both datasets.
        3. Left join 4_technology_node_innovation onto production using (company, year, node_size).
        4. Lookup surrogate keys from dimensions using normalized natural keys.
        5. Validate uniqueness on intended grain and inspect integrated sample rows.
        6. Assemble and write final Gold table.
    """
    logger.info("Creating fact_semiconductor_production...")
    dim_date, dim_comp, dim_country, dim_tech = dims['date'], dims['company'], dims['country'], dims['tech']

    # -----------------------------------------------------
    # Step 1: Normalize natural keys BEFORE any joins
    # -----------------------------------------------------
    prod_clean = df_prod.select(
        F.col("year").alias("prod_year"),
        F.col("company").alias("company_name"),
        F.col("country").alias("country_name"),
        F.col("technology_node_nm").alias("node_size_nm"),
        F.lower(F.trim(F.col("company"))).alias("prod_company_norm"),
        F.lower(F.trim(F.col("country"))).alias("prod_country_norm"),
        F.col("technology_node_nm").cast("double").alias("prod_node_norm"),
        "production_capacity_wafers", "fab_count", "ai_chip_production",
        "foundry_revenue_usd", "global_market_share"
    )

    tech_clean = df_tech.select(
        F.col("year").alias("tech_year"),
        F.lower(F.trim(F.col("company"))).alias("tech_company_norm"),
        F.col("node_size_nm").cast("double").alias("tech_node_norm"),
        "transistor_density", "rd_spending_usd", "patent_count",
        "ai_chip_performance", "energy_efficiency"
    )

    # -----------------------------------------------------
    # Step 2: Integrate datasets on natural keys (Production is driving table)
    # -----------------------------------------------------
    joined = prod_clean.join(
        tech_clean,
        (prod_clean.prod_company_norm == tech_clean.tech_company_norm) &
        (prod_clean.prod_year == tech_clean.tech_year) &
        (prod_clean.prod_node_norm == tech_clean.tech_node_norm),
        "left"
    ).withColumn("full_date", standardize_to_date(F.col("prod_year")))

    # -----------------------------------------------------
    # Step 3: Prepare dimension lookups on natural keys
    # -----------------------------------------------------
    d_comp = dim_comp.select(
        F.lower(F.trim(F.col("company_name"))).alias("dim_company_norm"),
        "company_key"
    ).dropDuplicates(["dim_company_norm"])

    d_country = dim_country.select(
        F.lower(F.trim(F.col("country_name"))).alias("dim_country_norm"),
        "country_key"
    ).dropDuplicates(["dim_country_norm"])

    d_tech = dim_tech.select(
        F.col("node_size_nm").cast("double").alias("dim_node_norm"),
        "technology_key"
    ).dropDuplicates(["dim_node_norm"])

    d_date = dim_date.select("full_date", "date_key").dropDuplicates(["full_date"])

    # -----------------------------------------------------
    # Step 4: Retrieve surrogate keys via natural-key joins
    # -----------------------------------------------------
    fact_integrated = joined.join(d_comp, joined.prod_company_norm == d_comp.dim_company_norm, "left") \
        .join(d_country, joined.prod_country_norm == d_country.dim_country_norm, "left") \
        .join(d_tech, joined.prod_node_norm == d_tech.dim_node_norm, "left") \
        .join(d_date, joined.full_date == d_date.full_date, "left") \
        .withColumn("production_fact_key", F.monotonically_increasing_id())

    # Keep business columns alongside surrogate keys for validation & inspection
    fact_integrated = fact_integrated.select(
        "production_fact_key", "date_key", "company_key", "country_key", "technology_key",
        F.col("prod_year").alias("year"),
        "company_name", "country_name", "node_size_nm",
        "production_capacity_wafers", "fab_count", "ai_chip_production", "foundry_revenue_usd",
        "global_market_share", "transistor_density", "rd_spending_usd", "patent_count",
        "ai_chip_performance", "energy_efficiency"
    )

    # -----------------------------------------------------
    # Step 5: Validate Grain and Inspect Sample Records
    # -----------------------------------------------------
    validate_grain_and_sample(
        fact_integrated,
        "fact_semiconductor_production",
        ["year", "company_name", "country_name"]
    )

    # -----------------------------------------------------
    # Step 6: Assemble and Write Final Gold Fact
    # -----------------------------------------------------
    fact_final = fact_integrated.select(
        "production_fact_key", "date_key", "company_key", "country_key", "technology_key",
        "year", "company_name", "country_name", "node_size_nm",
        "production_capacity_wafers", "fab_count", "ai_chip_production", "foundry_revenue_usd",
        "global_market_share", "transistor_density", "rd_spending_usd", "patent_count",
        "ai_chip_performance", "energy_efficiency"
    )

    write_gold_table(fact_final, f"{GOLD_BASE_PATH}/facts/fact_semiconductor_production", "fact_semiconductor_production")

    validate_null_fks(
        fact_final,
        "fact_semiconductor_production",
        ["company_key", "country_key", "technology_key"]
    )

    return fact_final


def create_fact_supply_risk(spark, df_trade, df_geo, df_disruption, dims):
    """
    fact_semiconductor_supply_risk -- built natural-keys-first.
    Grain: year + country.

    Pipeline:
        1. Base dataset: 5_geopolitical_risk_sanctions (defines Year, Country grain).
        2. Map country to region for every geopolitical record using deterministic rules.
        3. Left join 6_supply_chain_disruption (Year, Region) onto geopolitical records.
        4. Lookup surrogate keys from dim_country, dim_region, and dim_date.
        5. Validate uniqueness on intended grain and inspect integrated sample rows.
        6. Assemble and write final Gold table (excluding incompatible trade grain).
    """
    logger.info("Creating fact_semiconductor_supply_risk...")
    dim_date, dim_country, dim_region = dims['date'], dims['country'], dims['region']

    COUNTRY_REGION_MAPPING = {
    "japan": "east asia",
    "taiwan": "east asia",
    "china": "east asia",
    "uk": "europe",
    "switzerland": "europe",
    "germany": "europe",
    "israel": "middle east",
    "netherlands": "europe",
    "usa": "north america",
    "south korea": "east asia",
    "romania": "europe",
    "brazil": "south america",
    "vietnam": "southeast asia",
    "austria": "europe",
    "canada": "north america",
    "saudi arabia": "middle east",
    "finland": "europe",
    "denmark": "europe",
    "mexico": "north america",
    "norway": "europe",
    "sweden": "europe",
    "australia": "oceania",
    "czech republic": "europe",
    "singapore": "southeast asia",
    "belgium": "europe",
    "thailand": "southeast asia",
    "chile": "south america",
    "hungary": "europe",
    "indonesia": "southeast asia",
    "spain": "europe",
    "poland": "europe",
    "uae": "middle east",
    "philippines": "southeast asia",
    "argentina": "south america",
    "malaysia": "southeast asia",
    "south africa": "africa",
    "india": "south asia",
    "italy": "europe",
    "ireland": "europe",
    "france": "europe",}
    # -----------------------------------------------------
    # Step 1: Normalize natural keys per source, BEFORE any joins
    # -----------------------------------------------------
    geo_clean = df_geo.select(
        F.col("year").alias("geo_year"),
        F.col("country").alias("country_name"),
        F.lower(F.trim(F.col("country"))).alias("geo_country_norm"),
        "export_control_level", "sanctions_index", "trade_tension_level",
        "military_tech_influence", "semiconductor_security_risk"
    )

    disruption_clean = df_disruption.select(
        F.col("year").alias("dis_year"),
        F.lower(F.trim(F.col("region"))).alias("dis_region_norm"),
        "natural_disaster_risk", "energy_supply_risk", "water_shortage_risk",
        "factory_shutdown_risk", "supply_disruption_index"
    )

    # -----------------------------------------------------
    # Step 2: Map country to region for every geopolitical record
    # -----------------------------------------------------
    geo_clean = geo_clean.withColumn(
        "mapped_region_norm", dict_to_spark_expr(F.col("geo_country_norm"), COUNTRY_REGION_MAPPING)
    )

    # -----------------------------------------------------
    # Step 3: Integrate datasets on natural keys (Geopolitical is driving table)
    # -----------------------------------------------------
    joined = geo_clean.join(
        disruption_clean,
        (geo_clean.geo_year == disruption_clean.dis_year) &
        (geo_clean.mapped_region_norm == disruption_clean.dis_region_norm),
        "left"
    ).withColumn("full_date", standardize_to_date(F.col("geo_year")))

    # -----------------------------------------------------
    # Step 4: Dimension lookups on natural keys
    # -----------------------------------------------------
    d_country = dim_country.select(
        F.lower(F.trim(F.col("country_name"))).alias("dim_country_norm"),
        "country_key"
    ).dropDuplicates(["dim_country_norm"])

    d_region = dim_region.select(
        F.lower(F.trim(F.col("region_name"))).alias("dim_region_norm"),
        "region_key"
    ).dropDuplicates(["dim_region_norm"])

    d_date = dim_date.select("full_date", "date_key").dropDuplicates(["full_date"])

    # -----------------------------------------------------
    # Step 5: Retrieve surrogate keys via natural-key joins
    # -----------------------------------------------------
    fact_integrated = joined.join(d_country, joined.geo_country_norm == d_country.dim_country_norm, "left") \
        .join(d_region, joined.mapped_region_norm == d_region.dim_region_norm, "left") \
        .join(d_date, joined.full_date == d_date.full_date, "left") \
        .withColumn("risk_fact_key", F.monotonically_increasing_id())

    # Keep business columns alongside surrogate keys for validation & inspection
    fact_integrated = fact_integrated.select(
        "risk_fact_key", "date_key", "country_key", "region_key",
        F.col("geo_year").alias("year"),
        "country_name",
        F.col("mapped_region_norm").alias("region_name"),
        "export_control_level", "sanctions_index", "trade_tension_level",
        "military_tech_influence", "semiconductor_security_risk",
        "natural_disaster_risk", "energy_supply_risk", "water_shortage_risk",
        "factory_shutdown_risk", "supply_disruption_index"
    )

    # -----------------------------------------------------
    # Step 6: Validate Grain and Inspect Sample Records
    # -----------------------------------------------------
    validate_grain_and_sample(
        fact_integrated,
        "fact_semiconductor_supply_risk",
        ["year", "country_name"]
    )

    # -----------------------------------------------------
    # Step 7: Assemble and Write Final Gold Fact
    # Note: Trade data (export/import grain) is excluded here to preserve strict single-grain (Year, Country) design.
    # -----------------------------------------------------
    fact_final = fact_integrated.select(
        "risk_fact_key", "date_key", "country_key", "region_key",
        "year", "country_name", "region_name",
        "export_control_level", "sanctions_index", "trade_tension_level",
        "military_tech_influence", "semiconductor_security_risk",
        "natural_disaster_risk", "energy_supply_risk", "water_shortage_risk",
        "factory_shutdown_risk", "supply_disruption_index"
    )

    write_gold_table(fact_final, f"{GOLD_BASE_PATH}/facts/fact_semiconductor_supply_risk", "fact_semiconductor_supply_risk")

    validate_null_fks(
        fact_final,
        "fact_semiconductor_supply_risk",
        ["country_key", "region_key"]
    )

    return fact_final


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    # Fix for Windows PySpark worker issue when .collect() runs
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Obtain session from centralized factory
    spark = get_spark_session()
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
    df_yahoo = read_silver_table(spark, f"{SILVER_BASE_PATH}/yahoo", recursive=False)

    # 2. Build Dimensions
    logger.info("Building Dimensions...")
    dim_company = create_dim_company(df_prod, df_tech, df_yahoo)
    dim_country = create_dim_country(df_prod, df_demand, df_geo, df_trade)
    dim_region = create_dim_region(df_disruption)
    dim_product = create_dim_product(df_trendforce)
    dim_tech = create_dim_technology(df_tech, df_prod)

    dfs_with_year = [df_prod, df_demand, df_trade, df_tech, df_geo, df_disruption, df_market]
    dfs_with_date = [df_trendforce, df_yahoo]
    dim_date = create_dim_date(dfs_with_year, dfs_with_date)

    # Package dimensions for easy access in fact builders
    dims = {
        'date': dim_date.select("full_date", "date_key"),
        'company': dim_company.select("company_name", "ticker", "company_key"),
        'country': dim_country.select("country_name", "country_key"),
        'region': dim_region.select("region_name", "region_key"),
        'product': dim_product.select("product_name", "product_key"),
        'tech': dim_tech.select("node_size_nm", "technology_key")
    }

    # 3. Build Facts
    logger.info("Building Fact Tables...")
    create_fact_ai_demand(df_demand, df_market, dims)
    create_fact_production(spark, df_prod, df_tech, dims)
    create_fact_market_price(df_trendforce, dims)
    create_fact_stock_market(df_yahoo, dims)
    create_fact_supply_risk(spark, df_trade, df_geo, df_disruption, dims)

    logger.info("Silver -> Gold transformation completed successfully.")

    # Gracefully tear down using the factory stop function
    stop_spark_session()

if __name__ == "__main__":
    main()