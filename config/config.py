"""
config.py

Central configuration for the Data Acquisition & Ingestion stage of the
AI Semiconductor Demand Forecasting Lakehouse pipeline.

All paths, tickers, and runtime constants live here so that no module
hardcodes literals. Downstream stages (Silver/Gold, feature engineering,
XGBoost) will read from this same module later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------
# Base directories
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
 
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
BRONZE_DIR: Path = DATA_DIR / "bronze"
SILVER_DIR: Path = DATA_DIR / "silver"
LOG_DIR: Path = PROJECT_ROOT / "logs"
 
RAW_KAGGLE_DIR: Path = RAW_DIR / "kaggle"
RAW_YAHOO_DIR: Path = RAW_DIR / "yahoo"
RAW_TRENDFORCE_DIR: Path = RAW_DIR / "trendforce"
 
BRONZE_KAGGLE_DIR: Path = BRONZE_DIR / "kaggle"
BRONZE_YAHOO_DIR: Path = BRONZE_DIR / "yahoo"
BRONZE_TRENDFORCE_DIR: Path = BRONZE_DIR / "trendforce"
 
SILVER_KAGGLE_DIR: Path = SILVER_DIR / "kaggle"
SILVER_YAHOO_DIR: Path = SILVER_DIR / "yahoo"
SILVER_TRENDFORCE_DIR: Path = SILVER_DIR / "trendforce"
 
GOLD_DIR: Path = DATA_DIR / "gold"
GOLD_DIM_DATE_DIR: Path = GOLD_DIR / "dim_date"
GOLD_DIM_PRODUCT_DIR: Path = GOLD_DIR / "dim_product"
GOLD_DIM_COMPANY_DIR: Path = GOLD_DIR / "dim_company"
GOLD_DIM_COUNTRY_DIR: Path = GOLD_DIR / "dim_country"
GOLD_FACT_MARKET_PRICE_DIR: Path = GOLD_DIR / "fact_market_price"
GOLD_FACT_STOCK_MARKET_DIR: Path = GOLD_DIR / "fact_stock_market"
GOLD_FACT_PRODUCTION_DIR: Path = GOLD_DIR / "fact_production"
 
SPARK_WAREHOUSE_DIR: Path = PROJECT_ROOT / "spark-warehouse"


def ensure_directories() -> None:
    """Create every directory this project depends on, if missing."""
    for directory in (
        RAW_KAGGLE_DIR,
        RAW_YAHOO_DIR,
        RAW_TRENDFORCE_DIR,
        BRONZE_KAGGLE_DIR,
        BRONZE_YAHOO_DIR,
        BRONZE_TRENDFORCE_DIR,
        SILVER_KAGGLE_DIR,
        SILVER_YAHOO_DIR,
        SILVER_TRENDFORCE_DIR,
        LOG_DIR,
        SPARK_WAREHOUSE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Kaggle source configuration
# --------------------------------------------------------------------------
# Point this at whatever CSV(s) you downloaded from the Kaggle "Global AI
# Chip Supply Chain" dataset. Wildcards are supported (e.g. "*.csv").
KAGGLE_CSV_GLOB: str = os.environ.get("KAGGLE_CSV_GLOB", "*.csv")


# --------------------------------------------------------------------------
# Kaggle Silver-layer configuration
# --------------------------------------------------------------------------
# The Kaggle dataset's columns are dataset-dependent, so the explicit
# target schema for casting is declared here rather than hardcoded in the
# transformer. Update this mapping to match your actual downloaded CSV's
# columns (after they've been lowercased / space->underscore normalised).
# Supported type strings: "string", "double", "long", "date".
@dataclass(frozen=True)
class KaggleSilverConfig:
    date_column: str = "date"
    column_schema: dict = field(
        default_factory=lambda: {
            "date": "date",
            "company": "string",
            "chip_type": "string",
            "production_volume": "double",
            "region": "string",
            "revenue_usd": "double",
        }
    )
    # Numeric columns to check for statistical outliers (mean +/- N*stddev).
    outlier_columns: List[str] = field(
        default_factory=lambda: ["production_volume", "revenue_usd"]
    )
    outlier_std_threshold: float = 3.0


KAGGLE_SILVER_CONFIG = KaggleSilverConfig()


# --------------------------------------------------------------------------
# Yahoo Finance source configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class YahooConfig:
    tickers: List[str] = field(
        default_factory=lambda: [
            "NVDA",  # NVIDIA
            "AMD",  # AMD
            "INTC",  # Intel
            "QCOM",  # Qualcomm
            "AVGO",  # Broadcom
            "TSM",  # TSMC (ADR)
            "MU",  # Micron
        ]
    )
    period: str = "10y"   # at least 10 years of daily data
    interval: str = "1d"  # daily granularity


YAHOO_CONFIG = YahooConfig()


# --------------------------------------------------------------------------
# TrendForce source configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendForceConfig:
    base_url: str = "https://www.trendforce.com"
    # Verified public price-trend pages (checked 2026-07-01). Each page
    # stacks several sub-tables (Spot Price, Contract Price, Module Spot
    # Price, GDDR Spot Price, etc.). Member-gated sub-tables (LPDDR Spot,
    # Mobile DRAM Contract, eMMC Spot, Wafer Contract) render with no
    # visible numbers and are automatically skipped by the scraper - only
    # publicly visible data is ever collected.
    target_pages: List[str] = field(
        default_factory=lambda: [
            "https://www.trendforce.com/price",
            "https://www.trendforce.com/price/dram/dram_spot",
            "https://www.trendforce.com/price/flash/flash_spot",
        ]
    )
    request_timeout_seconds: int = 15
    request_delay_seconds: float = 1.5  # politeness delay between requests
    user_agent: str = (
        "Mozilla/5.0 (compatible; FYP-DataPipeline/1.0; "
        "+https://example.edu/fyp-contact)"
    )


TRENDFORCE_CONFIG = TrendForceConfig()


# --------------------------------------------------------------------------
# Spark configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SparkConfig:
    app_name: str = "AISemiconductorDemandForecasting-Ingestion"
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    shuffle_partitions: str = "8"  # small for local dev; raise in cluster mode
    log_level: str = "WARN"


SPARK_CONFIG = SparkConfig()


 
# --------------------------------------------------------------------------
# Gold-layer / galaxy schema configuration
# --------------------------------------------------------------------------
# Dim_Company is a CONFORMED dimension shared by Fact_StockMarket (Yahoo,
# identified by ticker) and Fact_Production (Kaggle, identified by company
# name). This mapping is what lets both facts join to the same company
# rows. Company names are matched against the Kaggle Silver "company"
# column AFTER it has been lowercased/trimmed (our KaggleTransformer
# already does this), so keep these values lowercase.
TICKER_TO_COMPANY_NAME: dict = {
    "NVDA": "nvidia",
    "AMD": "amd",
    "INTC": "intel",
    "QCOM": "qualcomm",
    "AVGO": "broadcom",
    "TSM": "tsmc",
    "MU": "micron",
}
 
 
# Fact_Production (Kaggle) target schema: Production_Capacity, Fab_Count,
# AI_Chip_Production, Foundry_Revenue, Global_Market_Share, grained at one
# row per company per calendar year.
#
# IMPORTANT: the real Kaggle "Global AI Chip Supply Chain" dataset's exact
# column names cannot be verified from this environment. The values below
# are the snake_case names implied by your Gold schema diagram - update
# them here if your actual Silver Kaggle columns are named differently.
# Any column listed here that isn't found in Silver at build time is
# logged as a warning and filled with NULL rather than crashing the run.
@dataclass(frozen=True)
class KaggleGoldConfig:
    company_column: str = "company"
    country_column: str = "country"  # adjust if your dataset calls this "country"
    date_column: str = "date"  # used to derive the calendar year grain
 
    # metric_column -> Spark aggregation function used to roll daily/raw
    # Kaggle rows up to one-row-per-company-per-year. "sum" for flow
    # metrics (production/revenue), "avg"/"max" for snapshot-style metrics
    # (capacity, fab count, market share) - adjust per column as needed.
    metric_aggregations: dict = field(
        default_factory=lambda: {
            "production_capacity_wafers": "avg",
            "fab_count": "max",
            "ai_chip_production": "sum",
            "foundry_revenue_usd": "sum",
            "global_market_share": "avg",
        }
    )
 
 
KAGGLE_GOLD_CONFIG = KaggleGoldConfig()

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"