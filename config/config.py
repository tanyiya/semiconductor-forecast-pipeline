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
LOG_DIR: Path = PROJECT_ROOT / "logs"

RAW_KAGGLE_DIR: Path = RAW_DIR / "kaggle"
RAW_YAHOO_DIR: Path = RAW_DIR / "yahoo"
RAW_TRENDFORCE_DIR: Path = RAW_DIR / "trendforce"

BRONZE_KAGGLE_DIR: Path = BRONZE_DIR / "kaggle"
BRONZE_YAHOO_DIR: Path = BRONZE_DIR / "yahoo"
BRONZE_TRENDFORCE_DIR: Path = BRONZE_DIR / "trendforce"

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
    # Category pages to scrape. These are examples; update with the exact
    # TrendForce URLs relevant to your project scope (DRAM / NAND / SSD).
    target_pages: List[str] = field(
        default_factory=lambda: [
            "https://www.trendforce.com/price",
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
# Logging configuration
# --------------------------------------------------------------------------
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
