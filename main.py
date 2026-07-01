"""
main.py

Orchestrates the Data Acquisition & Ingestion stage of the AI
Semiconductor Demand Forecasting Lakehouse pipeline.

Pipeline scope (this stage only):
    Kaggle CSV  ─┐
    Yahoo Finance├─► Spark ingestion ─► Validation ─► Bronze Layer (Parquet)
    TrendForce  ─┘

Run with:
    python main.py
    python main.py --sources kaggle yahoo
    python main.py --sources trendforce
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import os

# 1. Point PySpark to your stable Python 3.12 environment
py_312_path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
os.environ['PYSPARK_PYTHON'] = py_312_path
os.environ['PYSPARK_DRIVER_PYTHON'] = py_312_path

# 2. Tell Spark where winutils.exe and hadoop.dll live on your D: drive
os.environ['HADOOP_HOME'] = r"D:\01_Bomi\01_ProgramFiles\hadoop"
os.environ['PATH'] = os.environ['PATH'] + r";D:\01_Bomi\01_ProgramFiles\hadoop\bin"

from config.config import ensure_directories
from ingestion.kaggle_loader import run_kaggle_ingestion
from ingestion.trendforce_scraper import run_trendforce_ingestion
from ingestion.yahoo_loader import run_yahoo_ingestion
from spark.spark_session import get_spark_session, stop_spark_session
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)

ALL_SOURCES = ["kaggle", "yahoo", "trendforce"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Data Acquisition & Ingestion stage."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=ALL_SOURCES,
        default=ALL_SOURCES,
        help="Which sources to ingest. Defaults to all three.",
    )
    parser.add_argument(
        "--kaggle-date-column",
        default=None,
        help="Optional name of a date column in the Kaggle dataset to validate.",
    )
    return parser.parse_args()


@log_execution_time("Full Ingestion Pipeline")
def run_pipeline(sources: List[str], kaggle_date_column: str | None) -> None:
    ensure_directories()
    spark = get_spark_session()
    results = {}

    try:
        if "kaggle" in sources:
            try:
                results["kaggle"] = run_kaggle_ingestion(
                    spark, expected_date_column=kaggle_date_column
                )
            except FileNotFoundError as exc:
                logger.error("Kaggle ingestion skipped: %s", exc)

        if "yahoo" in sources:
            try:
                results["yahoo"] = run_yahoo_ingestion(spark)
            except Exception:  # noqa: BLE001
                logger.exception("Yahoo Finance ingestion failed")

        if "trendforce" in sources:
            try:
                results["trendforce"] = run_trendforce_ingestion(spark)
            except Exception:  # noqa: BLE001
                logger.exception("TrendForce ingestion failed")

        logger.info("Ingestion stage complete. Sources processed: %s", list(results.keys()))

    finally:
        stop_spark_session()


def main() -> None:
    args = parse_args()
    logger.info("Starting ingestion pipeline for sources: %s", args.sources)
    try:
        run_pipeline(args.sources, args.kaggle_date_column)
    except Exception:
        logger.exception("Pipeline run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
