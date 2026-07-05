"""
spark_session.py

Reusable SparkSession factory for the ingestion pipeline.

Configures:
    - memory (driver / executor)
    - logging level
    - local warehouse directory
    - local execution mode (local[*])

Every loader/scraper module obtains its SparkSession through
``get_spark_session()`` so configuration lives in exactly one place.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from config.config import SPARK_CONFIG, SPARK_WAREHOUSE_DIR, ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)

_spark_session: SparkSession | None = None


def get_spark_session() -> SparkSession:
    """
    Return a singleton, pre-configured SparkSession for local execution.

    The session is created once per process and reused on subsequent
    calls, which avoids the overhead of repeatedly tearing down and
    rebuilding the JVM-backed Spark context.

    Returns
    -------
    pyspark.sql.SparkSession
    """
    global _spark_session

    if _spark_session is not None:
        return _spark_session

    ensure_directories()

    logger.info("Initialising SparkSession '%s'", SPARK_CONFIG.app_name)

    builder = (
        SparkSession.builder.appName(SPARK_CONFIG.app_name)
        .master(SPARK_CONFIG.master)
        .config("spark.driver.memory", SPARK_CONFIG.driver_memory)
        .config("spark.executor.memory", SPARK_CONFIG.executor_memory)
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG.shuffle_partitions)
        .config("spark.sql.warehouse.dir", str(SPARK_WAREHOUSE_DIR))
        # Arrow-based conversion is left OFF. It routes createDataFrame()/
        # collect()-adjacent operations through a separate Python worker
        # subprocess, which has proven unreliable on Windows (worker
        # crashes with an opaque "Python worker exited unexpectedly" /
        # EOFException under otherwise normal load). The Yahoo Finance
        # loader's pandas -> Spark conversion works fine without Arrow;
        # it's just marginally slower, which is an acceptable trade-off
        # for reliability here.
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    )

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(SPARK_CONFIG.log_level)

    logger.info(
        "SparkSession ready | master=%s | driver_memory=%s | executor_memory=%s",
        SPARK_CONFIG.master,
        SPARK_CONFIG.driver_memory,
        SPARK_CONFIG.executor_memory,
    )

    _spark_session = session
    return _spark_session


def stop_spark_session() -> None:
    """Gracefully stop the active SparkSession, if one exists."""
    global _spark_session
    if _spark_session is not None:
        logger.info("Stopping SparkSession")
        _spark_session.stop()
        _spark_session = None