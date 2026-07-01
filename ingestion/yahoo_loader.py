"""
yahoo_loader.py

Downloads historical daily stock data for semiconductor companies via
yfinance, standardises the schema, converts to Spark DataFrames, and
persists to the Bronze layer in Parquet format.

yfinance only returns pandas DataFrames, so pandas is used transiently
here (as explicitly permitted by the project spec) purely as a bridge
before converting into Spark.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import yfinance as yf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from config.config import BRONZE_YAHOO_DIR, YAHOO_CONFIG, YahooConfig
from ingestion.validator import validate_dataframe
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)

STANDARD_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

EXPECTED_SCHEMA: dict = {
    "Date": StringType(),  # cast from date after ingestion; kept as string pre-validation
    "Ticker": StringType(),
    "Open": DoubleType(),
    "High": DoubleType(),
    "Low": DoubleType(),
    "Close": DoubleType(),
    "Adj Close": DoubleType(),
    "Volume": LongType(),
}


class YahooFinanceLoader:
    """Encapsulates the Yahoo Finance ingestion workflow."""

    def __init__(self, spark: SparkSession, config: YahooConfig = YAHOO_CONFIG):
        self.spark = spark
        self.config = config

    @log_execution_time("Yahoo Finance Download")
    def _download_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single ticker via yfinance.
        Returns None (and logs a warning) if no data was returned.
        """
        logger.info(
            "Downloading %s | period=%s | interval=%s",
            ticker,
            self.config.period,
            self.config.interval,
        )
        history = yf.Ticker(ticker).history(
            period=self.config.period,
            interval=self.config.interval,
            auto_adjust=False,
        )

        if history.empty:
            logger.warning("No data returned for ticker '%s'", ticker)
            return None

        history = history.reset_index()
        history["Ticker"] = ticker

        # yfinance sometimes omits "Adj Close" depending on auto_adjust;
        # guard against that for schema consistency.
        if "Adj Close" not in history.columns:
            history["Adj Close"] = history["Close"]

        history = history.rename(columns={"index": "Date"})
        history = history[STANDARD_COLUMNS]
        logger.info("Downloaded %d rows for %s", len(history), ticker)
        return history

    def download_all(self) -> pd.DataFrame:
        """
        Iterate through every configured ticker, downloading and
        concatenating their histories into a single pandas DataFrame.
        """
        frames: List[pd.DataFrame] = []
        for ticker in self.config.tickers:
            df = self._download_ticker(ticker)
            if df is not None:
                frames.append(df)

        if not frames:
            raise RuntimeError(
                "Yahoo Finance download returned no data for any configured "
                "ticker. Check network access and ticker symbols."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Combined Yahoo Finance data: %d rows across %d tickers",
            len(combined),
            len(frames),
        )
        return combined
        
    """
    def to_spark(self, pandas_df: pd.DataFrame) -> DataFrame:
            # 1. Clean the pandas data explicitly
            pandas_df = pandas_df.copy()
            
            # Convert Datetime to string for safe transfer
            pandas_df["Date"] = pandas_df["Date"].dt.strftime("%Y-%m-%d")
            
            # Ensure all types are explicitly matched to standard Python types
            pandas_df = pandas_df.astype({
                "Ticker": "str",
                "Open": "float64",
                "High": "float64",
                "Low": "float64",
                "Close": "float64",
                "Adj Close": "float64",
                "Volume": "int64"
            })

            # 2. Use the schema defined at the top of your file for the handoff
            # This prevents Spark from trying to 'guess' types, which often triggers the worker crash
            from pyspark.sql.types import StructType
            
            # Reconstruct the schema for the createDataFrame call
            schema = StructType([
                StructField("Date", StringType(), True),
                StructField("Ticker", StringType(), True),
                StructField("Open", DoubleType(), True),
                StructField("High", DoubleType(), True),
                StructField("Low", DoubleType(), True),
                StructField("Close", DoubleType(), True),
                StructField("Adj Close", DoubleType(), True),
                StructField("Volume", LongType(), True)
            ])

            # 3. Create DataFrame with explicit schema
            return self.spark.createDataFrame(pandas_df, schema=schema)
    """
    def to_spark(self, pandas_df: pd.DataFrame) -> DataFrame:
        """Convert the standardised pandas DataFrame into a Spark DataFrame safely."""
        # Clean the pandas data explicitly
        df_clean = pandas_df.copy()
        df_clean["Date"] = df_clean["Date"].astype(str)
        
        df_clean = df_clean.astype({
            "Ticker": "str",
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Adj Close": "float64",
            "Volume": "int64"
        })
        
        # Write out the safe temp file
        temp_path = "temp_yahoo_data.parquet"
        df_clean.to_parquet(temp_path, index=False, engine="pyarrow", coerce_timestamps="ms")
        
        # Read into Spark (Do NOT use os.remove here yet!)
        return self.spark.read.parquet(temp_path)

    @log_execution_time("Yahoo Bronze Write")
    def write_bronze(self, df: DataFrame, output_dir=BRONZE_YAHOO_DIR) -> None:
        """Persist the Spark DataFrame to the Bronze layer as Parquet, partitioned by Ticker."""
        #print("BEFORE WRITE CHECK", df.count())
        
        output_dir.mkdir(parents=True, exist_ok=True)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .partitionBy("Ticker")
            .parquet(str(output_dir))
        )
        logger.info("Yahoo Finance Bronze data written to %s", output_dir)

    @log_execution_time("Yahoo Finance Ingestion Pipeline")
    def run(self) -> DataFrame:
        """
        Execute the full Yahoo Finance ingestion workflow:
        download -> standardise -> convert to Spark -> validate -> write to Bronze.
        """
        pandas_df = self.download_all()
        spark_df = self.to_spark(pandas_df)

        validate_dataframe(
            spark_df,
            source_name="yahoo",
            expected_columns=STANDARD_COLUMNS,
            expected_types=EXPECTED_SCHEMA,
            date_column="Date",
        )

        self.write_bronze(spark_df)
        return spark_df


def run_yahoo_ingestion(spark: SparkSession) -> DataFrame:
    """Convenience function used by main.py to run this stage."""
    loader = YahooFinanceLoader(spark)
    return loader.run()

def run(self) -> DataFrame:
        """
        Execute the full Yahoo Finance ingestion workflow:
        download -> standardise -> convert to Spark -> validate -> write to Bronze.
        """
        import os
        
        pandas_df = self.download_all()
        spark_df = self.to_spark(pandas_df)

        validate_dataframe(
            spark_df,
            source_name="yahoo",
            expected_columns=STANDARD_COLUMNS,
            expected_types=EXPECTED_SCHEMA,
            date_column="Date",
        )

        # 1. Complete the real persistent Bronze write while the temp file is active
        self.write_bronze(spark_df)
        
        # 2. Safely delete the transitional file after all actions are finished
        temp_path = "temp_yahoo_data.parquet"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning("Could not clean up temporary file %s: %s", temp_path, e)
                
        return spark_df