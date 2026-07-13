"""
kaggle_bronze_silver.py

Bronze -> Silver transformer for the Kaggle "Global AI Chip Supply
Chain" dataset family.

The Kaggle download is made up of SEVEN independent CSVs (see
`ingestion/kaggle_loader.py::DATASET_FILENAMES`), each with its own
schema and each landed in its own Bronze subfolder:

    data/bronze/kaggle/1_semiconductor_production/
    data/bronze/kaggle/2_ai_hardware_demand/
    data/bronze/kaggle/3_semiconductor_trade_supply_chain/
    data/bronze/kaggle/4_technology_node_innovation/
    data/bronze/kaggle/5_geopolitical_risk_sanctions/
    data/bronze/kaggle/6_supply_chain_disruption/
    data/bronze/kaggle/semiconductor_market_economics/

None of these datasets has a true date column -- each only has a
`Year` integer column -- so unlike the previous single-dataset version
of this transformer, there is no special "cast to DateType" step.
`year` is just cast to LongType like any other declared column.

Responsibilities (per dataset)
-------------------------------
    - Standardise column names (lowercase, trim, spaces -> underscores)
    - Cast every column strictly to its declared type
    - Standardise string label values (lowercase, trim)
    - Flag and cap statistical outliers (mean +/- N*stddev) in configured
      numeric columns, computed dynamically via Spark summary stats
    - Write each dataset to its own folder under data/silver/kaggle/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType

from config.config import BRONZE_KAGGLE_DIR, SILVER_KAGGLE_DIR
from processing.base_transformer import BaseTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

_TYPE_MAP = {
    "string": StringType(),
    "double": DoubleType(),
    "long": LongType(),
}


@dataclass
class _DatasetConfig:
    """Per-dataset Silver config: column types + which columns get outlier handling."""

    column_schema: Dict[str, str]
    outlier_columns: List[str] = field(default_factory=list)
    outlier_std_threshold: float = 3.0


# Column schemas below reflect the ACTUAL columns loaded into Bronze for
# each of the 7 CSVs, after `_standardise_column_names` lowercases them
# and replaces spaces with underscores. If the upstream Kaggle CSVs ever
# change their columns, update the relevant entry here.
DATASET_SCHEMAS: Dict[str, _DatasetConfig] = {
    "1_semiconductor_production": _DatasetConfig(
        column_schema={
            "year": "long",
            "country": "string",
            "company": "string",
            "production_capacity_wafers": "double",
            "fab_count": "long",
            "technology_node_nm": "double",
            "ai_chip_production": "double",
            "foundry_revenue_usd": "double",
            "global_market_share": "double",
        },
        outlier_columns=["production_capacity_wafers", "foundry_revenue_usd", "ai_chip_production"],
    ),
    "2_ai_hardware_demand": _DatasetConfig(
        column_schema={
            "year": "long",
            "country": "string",
            "ai_gpu_demand": "double",
            "data_center_count": "long",
            "ai_compute_power": "double",
            "cloud_ai_investment": "double",
            "training_compute_flops": "double",
            "ai_model_count": "long",
        },
        outlier_columns=["ai_gpu_demand", "cloud_ai_investment", "training_compute_flops"],
    ),
    "3_semiconductor_trade_supply_chain": _DatasetConfig(
        column_schema={
            "year": "long",
            "exporting_country": "string",
            "importing_country": "string",
            "chip_export_value_usd": "double",
            "chip_import_value_usd": "double",
            "trade_balance": "double",
            "logistics_route": "string",
            "supply_chain_dependency": "double",
        },
        outlier_columns=["chip_export_value_usd", "chip_import_value_usd", "trade_balance"],
    ),
    "4_technology_node_innovation": _DatasetConfig(
        column_schema={
            "year": "long",
            "company": "string",
            "node_size_nm": "double",
            "transistor_density": "double",
            "rd_spending_usd": "double",
            "patent_count": "long",
            "ai_chip_performance": "double",
            "energy_efficiency": "double",
        },
        outlier_columns=["rd_spending_usd", "transistor_density", "ai_chip_performance"],
    ),
    "5_geopolitical_risk_sanctions": _DatasetConfig(
        column_schema={
            "year": "long",
            "country": "string",
            "export_control_level": "double",
            "sanctions_index": "double",
            "trade_tension_level": "double",
            "military_tech_influence": "double",
            "semiconductor_security_risk": "double",
        },
        outlier_columns=["sanctions_index", "trade_tension_level", "semiconductor_security_risk"],
    ),
    "6_supply_chain_disruption": _DatasetConfig(
        column_schema={
            "year": "long",
            "region": "string",
            "natural_disaster_risk": "double",
            "energy_supply_risk": "double",
            "water_shortage_risk": "double",
            "factory_shutdown_risk": "double",
            "supply_disruption_index": "double",
        },
        outlier_columns=["supply_disruption_index"],
    ),
    "7_semiconductor_market_economics": _DatasetConfig(
        column_schema={
            "year": "long",
            "global_semiconductor_revenue": "double",
            "ai_chip_revenue": "double",
            "consumer_electronics_demand": "double",
            "automotive_chip_demand": "double",
            "chip_price_index": "double",
            "market_growth_rate": "double",
        },
        outlier_columns=["global_semiconductor_revenue", "ai_chip_revenue", "chip_price_index"],
    ),
}


class KaggleTransformer(BaseTransformer):
    """Cleans, casts, and outlier-flags a single Kaggle Bronze dataset into Silver.

    Unlike the previous version, this class is instantiated ONCE PER
    DATASET (see `run_kaggle_silver_transform` below), with its own
    bronze/silver subfolder and its own column schema, since the 7
    source CSVs have entirely different columns.
    """

    source_name = "kaggle"

    def __init__(self, spark, dataset_name: str, dataset_config: _DatasetConfig, config=None):
        self.dataset_name = dataset_name
        # Override per-instance so each dataset reads/writes its own subfolder
        self.bronze_dir = BRONZE_KAGGLE_DIR / dataset_name
        self.silver_dir = SILVER_KAGGLE_DIR / dataset_name

        merged_config = {
            "column_schema": dict(dataset_config.column_schema),
            "outlier_columns": list(dataset_config.outlier_columns),
            "outlier_std_threshold": dataset_config.outlier_std_threshold,
        }
        if config:
            merged_config.update(config)
        super().__init__(spark, merged_config)

    def transform(self, df: DataFrame) -> DataFrame:
        df = self._standardise_column_names(df)
        df = self._cast_columns(df)
        df = self._standardise_string_values(df)
        df = self._flag_and_cap_outliers(df)
        return df

    # ------------------------------------------------------------------
    def _standardise_column_names(self, df: DataFrame) -> DataFrame:
        return super()._standardise_column_names(df)

    def _cast_columns(self, df: DataFrame) -> DataFrame:
        """
        Cast every column present in ``column_schema`` to its declared
        type. Columns in the schema but missing from the data are
        logged and skipped (not fatal - upstream CSV may have changed).
        Columns present in the data but absent from the schema are left
        as-is and logged, so nothing is silently dropped.

        Note: there is no dedicated date column for these datasets --
        `year` is just an integer and is cast to LongType like any other
        configured column.

        Special case: in dataset "1_semiconductor_production", the
        `technology_node_nm` column arrives with a literal "nm" unit
        suffix baked into the value (e.g. "1731nm"), which would cast to
        null against DoubleType. That suffix is stripped here before the
        generic cast loop runs so the numeric value survives the cast.
        """
        schema: Dict[str, str] = self.config["column_schema"]

        if self.dataset_name == "1_semiconductor_production" and "technology_node_nm" in df.columns:
            df = df.withColumn(
                "technology_node_nm",
                F.trim(F.regexp_replace(F.col("technology_node_nm"), "(?i)nm", "")),
            )

        missing = [c for c in schema if c not in df.columns]
        if missing:
            logger.warning(
                "[kaggle:%s] Configured schema columns not found in data (skipped): %s",
                self.dataset_name,
                missing,
            )

        unmapped = [c for c in df.columns if c not in schema]
        if unmapped:
            logger.info(
                "[kaggle:%s] Columns present but not in configured schema (left as-is): %s",
                self.dataset_name,
                unmapped,
            )

        for column, type_name in schema.items():
            if column not in df.columns:
                continue
            spark_type = _TYPE_MAP.get(type_name)
            if spark_type is None:
                logger.warning(
                    "[kaggle:%s] Unknown type '%s' for column '%s' - skipping cast",
                    self.dataset_name,
                    type_name,
                    column,
                )
                continue
            df = df.withColumn(column, F.col(column).try_cast(spark_type))

        return df

    def _standardise_string_values(self, df: DataFrame) -> DataFrame:
        """Lowercase and trim every StringType column's values."""
        string_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        for column in string_columns:
            df = df.withColumn(column, F.lower(F.trim(F.col(column))))
        if string_columns:
            logger.info(
                "[kaggle:%s] Standardised string values in columns: %s",
                self.dataset_name,
                string_columns,
            )
        return df

    def _flag_and_cap_outliers(self, df: DataFrame) -> DataFrame:
        """
        For each configured numeric column, compute mean/stddev dynamically
        via Spark aggregate stats, flag rows outside
        [mean - k*stddev, mean + k*stddev] in a new "<col>_is_outlier"
        boolean column, and cap the value itself to those bounds.
        """
        outlier_columns: List[str] = self.config["outlier_columns"]
        k: float = self.config["outlier_std_threshold"]

        for column in outlier_columns:
            if column not in df.columns:
                logger.warning(
                    "[kaggle:%s] Outlier column '%s' not present in data - skipping",
                    self.dataset_name,
                    column,
                )
                continue

            stats = df.select(
                F.mean(F.col(column)).alias("mean"), F.stddev(F.col(column)).alias("stddev")
            ).collect()[0]
            mean_val, stddev_val = stats["mean"], stats["stddev"]

            if mean_val is None or stddev_val is None or stddev_val == 0:
                logger.info(
                    "[kaggle:%s] Skipping outlier check for '%s' (insufficient variance/data)",
                    self.dataset_name,
                    column,
                )
                continue

            lower_bound = mean_val - k * stddev_val
            upper_bound = mean_val + k * stddev_val

            flag_col = f"{column}_is_outlier"
            df = df.withColumn(
                flag_col,
                (F.col(column) < lower_bound) | (F.col(column) > upper_bound),
            )

            outlier_count = df.filter(F.col(flag_col)).count()
            if outlier_count:
                logger.info(
                    "[kaggle:%s] Column '%s': flagged and capped %d outlier row(s) "
                    "(bounds=[%.2f, %.2f])",
                    self.dataset_name,
                    column,
                    outlier_count,
                    lower_bound,
                    upper_bound,
                )

            df = df.withColumn(
                column,
                F.when(F.col(column) < lower_bound, lower_bound)
                .when(F.col(column) > upper_bound, upper_bound)
                .otherwise(F.col(column)),
            )

        return df

    def apply_quality_filters(self, df: DataFrame) -> DataFrame:
        """Beyond the base all-null filter, drop strict duplicate rows."""
        df = super().apply_quality_filters(df)
        before = df.count()
        deduped = df.dropDuplicates()
        removed = before - deduped.count()
        if removed:
            logger.info(
                "[kaggle:%s] Removed %d strict duplicate row(s)", self.dataset_name, removed
            )
        return deduped


def run_kaggle_silver_transform(spark) -> Dict[str, DataFrame]:
    """Convenience function used by main.py to run this stage.

    Runs the Bronze -> Silver transform independently for each of the 7
    Kaggle datasets and returns a dict mapping dataset name to its
    resulting Silver DataFrame.
    """
    results: Dict[str, DataFrame] = {}
    for dataset_name, dataset_config in DATASET_SCHEMAS.items():
        transformer = KaggleTransformer(spark, dataset_name, dataset_config)
        results[dataset_name] = transformer.run()
    return transformer.run()