import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix Windows PySpark binary crash by binding environment variables before Spark boots
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARK_DIR = PROJECT_ROOT / "spark"

sys.path.append(str(PROJECT_ROOT))
from spark.spark_session import get_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_semiconductor_forecast")

GOLD_PATH = PROJECT_ROOT / "data" / "gold"
DIM_PATH = GOLD_PATH / "dimensions"
FACT_PATH = GOLD_PATH / "facts"

FORECAST_OUTPUT_PATH = GOLD_PATH / "forecast"
EVALUATION_PATH = FORECAST_OUTPUT_PATH / "evaluation"
FEATURE_IMPORTANCE_PATH = FORECAST_OUTPUT_PATH / "feature_importance"
FORECAST_COUNTRY_PATH = FORECAST_OUTPUT_PATH / "forecast_country"
FORECAST_GLOBAL_PATH = FORECAST_OUTPUT_PATH / "forecast_global"

FORECAST_YEARS: List[int] = [2026, 2027, 2028, 2029, 2030]

TARGET_COLUMNS: List[str] = ["ai_gpu_demand", "ai_chip_revenue"]

# Train on YoY growth rates to allow native time-series compounding without leaf ceilings
GROWTH_TARGET_COLUMNS: List[str] = [f"{col}_yoy_growth" for col in TARGET_COLUMNS]

DEMAND_FEATURE_COLUMNS: List[str] = [
    "data_center_count",
    "ai_compute_power",
    "cloud_ai_investment",
    "training_compute_flops",
    "ai_model_count",
    "global_semiconductor_revenue",
    "consumer_electronics_demand",
    "automotive_chip_demand",
    "chip_price_index",
    "market_growth_rate",
]

PRODUCTION_FEATURE_COLUMNS: List[str] = [
    "production_capacity_wafers",
    "fab_count",
    "ai_chip_production",
    "foundry_revenue_usd",
    "global_market_share",
    "transistor_density",
    "rd_spending_usd",
    "patent_count",
    "ai_chip_performance",
    "energy_efficiency",
]

LAG_FEATURE_COLUMNS: List[str] = [f"{col}_lag_1" for col in TARGET_COLUMNS]

EXOGENOUS_FEATURES: List[str] = DEMAND_FEATURE_COLUMNS + PRODUCTION_FEATURE_COLUMNS

# Include year and autoregressive lags so the tree can partition on temporal progression
ALL_FEATURE_COLUMNS: List[str] = (
    ["year"] + EXOGENOUS_FEATURES + LAG_FEATURE_COLUMNS
)

XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}


def load_gold_tables(spark: SparkSession) -> Dict[str, DataFrame]:
    logger.info("Loading gold layer tables from %s", GOLD_PATH)
    tables = {
        "dim_company": spark.read.parquet(str(DIM_PATH / "dim_company")),
        "dim_country": spark.read.parquet(str(DIM_PATH / "dim_country")),
        "dim_date": spark.read.parquet(str(DIM_PATH / "dim_date")),
        "dim_product": spark.read.parquet(str(DIM_PATH / "dim_product")),
        "dim_region": spark.read.parquet(str(DIM_PATH / "dim_region")),
        "dim_technology": spark.read.parquet(str(DIM_PATH / "dim_technology")),
        "fact_ai_demand": spark.read.parquet(str(FACT_PATH / "fact_ai_demand")),
        "fact_chip_market_price": spark.read.parquet(
            str(FACT_PATH / "fact_chip_market_price")
        ),
        "fact_semiconductor_production": spark.read.parquet(
            str(FACT_PATH / "fact_semiconductor_production")
        ),
        "fact_semiconductor_supply_risk": spark.read.parquet(
            str(FACT_PATH / "fact_semiconductor_supply_risk")
        ),
        "fact_stock_market": spark.read.parquet(str(FACT_PATH / "fact_stock_market")),
    }
    return tables


def build_training_dataset(tables: Dict[str, DataFrame]) -> DataFrame:
    logger.info("Building training dataset from fact_ai_demand and fact_semiconductor_production")

    dim_date = tables["dim_date"].select("date_key", "year")
    dim_country = tables["dim_country"].select("country_key", "country_name")

    demand = tables["fact_ai_demand"].join(dim_date, on="date_key", how="left")

    production_agg = (
        tables["fact_semiconductor_production"]
        .groupBy("country_key", "year")
        .agg(
            F.avg("production_capacity_wafers").alias("production_capacity_wafers"),
            F.avg("fab_count").alias("fab_count"),
            F.avg("ai_chip_production").alias("ai_chip_production"),
            F.avg("foundry_revenue_usd").alias("foundry_revenue_usd"),
            F.avg("global_market_share").alias("global_market_share"),
            F.avg("transistor_density").alias("transistor_density"),
            F.avg("rd_spending_usd").alias("rd_spending_usd"),
            F.avg("patent_count").alias("patent_count"),
            F.avg("ai_chip_performance").alias("ai_chip_performance"),
            F.avg("energy_efficiency").alias("energy_efficiency"),
        )
    )

    joined = demand.join(production_agg, on=["country_key", "year"], how="left")
    joined = joined.join(dim_country, on="country_key", how="left")

    select_columns = (
        ["date_key", "country_key", "country_name", "year"]
        + DEMAND_FEATURE_COLUMNS
        + PRODUCTION_FEATURE_COLUMNS
        + TARGET_COLUMNS
    )
    select_columns = list(dict.fromkeys(select_columns))

    return joined.select(*select_columns)


def spark_to_pandas(df: DataFrame) -> pd.DataFrame:
    logger.info("Converting Spark DataFrame to Pandas for model training")
    return df.toPandas()


def preprocess_features(pdf: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing features and transforming targets to YoY growth rates")
    pdf = pdf.copy()

    for col in EXOGENOUS_FEATURES:
        if col not in pdf.columns:
            pdf[col] = np.nan

    numeric_columns = EXOGENOUS_FEATURES + TARGET_COLUMNS + ["year"]
    for col in numeric_columns:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    group_cols = ["country_key"]
    
    with np.errstate(invalid="ignore"):
        for col in EXOGENOUS_FEATURES:
            pdf[col] = pdf.groupby(group_cols)[col].transform(
                lambda s: s.fillna(s.median())
            )
            pdf[col] = pdf[col].fillna(pdf[col].median())
            pdf[col] = pdf[col].fillna(0.0)

    pdf = pdf.sort_values(by=["country_key", "year"]).reset_index(drop=True)

    # Engineer lags and YoY growth rates so the model learns relative momentum
    for target in TARGET_COLUMNS:
        lag_col = f"{target}_lag_1"
        pdf[lag_col] = pdf.groupby("country_key")[target].shift(1)
        
        pdf[lag_col] = pdf.groupby("country_key")[lag_col].transform(
            lambda s: s.bfill().ffill().fillna(0.0)
        )
        
        growth_col = f"{target}_yoy_growth"
        pdf[growth_col] = np.where(
            pdf[lag_col] != 0,
            (pdf[target] - pdf[lag_col]) / pdf[lag_col],
            0.0
        )
        # Clip extreme historical anomalies (-50% to +150%) to stabilize tree split thresholds
        pdf[growth_col] = pd.Series(pdf[growth_col]).clip(lower=-0.5, upper=1.5).fillna(0.0)

    pdf["country_key"] = pdf["country_key"].astype("category").cat.codes
    pdf = pdf.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)

    return pdf


def chronological_split(
    pdf: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years_sorted = sorted(pdf["year"].unique())
    split_index = int(len(years_sorted) * (1 - test_fraction))
    split_index = max(1, min(split_index, len(years_sorted) - 1))
    train_years = years_sorted[:split_index]
    test_years = years_sorted[split_index:]

    train_df = pdf[pdf["year"].isin(train_years)].reset_index(drop=True)
    test_df = pdf[pdf["year"].isin(test_years)].reset_index(drop=True)

    logger.info(
        "Chronological split -> train years: %s, test years: %s",
        train_years,
        test_years,
    )
    return train_df, test_df


def train_xgboost_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame]:
    logger.info("Training XGBoost model for target: %s", target_column)

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    non_zero_mask = y_test != 0
    if non_zero_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (y_test[non_zero_mask] - y_pred[non_zero_mask])
                    / y_test[non_zero_mask]
                )
            ) * 100
        )
    else:
        mape = float("nan")

    metrics_df = pd.DataFrame(
        [
            {
                "target_metric": str(target_column),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
            }
        ]
    )

    logger.info(
        "Model evaluation for %s -> MAE: %.4f, RMSE: %.4f, R2: %.4f, MAPE: %.4f",
        target_column,
        mae,
        rmse,
        r2,
        mape,
    )

    return model, metrics_df


def extract_feature_importance(
    model: xgb.XGBRegressor, feature_columns: List[str], target_column: str
) -> pd.DataFrame:
    logger.info("Extracting feature importance for target: %s", target_column)

    importances = [float(val) for val in model.feature_importances_]
    importance_df = pd.DataFrame(
        {
            "feature_name": [str(c) for c in feature_columns],
            "importance": importances,
            "target_metric": str(target_column),
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df


def build_latest_country_snapshot(pdf: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building latest per-country feature snapshot for iterative forecasting")

    latest_year = pdf["year"].max()
    snapshot = pdf[pdf["year"] == latest_year].copy()
    snapshot = snapshot.drop_duplicates(subset=["country_key"]).reset_index(drop=True)

    for target in TARGET_COLUMNS:
        snapshot[f"{target}_lag_1"] = snapshot[target]

    return snapshot


def iterative_forecast(
    model_gpu: xgb.XGBRegressor,
    model_revenue: xgb.XGBRegressor,
    snapshot: pd.DataFrame,
    country_lookup: pd.DataFrame,
    feature_columns: List[str],
    forecast_years: List[int],
) -> pd.DataFrame:
    logger.info("Generating dynamic iterative forecasts through year %d without artificial multipliers", forecast_years[-1])

    working = snapshot.copy()
    results: List[pd.DataFrame] = []

    for year in forecast_years:
        working["year"] = year

        x_input = working[feature_columns]
        
        # 1. Model predicts the YoY growth rate natively from features and autoregressive lags
        predicted_gpu_growth = model_gpu.predict(x_input)
        predicted_revenue_growth = model_revenue.predict(x_input)

        # 2. Dynamically compound absolute demand and revenue from predicted growth rates
        predicted_gpu_demand = working["ai_gpu_demand"] * (1.0 + predicted_gpu_growth)
        predicted_chip_revenue = working["ai_chip_revenue"] * (1.0 + predicted_revenue_growth)

        year_result = working[["country_key"]].copy()
        year_result["forecast_year"] = year
        year_result["predicted_ai_gpu_demand"] = predicted_gpu_demand
        year_result["predicted_ai_chip_revenue"] = predicted_chip_revenue
        results.append(year_result)

        # 3. Update autoregressive state organically for the next iteration (NO arbitrary drift)
        working["ai_gpu_demand_lag_1"] = working["ai_gpu_demand"]
        working["ai_chip_revenue_lag_1"] = working["ai_chip_revenue"]
        
        working["ai_gpu_demand"] = predicted_gpu_demand
        working["ai_chip_revenue"] = predicted_chip_revenue
        
        # 4. Update market growth rate feature with the model's own predicted revenue growth
        working["market_growth_rate"] = predicted_revenue_growth

    forecast_df = pd.concat(results, ignore_index=True)
    forecast_df = forecast_df.merge(country_lookup, on="country_key", how="left")
    forecast_df = forecast_df.sort_values(by=["forecast_year", "country_name"]).reset_index(drop=True)

    forecast_df["predicted_market_growth_rate"] = forecast_df.groupby("country_key")[
        "predicted_ai_chip_revenue"
    ].pct_change().fillna(0.0)

    forecast_df = forecast_df[
        [
            "forecast_year",
            "country_key",
            "country_name",
            "predicted_ai_gpu_demand",
            "predicted_ai_chip_revenue",
            "predicted_market_growth_rate",
        ]
    ]

    return forecast_df


def build_global_summary(forecast_country_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building global yearly forecast summary")

    global_summary = (
        forecast_country_df.groupby("forecast_year")
        .agg(
            global_unit_demand=("predicted_ai_gpu_demand", "sum"),
            global_market_value=("predicted_ai_chip_revenue", "sum"),
        )
        .reset_index()
        .sort_values(by="forecast_year")
        .reset_index(drop=True)
    )

    base_value = global_summary.loc[0, "global_market_value"]
    min_year = global_summary["forecast_year"].min()
    
    global_summary["cagr"] = global_summary.apply(
        lambda row: (
            ((row["global_market_value"] / base_value) ** (1 / max(row["forecast_year"] - min_year, 1))) - 1
        )
        if base_value not in (0, np.nan) and row["forecast_year"] != min_year
        else 0.0,
        axis=1,
    )

    return global_summary


def save_pandas_as_parquet(
    spark: SparkSession, pdf: pd.DataFrame, output_path: Path
) -> None:
    logger.info("Saving output to %s using pandas to bypass PySpark worker crash", output_path)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "part-00000.parquet"
    pdf.to_parquet(str(file_path), engine="pyarrow", index=False)

    (output_path / "_SUCCESS").touch()


def run_pipeline() -> None:
    spark = get_spark_session()

    tables = load_gold_tables(spark)
    training_spark_df = build_training_dataset(tables)
    raw_pdf = spark_to_pandas(training_spark_df)

    country_lookup = (
        raw_pdf[["country_key", "country_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    processed_pdf = preprocess_features(raw_pdf)

    country_key_codes = raw_pdf[["country_key"]].drop_duplicates().reset_index(drop=True)
    country_key_codes["country_key_encoded"] = country_key_codes["country_key"].astype(
        "category"
    ).cat.codes
    country_lookup = country_lookup.merge(
        country_key_codes, on="country_key", how="left"
    )
    country_lookup_encoded = country_lookup[
        ["country_key_encoded", "country_name"]
    ].rename(columns={"country_key_encoded": "country_key"})

    feature_columns = ALL_FEATURE_COLUMNS

    train_df, test_df = chronological_split(processed_pdf)

    all_metrics: List[pd.DataFrame] = []
    all_importances: List[pd.DataFrame] = []
    trained_models: Dict[str, xgb.XGBRegressor] = {}

    # Train models on growth targets instead of absolute level ceilings
    for target_column, growth_target in zip(TARGET_COLUMNS, GROWTH_TARGET_COLUMNS):
        model, metrics_df = train_xgboost_model(
            train_df, test_df, feature_columns, growth_target
        )
        importance_df = extract_feature_importance(
            model, feature_columns, growth_target
        )

        trained_models[target_column] = model
        all_metrics.append(metrics_df)
        all_importances.append(importance_df)

    evaluation_df = pd.concat(all_metrics, ignore_index=True)
    feature_importance_df = pd.concat(all_importances, ignore_index=True)

    snapshot = build_latest_country_snapshot(processed_pdf)

    forecast_country_df = iterative_forecast(
        model_gpu=trained_models["ai_gpu_demand"],
        model_revenue=trained_models["ai_chip_revenue"],
        snapshot=snapshot,
        country_lookup=country_lookup_encoded,
        feature_columns=feature_columns,
        forecast_years=FORECAST_YEARS,
    )

    forecast_global_df = build_global_summary(forecast_country_df)

    save_pandas_as_parquet(spark, evaluation_df, EVALUATION_PATH)
    save_pandas_as_parquet(spark, feature_importance_df, FEATURE_IMPORTANCE_PATH)
    save_pandas_as_parquet(spark, forecast_country_df, FORECAST_COUNTRY_PATH)
    save_pandas_as_parquet(spark, forecast_global_df, FORECAST_GLOBAL_PATH)

    logger.info("Forecast pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()