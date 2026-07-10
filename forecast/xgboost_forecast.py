"""
xgboost_forecast.py

Forecasting pipeline for semiconductor production planning.

Builds FOUR independent XGBoost regression models, one per target:
    1. production_capacity_wafers
    2. ai_chip_production
    3. foundry_revenue_usd
    4. global_market_share

Data layout expected (star schema, parquet files):
    data/gold/dim_company.parquet
    data/gold/dim_country.parquet
    data/gold/dim_date.parquet
    data/gold/dim_product.parquet
    data/gold/fact_market_price.parquet
    data/gold/fact_production.parquet
    data/gold/fact_stock_market.parquet

Outputs
-------
1. Trained models (one per target):
       models/<target>_model.json

2. EVALUATION results - actual vs. predicted on the held-out chronological
   test set, used to judge model quality. ONE combined file stacking all
   four targets (long format, distinguished by `target_metric`):
       data/evaluation/evaluation_results.parquet
       columns: company, country, year, actual_value, predicted_value, target_metric

3. FORECAST results - genuine future predictions beyond the latest year
   present in the dataset (at least the next 3 years, for every company).
   ONE combined file stacking all four targets:
       data/forecast/forecast_results.parquet
       columns: company, country, forecast_year, predicted_value, target_metric
   (no actual_value column, since these years have not happened yet)

Design notes / assumptions
---------------------------
- fact_production is at (year, company, country) grain. Its `year_key` is
  assumed to represent a calendar year value directly (e.g. 2020, 2021, ...),
  matching the `year` column found in dim_date. A small lookup table is built
  from dim_date to translate year_key -> year robustly, but if year_key is
  already a 4-digit year we fall back to using it as-is.
- fact_stock_market and fact_market_price are DAILY grain (date_key). They
  are aggregated up to YEARLY grain (mean of numeric columns) per company /
  overall market before being joined onto the yearly production table, since
  the production facts are yearly.
- fact_market_price has no company/country key, only product_key. Since the
  production targets are not product-specific, market price is aggregated
  into a single yearly "average market price" macro feature (mean across all
  products for that year) rather than joined per-row.
- LEAKAGE CONTROL: to predict a given (company, year) target, we only use
  information available strictly BEFORE that year. Every training feature is
  a *lag* (previous year) version of a metric - current-year production
  metrics, stock aggregates, and price aggregates are never used to predict
  the current year's own target. The only "current year" information used is
  the calendar year itself (a trend feature) and static company/country
  identity.
- Rows with no prior-year history (the first observed year for a company)
  cannot have lag features and are dropped from modeling.
- Train/test split is chronological: the most recent slice of years is held
  out as the test set. No random shuffling is used anywhere.
- MULTI-YEAR FUTURE FORECASTING is done recursively: to predict year Y+2 we
  need year Y+1's metrics as lag features, but year Y+1 hasn't happened yet
  either. So we predict Y+1 first using all four models, feed those
  predictions back in as the "lag" state, then predict Y+2, and so on. Exogenous
  drivers we don't model directly (fab_count, stock-market aggregates, average
  market price) are held constant at their last known value for the whole
  forecast horizon (a simple persistence assumption) - this is called out
  explicitly in the code below.
- For deployment-quality future forecasts, each target's FINAL model is
  refit on the ENTIRE historical dataset (train + test years) after the
  train/test evaluation is complete. The evaluation metrics themselves are
  always computed from a model that only ever saw the training years, so
  they remain an honest, leakage-free estimate of accuracy.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is required. Install it with: pip install xgboost"
    ) from e

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GOLD_DIR = Path("data/gold")
MODELS_DIR = Path("models")
EVALUATION_DIR = Path("data/evaluation")
FORECAST_DIR = Path("data/forecast")

EVALUATION_OUTPUT_PATH = EVALUATION_DIR / "evaluation_results.parquet"
FORECAST_OUTPUT_PATH = FORECAST_DIR / "forecast_results.parquet"

# Metrics we want to forecast.
TARGET_COLUMNS = [
    "production_capacity_wafers",
    "ai_chip_production",
    "foundry_revenue_usd",
    "global_market_share",
]

# All numeric production metrics that get lagged as candidate features.
PRODUCTION_METRICS = [
    "production_capacity_wafers",
    "ai_chip_production",
    "foundry_revenue_usd",
    "global_market_share",
    "fab_count",
]

TEST_SIZE_FRACTION = 0.2  # fraction of the most-recent years held out for testing
MIN_TEST_YEARS = 1
N_FORECAST_YEARS = 3  # forecast at least this many years beyond the latest year


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_gold_tables(gold_dir: Path = GOLD_DIR) -> dict:
    """Load every parquet file from data/gold into a dict of DataFrames.

    Missing files are tolerated with a warning so the pipeline can degrade
    gracefully instead of crashing outright.
    """
    expected_tables = [
        "dim_company",
        "dim_country",
        "dim_date",
        "dim_product",
        "fact_market_price",
        "fact_production",
        "fact_stock_market",
    ]

    tables = {}
    for name in expected_tables:
        # Support both "data/gold/name.parquet" and "data/gold/name/*.parquet"
        file_path = gold_dir / f"{name}.parquet"
        dir_path = gold_dir / name
        try:
            if file_path.exists():
                tables[name] = pd.read_parquet(file_path)
            elif dir_path.exists() and dir_path.is_dir():
                tables[name] = pd.read_parquet(dir_path)
            else:
                print(f"[WARN] Table '{name}' not found under {gold_dir}, skipping.")
                tables[name] = pd.DataFrame()
        except Exception as exc:
            print(f"[WARN] Failed to load '{name}': {exc}")
            tables[name] = pd.DataFrame()

    return tables


# --------------------------------------------------------------------------
# 2. Joining dimensions with facts to build a clean yearly dataset
# --------------------------------------------------------------------------
def build_year_lookup(dim_date: pd.DataFrame) -> pd.DataFrame:
    """Build a year_key -> year lookup table from dim_date.

    fact_production.year_key is assumed to reference a calendar year. If
    dim_date contains a matching 'year' column we use the distinct years
    directly; otherwise year_key is treated as the literal year.
    """
    if dim_date is None or dim_date.empty or "year" not in dim_date.columns:
        return pd.DataFrame(columns=["year_key", "year"])

    years = dim_date["year"].dropna().unique()
    return pd.DataFrame({"year_key": years, "year": years})


def build_production_base(tables: dict) -> pd.DataFrame:
    """Join fact_production with dim_company, dim_country, and a year lookup."""
    fact_production = tables.get("fact_production", pd.DataFrame())
    dim_company = tables.get("dim_company", pd.DataFrame())
    dim_country = tables.get("dim_country", pd.DataFrame())
    dim_date = tables.get("dim_date", pd.DataFrame())

    if fact_production.empty:
        raise ValueError("fact_production is empty or missing - cannot build dataset.")

    df = fact_production.copy()

    # Resolve year_key -> year. Fall back to using year_key as-is if we
    # can't build a lookup (e.g. dim_date missing 'year' column).
    year_lookup = build_year_lookup(dim_date)
    if not year_lookup.empty and "year_key" in df.columns:
        merged = df.merge(year_lookup, on="year_key", how="left")
        if merged["year"].isna().all():
            df["year"] = df["year_key"]  # lookup didn't match -> year_key IS the year
        else:
            df = merged
            df["year"] = df["year"].fillna(df["year_key"])
    else:
        df["year"] = df["year_key"]

    # Join company / country dimensions
    if not dim_company.empty and "company_key" in df.columns:
        df = df.merge(dim_company, on="company_key", how="left")
    if not dim_country.empty and "country_key" in df.columns:
        df = df.merge(dim_country, on="country_key", how="left")

    if "company" not in df.columns:
        df["company"] = df.get("company_key")
    if "country" not in df.columns:
        df["country"] = df.get("country_key")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "company_key"])

    # Filter out rows where 'company' is actually a country (due to Kaggle data column shifts)
    if not dim_country.empty and "country" in dim_country.columns:
        country_names = set(dim_country["country"].dropna().astype(str).str.lower())
        # Also explicitly add some known countries in case they were filtered out of dim_country earlier
        country_names.update([
            "china", "japan", "mexico", "south korea", "taiwan", "uk", "usa", "vietnam", "germany", "india"
        ])
        df = df[~df["company"].astype(str).str.lower().isin(country_names)]

    return df


def aggregate_stock_yearly(tables: dict) -> pd.DataFrame:
    """Aggregate daily fact_stock_market up to (company, year) grain."""
    fact_stock = tables.get("fact_stock_market", pd.DataFrame())
    dim_date = tables.get("dim_date", pd.DataFrame())

    if fact_stock.empty or dim_date.empty or "year" not in dim_date.columns:
        return pd.DataFrame(columns=["company_key", "year"])

    df = fact_stock.merge(dim_date[["date_key", "year"]], on="date_key", how="left")

    agg_cols = ["open", "high", "low", "close", "volume", "daily_return", "ma_5", "ma_20"]
    agg_cols = [c for c in agg_cols if c in df.columns]

    yearly = (
        df.groupby(["company_key", "year"], as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"stock_{c}_yearly_avg" for c in agg_cols})
    )
    return yearly


def aggregate_price_yearly(tables: dict) -> pd.DataFrame:
    """Aggregate daily fact_market_price up to a single yearly macro feature
    (mean price across all products), since price has no company/country key."""
    fact_price = tables.get("fact_market_price", pd.DataFrame())
    dim_date = tables.get("dim_date", pd.DataFrame())

    if fact_price.empty or dim_date.empty or "year" not in dim_date.columns:
        return pd.DataFrame(columns=["year", "avg_market_price_yearly"])

    df = fact_price.merge(dim_date[["date_key", "year"]], on="date_key", how="left")
    yearly = (
        df.groupby("year", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "avg_market_price_yearly"})
    )
    return yearly


def build_enriched_yearly_table(tables: dict) -> tuple:
    """Build the (company, year) grain table with production metrics, stock
    aggregates, and price aggregates all attached, BEFORE any lagging.

    Returns (enriched_df, stock_price_cols) where stock_price_cols lists the
    exogenous aggregate column names that were attached (so callers know
    which columns are the "current year" versions vs. lag versions).
    """
    base = build_production_base(tables)
    stock_yearly = aggregate_stock_yearly(tables)
    price_yearly = aggregate_price_yearly(tables)

    df = base.copy()
    if not stock_yearly.empty:
        df = df.merge(stock_yearly, on=["company_key", "year"], how="left")
    if not price_yearly.empty:
        df = df.merge(price_yearly, on="year", how="left")

    df = df.sort_values(["company_key", "year"]).reset_index(drop=True)

    stock_price_cols = [
        c for c in df.columns if c.startswith("stock_") or c == "avg_market_price_yearly"
    ]

    # Trend feature: years since this company's first observed record.
    df["years_since_start"] = df.groupby("company_key")["year"].rank(method="dense") - 1

    return df, stock_price_cols


# --------------------------------------------------------------------------
# 3. Feature engineering (lag features, encoding, leakage control)
# --------------------------------------------------------------------------
def engineer_features(tables: dict) -> tuple:
    """Build the full modeling dataset with lag-only features to prevent leakage.

    Returns (model_df, enriched_df, lag_feature_names, stock_price_cols):
        model_df       - lagged dataset ready for training (leak-free)
        enriched_df    - the un-lagged (company, year) table, kept around so we
                          can seed recursive future forecasting from each
                          company's most recent actual values
        lag_feature_names - list of the lag column names used as features
        stock_price_cols  - list of exogenous aggregate column names (unlagged)
    """
    enriched_df, stock_price_cols = build_enriched_yearly_table(tables)
    df = enriched_df.copy()

    lag_source_cols = [c for c in PRODUCTION_METRICS if c in df.columns] + stock_price_cols

    lag_feature_names = []
    for col in lag_source_cols:
        lag_col = f"{col}_lag1"
        df[lag_col] = df.groupby("company_key")[col].shift(1)
        lag_feature_names.append(lag_col)

    # Drop the raw (same-year) stock/price columns now that lags exist -
    # keeping them around would leak current-year info into the features.
    # (Production metric columns stay - they are the targets themselves.)
    df = df.drop(columns=stock_price_cols, errors="ignore")

    # Rows with no prior year (first year per company) can't be modeled.
    df = df.dropna(subset=lag_feature_names, how="all")

    return df, enriched_df, lag_feature_names, stock_price_cols


def encode_categoricals(df: pd.DataFrame, columns=("company", "country")) -> tuple:
    """Label-encode categorical identity columns. Returns (df, encoders)."""
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


# --------------------------------------------------------------------------
# 4. Train / test split (chronological, no shuffling)
# --------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, year_col: str = "year",
                         test_fraction: float = TEST_SIZE_FRACTION):
    """Split rows into train/test by year, preserving time order.

    The most recent N years become the test set - never randomly shuffled.
    """
    years = sorted(df[year_col].dropna().unique())
    n_test_years = max(MIN_TEST_YEARS, int(round(len(years) * test_fraction)))
    n_test_years = min(n_test_years, len(years) - 1) if len(years) > 1 else 0

    if n_test_years <= 0:
        return df, df.iloc[0:0]

    test_years = set(years[-n_test_years:])
    train_df = df[~df[year_col].isin(test_years)].copy()
    test_df = df[df[year_col].isin(test_years)].copy()
    return train_df, test_df


# --------------------------------------------------------------------------
# 5. Model training and evaluation
# --------------------------------------------------------------------------
def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor with reasonable general-purpose defaults."""
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute RMSE, MAE, R2 on the held-out chronological test set."""
    if len(X_test) == 0:
        return {"rmse": None, "mae": None, "r2": None}

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds)) if len(y_test) > 1 else None
    return {"rmse": rmse, "mae": mae, "r2": r2}


def show_feature_importance(model: XGBRegressor, feature_names: list, target: str, top_n: int = 15):
    """Print feature importances sorted from most to least important."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    print(f"\nFeature importance for '{target}':")
    for idx in order:
        print(f"  {feature_names[idx]:<40s} {importances[idx]:.4f}")


# --------------------------------------------------------------------------
# 6. Saving outputs
# --------------------------------------------------------------------------
def save_model(model: XGBRegressor, target: str, models_dir: Path = MODELS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{target}_model.json"
    model.save_model(out_path)
    print(f"Saved model -> {out_path}")


def save_evaluation_results(eval_df: pd.DataFrame, out_path: Path = EVALUATION_OUTPUT_PATH):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(out_path, index=False)
    print(f"\nSaved evaluation results -> {out_path}  ({len(eval_df)} rows)")


def save_forecast_results(forecast_df: pd.DataFrame, out_path: Path = FORECAST_OUTPUT_PATH):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(out_path, index=False)
    print(f"Saved forecast results -> {out_path}  ({len(forecast_df)} rows)")


# --------------------------------------------------------------------------
# 7. Per-target pipeline: evaluate on chronological holdout, then refit on
#    all historical data to get the "final" model used for future forecasts.
# --------------------------------------------------------------------------
def run_target_pipeline(df: pd.DataFrame, target: str, feature_cols: list) -> dict:
    """Train + evaluate a target's model on the chronological split, then
    refit a final model on the FULL dataset for downstream forecasting.

    Returns a dict with: eval_df, metrics, final_model
    """
    print("\n" + "=" * 70)
    print(f"Target: {target}")
    print("=" * 70)

    model_df = df.dropna(subset=[target]).copy()
    train_df, test_df = chronological_split(model_df, "year")

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Train years: {sorted(train_df['year'].unique())}")
    print(f"Test years:  {sorted(test_df['year'].unique())}")

    # --- Evaluation model: trained ONLY on the training years ---
    eval_model = train_xgb_model(X_train, y_train)
    metrics = evaluate_model(eval_model, X_test, y_test)

    print(f"RMSE: {metrics['rmse']}")
    print(f"MAE:  {metrics['mae']}")
    print(f"R2:   {metrics['r2']}")

    show_feature_importance(eval_model, feature_cols, target)

    eval_df = pd.DataFrame(columns=["company", "country", "year", "actual_value",
                                     "predicted_value", "target_metric"])
    if len(X_test) > 0:
        preds = eval_model.predict(X_test)
        eval_df = pd.DataFrame({
            "company": test_df["company"].values,
            "country": test_df["country"].values,
            "year": test_df["year"].values,
            "actual_value": y_test.values,
            "predicted_value": preds,
            "target_metric": target,
        })
    else:
        print(f"[WARN] No test rows available for '{target}' - evaluation rows skipped.")

    # --- Final model: refit on ALL historical rows (train + test) so future
    # forecasts benefit from the most recent known years too. ---
    X_full, y_full = model_df[feature_cols], model_df[target]
    final_model = train_xgb_model(X_full, y_full)
    save_model(final_model, target)

    return {"eval_df": eval_df, "metrics": metrics, "final_model": final_model}


# --------------------------------------------------------------------------
# 8. Recursive multi-year future forecasting
# --------------------------------------------------------------------------
def forecast_future_years(enriched_df: pd.DataFrame, stock_price_cols: list,
                           models: dict, encoders: dict, feature_cols: list,
                           n_years: int = N_FORECAST_YEARS) -> pd.DataFrame:
    """Recursively forecast the next `n_years` beyond each company's latest
    known year, for every target metric.

    Because our lag-1 features need last year's values, and future years by
    definition don't have real observed values, we forecast one year at a
    time and feed each year's predictions back in as the "lag" state for the
    next year. Exogenous inputs we don't forecast ourselves (fab_count,
    stock-market aggregates, average market price) are held constant at the
    company's last known value for the whole horizon (persistence
    assumption) - a simple, transparent stand-in for a full driver forecast.
    """
    results = []

    for company_key, group in enriched_df.groupby("company_key"):
        group = group.sort_values("year")
        seed = group.iloc[-1]  # this company's most recent known year
        last_year = int(seed["year"])
        company_name = seed["company"]
        country_name = seed["country"]
        years_since_start = seed["years_since_start"]

        # Encode company/country using the SAME encoders fit during training.
        try:
            company_encoded = encoders["company"].transform([str(company_name)])[0]
        except Exception:
            company_encoded = -1
        try:
            country_encoded = encoders["country"].transform([str(country_name)])[0]
        except Exception:
            country_encoded = -1

        # "state" holds the values that will be used as next year's lag
        # features. Initialize from the company's actual last known values.
        state = {}
        for metric in PRODUCTION_METRICS:
            if metric in seed:
                state[metric] = seed[metric]
        for col in stock_price_cols:
            if col in seed:
                state[col] = seed[col]

        for step in range(1, n_years + 1):
            forecast_year = last_year + step

            feature_row = {
                "year": forecast_year,
                "years_since_start": years_since_start + step,
                "company_encoded": company_encoded,
                "country_encoded": country_encoded,
            }
            for metric in PRODUCTION_METRICS:
                feature_row[f"{metric}_lag1"] = state.get(metric, 0)
            for col in stock_price_cols:
                feature_row[f"{col}_lag1"] = state.get(col, 0)

            X_future = pd.DataFrame([feature_row])
            # Ensure every expected feature column is present (fill 0 if not)
            for col in feature_cols:
                if col not in X_future.columns:
                    X_future[col] = 0
            X_future = X_future[feature_cols]

            step_predictions = {}
            for target, model in models.items():
                pred = float(model.predict(X_future)[0])
                step_predictions[target] = pred
                results.append({
                    "company": company_name,
                    "country": country_name,
                    "forecast_year": forecast_year,
                    "predicted_value": pred,
                    "target_metric": target,
                })

            # Roll the state forward: predicted production metrics become
            # next year's lag inputs. fab_count / stock / price aggregates
            # are exogenous here and simply persist at their last known
            # value (see docstring above).
            for metric in ["production_capacity_wafers", "ai_chip_production",
                           "foundry_revenue_usd", "global_market_share"]:
                if metric in step_predictions:
                    state[metric] = step_predictions[metric]

    forecast_df = pd.DataFrame(
        results,
        columns=["company", "country", "forecast_year", "predicted_value", "target_metric"],
    )
    return forecast_df


# --------------------------------------------------------------------------
# 9. Main
# --------------------------------------------------------------------------
def main():
    print("Loading gold-layer parquet tables...")
    tables = load_gold_tables()

    print("Joining dimensions with facts and engineering lag features...")
    model_df, enriched_df, lag_feature_names, stock_price_cols = engineer_features(tables)
    model_df, encoders = encode_categoricals(model_df)

    # Normalize company/country to strings on the un-lagged enriched table so
    # the forecasting step can encode them with the SAME encoders fit above.
    enriched_df["company"] = enriched_df["company"].astype(str)
    enriched_df["country"] = enriched_df["country"].astype(str)

    # Feature set shared across all 4 models: year trend, encoded identity,
    # and every lagged (past-only) metric. No same-year metric is ever used.
    base_features = ["year", "years_since_start", "company_encoded", "country_encoded"]
    feature_cols = [c for c in base_features if c in model_df.columns] + [
        c for c in lag_feature_names if c in model_df.columns
    ]

    # Fill remaining NaNs in numeric features (e.g. companies with sparse
    # stock/price history) with 0 so XGBoost's dense matrix path is stable.
    model_df[feature_cols] = model_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Modeling dataset shape: {model_df.shape}")
    print(f"Feature columns used ({len(feature_cols)}): {feature_cols}")

    # --- 1. Train, evaluate, and refit a final model for every target ---
    eval_frames = []
    final_models = {}
    all_metrics = {}

    for target in TARGET_COLUMNS:
        if target not in model_df.columns:
            print(f"[WARN] Target column '{target}' not found in dataset - skipping.")
            continue
        result = run_target_pipeline(model_df, target, feature_cols)
        eval_frames.append(result["eval_df"])
        final_models[target] = result["final_model"]
        all_metrics[target] = result["metrics"]

    evaluation_results = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    save_evaluation_results(evaluation_results)

    # --- 2. Recursive future forecasting beyond the latest year in the data ---
    print("\n" + "=" * 70)
    print(f"Forecasting {N_FORECAST_YEARS} years beyond each company's latest year")
    print("=" * 70)
    forecast_results = forecast_future_years(
        enriched_df, stock_price_cols, final_models, encoders, feature_cols,
        n_years=N_FORECAST_YEARS,
    )
    save_forecast_results(forecast_results)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary of model performance (chronological holdout)")
    print("=" * 70)
    for target, metrics in all_metrics.items():
        print(f"{target}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}")


if __name__ == "__main__":
    main()