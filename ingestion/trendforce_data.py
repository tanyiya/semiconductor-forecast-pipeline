"""
trendforce_data.py

Generates SYNTHETIC TrendForce-shaped price data and writes it to the
Bronze layer as a single ``data.parquet`` file at
``data/bronze/trendforce/data.parquet``.

This is a TEST FIXTURE, not real scraped data. It exists so the Silver
transformer (``processing/trendforce_bronze_silver.py``) and anything
downstream can be developed and tested end-to-end without depending on
live network access to trendforce.com. The output schema exactly matches
what ``ingestion/trendforce_scraper.py`` produces
(Date, Product, Price, Unit, Category), so it's a drop-in stand-in.

Usage
-----
    python trendforce_data.py
    python trendforce_data.py --days 120 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from config.config import BRONZE_TRENDFORCE_DIR, ensure_directories
from ingestion.trendforce_scraper import STANDARD_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)

# (category, product, starting_price, daily_volatility)
PRODUCT_SEEDS = [
    ("DRAM Spot Price", "DDR4 8Gb 1Gx8 2666MHz", 1.80, 0.02),
    ("DRAM Spot Price", "DDR4 16Gb 2Gx8 3200MHz", 3.10, 0.02),
    ("DRAM Spot Price", "DDR5 16Gb 2Gx8 4800MHz", 4.25, 0.03),
    ("DRAM Contract Price", "DDR4 8Gb 1Gx8 2666MHz", 1.75, 0.01),
    ("DRAM Contract Price", "DDR5 16Gb 2Gx8 4800MHz", 4.10, 0.015),
    ("GDDR Spot Price", "GDDR6 8Gb", 5.60, 0.025),
    ("Module Spot Price", "DDR4 8GB 2666MHz UDIMM", 14.50, 0.02),
    ("Module Spot Price", "DDR5 16GB 4800MHz UDIMM", 32.00, 0.025),
    ("NAND Flash Spot Price", "128Gb MLC", 3.40, 0.03),
    ("NAND Flash Spot Price", "512Gb TLC", 2.15, 0.03),
    ("NAND Flash Contract Price", "512Gb TLC", 2.05, 0.015),
    ("NAND Flash SSD Street Price", "samsung 970 evo 1tb", 82.00, 0.02),
    ("NAND Flash SSD Street Price", "western digital blue 1tb", 74.50, 0.02),
]

UNIT = "USD"


def _generate_price_series(start_price: float, volatility: float, days: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a simple bounded random walk to stand in for daily prices."""
    returns = rng.normal(loc=0.0, scale=volatility, size=days)
    series = start_price * np.cumprod(1 + returns)
    # Keep prices sane (no negative/near-zero synthetic prices).
    series = np.clip(series, a_min=start_price * 0.5, a_max=start_price * 1.8)
    return series


def generate_synthetic_trendforce_data(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic DataFrame shaped exactly like a real TrendForce
    Bronze extract: columns Date, Product, Price, Unit, Category (all
    strings, matching ingestion.trendforce_scraper.STANDARD_COLUMNS).

    Parameters
    ----------
    days : int
        Number of daily observations to generate per product.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    end_date = datetime.today().date()
    dates = [end_date - timedelta(days=offset) for offset in range(days)][::-1]

    rows: List[dict] = []
    for category, product, start_price, volatility in PRODUCT_SEEDS:
        prices = _generate_price_series(start_price, volatility, days, rng)
        for date_val, price_val in zip(dates, prices):
            rows.append(
                {
                    "Date": date_val.strftime("%Y-%m-%d"),
                    "Product": product,
                    "Price": f"{price_val:.2f}",
                    "Unit": UNIT,
                    "Category": category,
                }
            )

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    logger.info(
        "Generated %d synthetic TrendForce row(s) across %d product(s) over %d day(s)",
        len(df),
        len(PRODUCT_SEEDS),
        days,
    )
    return df


def write_bronze_parquet(df: pd.DataFrame, output_path=None) -> None:
    """Write the synthetic DataFrame to data/bronze/trendforce/data.parquet."""
    ensure_directories()
    output_path = output_path or (BRONZE_TRENDFORCE_DIR / "data.parquet")
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")
    logger.info("Synthetic TrendForce Bronze data written to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TrendForce-shaped Bronze data for testing."
    )
    parser.add_argument("--days", type=int, default=90, help="Number of daily observations per product.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        "Generating synthetic TrendForce data (days=%d, seed=%d). "
        "NOTE: this is fixture data for testing, not real scraped data.",
        args.days,
        args.seed,
    )
    df = generate_synthetic_trendforce_data(days=args.days, seed=args.seed)
    write_bronze_parquet(df)


if __name__ == "__main__":
    main()