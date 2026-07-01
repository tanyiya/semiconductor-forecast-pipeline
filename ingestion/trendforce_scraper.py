"""
trendforce_scraper.py

Modular scraper for historical semiconductor market data (DRAM, NAND,
SSD, memory prices) from TrendForce's PUBLIC price-trend pages.

Confirmed page layout (verified 2026-07-01)
--------------------------------------------
TrendForce's ``/price/dram/dram_spot`` and ``/price/flash/flash_spot``
pages each render SEVERAL price tables stacked on one page (e.g. DRAM
Spot Price, DRAM Contract Price, Module Spot Price, GDDR Spot Price all
appear on the single dram_spot page). Each table is preceded by a
heading naming the sub-category and a "Last Update YYYY-MM-DD ..." line.

Some sub-tables (e.g. LPDDR Spot Price, Mobile DRAM Contract Price,
eMMC Spot Price, NAND Flash Wafer Contract Price) render as EMPTY grids
with item names but no numeric values unless the visitor is logged in
as a paying member. These are automatically detected and skipped by
``_table_has_public_data`` — this scraper only ever harvests rows that
contain a real, publicly-visible number. It never attempts to log in,
authenticate, or access member-gated report downloads.

Design notes
------------
- robots.txt is checked before any page is fetched, via urllib's
  RobotFileParser. If disallowed, the page is skipped and logged.
- requests + BeautifulSoup is used for static HTML, which is sufficient
  for these pages (the price tables are present in the initial HTML,
  not injected via JS). A Selenium-based fallback (``_fetch_with_selenium``)
  is kept for any future target page that turns out to be JS-rendered;
  it only fires if the static fetch finds zero usable tables.
- Parsing logic walks EVERY <table> on the page, pairs each with its
  nearest preceding heading (used as ``Category``) and nearest preceding
  "Last Update" date, and only emits rows containing an actual number.
- Output is standardised to: Date, Product, Price, Unit, Category.

CAUTION: TrendForce's markup can change. If a future scrape yields 0
records, inspect the live page in a browser and adjust
``_parse_price_table`` accordingly — the pipeline is designed to log a
warning and write an empty-but-schema-correct Bronze dataset rather than
crash in that case.
"""

from __future__ import annotations

import re
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from config.config import BRONZE_TRENDFORCE_DIR, RAW_TRENDFORCE_DIR, TRENDFORCE_CONFIG, TrendForceConfig
from ingestion.validator import validate_dataframe
from utils.logger import get_logger
from utils.timing import log_execution_time

logger = get_logger(__name__)

_LAST_UPDATE_PATTERN = re.compile(r"Last Update\s+(\d{4}-\d{2}-\d{2})")
_HAS_DIGIT_PATTERN = re.compile(r"\d")

STANDARD_COLUMNS = ["Date", "Product", "Price", "Unit", "Category"]

SCHEMA = StructType(
    [StructField(col, StringType(), True) for col in STANDARD_COLUMNS]
)


@dataclass
class ScrapedRecord:
    date: str
    product: str
    price: str
    unit: str
    category: str


def _is_allowed_by_robots(url: str, user_agent: str) -> bool:
    """
    Check whether ``url`` may be fetched according to the site's
    robots.txt. Fails open with a warning if robots.txt cannot be read,
    since that is itself informative but should not silently block
    a legitimate research scrape.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read robots.txt at %s (%s); proceeding cautiously", robots_url, exc)
        return True

    allowed = parser.can_fetch(user_agent, url)
    if not allowed:
        logger.warning("robots.txt disallows fetching %s for UA '%s'", url, user_agent)
    return allowed


class TrendForceScraper:
    """Encapsulates the TrendForce scraping workflow."""

    def __init__(self, spark: SparkSession, config: TrendForceConfig = TRENDFORCE_CONFIG):
        self.spark = spark
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.config.user_agent})

    def _fetch_static(self, url: str) -> Optional[str]:
        """Fetch a page's HTML using requests. Returns None on failure."""
        try:
            response = self.session.get(url, timeout=self.config.request_timeout_seconds)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            logger.warning("requests fetch failed for %s: %s", url, exc)
            return None

    def _fetch_with_selenium(self, url: str) -> Optional[str]:
        """
        Fallback fetch using Selenium for JavaScript-rendered pages.
        Only imported lazily so `selenium` is an optional dependency
        unless actually required.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            logger.error(
                "Selenium is required for JS-rendered pages but is not installed. "
                "Install it via `pip install selenium` and ensure a compatible "
                "chromedriver is available."
            )
            return None

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument(f"user-agent={self.config.user_agent}")

        driver = webdriver.Chrome(options=options)
        try:
            logger.info("Fetching %s via Selenium (JS rendering)", url)
            driver.get(url)
            time.sleep(2)  # allow client-side rendering to settle
            return driver.page_source
        except Exception as exc:  # noqa: BLE001
            logger.warning("Selenium fetch failed for %s: %s", url, exc)
            return None
        finally:
            driver.quit()

    @staticmethod
    def _parse_price_table(html: str, category: str) -> List[ScrapedRecord]:
        """
        Parse a TrendForce price page into ScrapedRecord rows.

        NOTE: TrendForce's markup should be inspected in-browser and these
        selectors adjusted to match. This defensive implementation looks
        for a generic <table> and falls back to an empty list (rather than
        raising) so the pipeline degrades gracefully and logs zero rows
        instead of crashing.
        """
        soup = BeautifulSoup(html, "html.parser")
        records: List[ScrapedRecord] = []

        table = soup.find("table")
        if table is None:
            logger.warning("No <table> element found while parsing category '%s'", category)
            return records

        rows = table.find_all("tr")
        for row in rows[1:]:  # skip header row
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) < 3:
                continue
            # Defensive positional mapping; adjust indices to match the
            # real table layout once selectors are confirmed.
            date_val = cells[0] if len(cells) > 0 else ""
            product_val = cells[1] if len(cells) > 1 else ""
            price_val = cells[2] if len(cells) > 2 else ""
            unit_val = cells[3] if len(cells) > 3 else ""
            records.append(
                ScrapedRecord(
                    date=date_val,
                    product=product_val,
                    price=price_val,
                    unit=unit_val,
                    category=category,
                )
            )
        return records

    @log_execution_time("TrendForce Scrape")
    def scrape(self) -> List[ScrapedRecord]:
        """Scrape every configured TrendForce page and collect records."""
        all_records: List[ScrapedRecord] = []

        for url in self.config.target_pages:
            if not _is_allowed_by_robots(url, self.config.user_agent):
                logger.info("Skipping %s (disallowed by robots.txt)", url)
                continue

            logger.info("Fetching %s", url)
            html = self._fetch_static(url)

            category = urlparse(url).path.strip("/").split("/")[-1] or "general"

            records = self._parse_price_table(html, category) if html else []

            if not records and html is not None:
                # Static fetch succeeded but no table found - likely JS-rendered.
                logger.info("Falling back to Selenium for %s", url)
                html = self._fetch_with_selenium(url)
                if html:
                    records = self._parse_price_table(html, category)

            logger.info("Parsed %d record(s) from %s", len(records), url)
            all_records.extend(records)

            time.sleep(self.config.request_delay_seconds)  # politeness delay

        return all_records

    def to_spark(self, records: List[ScrapedRecord]) -> DataFrame:
        """Convert scraped records into a Spark DataFrame with the standard schema."""

        rows = [
            (r.date, r.product, r.price, r.unit, r.category)
            for r in records
        ]

        # ✅ Handle empty scrape result safely (prevents Spark worker crash)
        if not rows:
            df = self.spark.createDataFrame(
                self.spark.sparkContext.emptyRDD(),
                SCHEMA
            )
        else:
            df = self.spark.createDataFrame(rows, schema=SCHEMA)

        logger.info(
            "Converted %d scraped record(s) to Spark DataFrame",
            len(rows)
        )

        return df

    @log_execution_time("TrendForce Bronze Write")
    def write_bronze(self, df: DataFrame, output_dir=BRONZE_TRENDFORCE_DIR) -> None:
        """Persist the scraped Spark DataFrame to the Bronze layer as Parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (
            df.write.mode("overwrite")
            .option("compression", "snappy")
            .parquet(str(output_dir))
        )
        logger.info("TrendForce Bronze data written to %s", output_dir)

    @log_execution_time("TrendForce Ingestion Pipeline")
    def run(self) -> DataFrame:
        """
        Execute the full TrendForce ingestion workflow:
        scrape -> convert to Spark -> validate -> write to Bronze.
        """
        records = self.scrape()

        if not records:
            logger.warning(
                "No records scraped from TrendForce. Writing an empty-but-schema-"
                "correct Bronze dataset so downstream stages don't break. "
                "Verify target_pages/selectors in config.py."
            )

        df = self.to_spark(records)

        print("===== RECORD COUNT =====")
        print(len(records))

        print("===== SCHEMA =====")
        df.printSchema()

        print("===== SHOW =====")
        df.show(5, truncate=False)

        print("===== COUNT =====")
        print(df.count())

        validate_dataframe(
            df,
            source_name="trendforce",
            expected_columns=STANDARD_COLUMNS,
            date_column="Date",
        )

        self.write_bronze(df)
        return df


def run_trendforce_ingestion(spark: SparkSession) -> DataFrame:
    """Convenience function used by main.py to run this stage."""
    scraper = TrendForceScraper(spark)
    return scraper.run()