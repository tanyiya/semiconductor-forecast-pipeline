import os
os.environ['PYSPARK_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.master('local[*]').getOrCreate()

# Create dummy yahoo_df and kaggle_df
yahoo_df = spark.createDataFrame([("NVDA",)], ["Ticker"])
kaggle_df = spark.createDataFrame([("nvidia",)], ["company"])

TICKER_TO_COMPANY_NAME = {"NVDA": "nvidia"}

def _create_map_expr(mapping):
    literals = []
    for k, v in mapping.items():
        literals.extend([F.lit(k), F.lit(v)])
    return F.create_map(*literals)

ticker_to_company = _create_map_expr(TICKER_TO_COMPANY_NAME)
company_to_ticker = _create_map_expr({v: k for k, v in TICKER_TO_COMPANY_NAME.items()})

yahoo_companies = (
    yahoo_df.select("ticker")
    .distinct()
    .withColumn(
        "company",
        F.coalesce(ticker_to_company.getItem(F.col("ticker")), F.lower(F.col("ticker"))),
    )
    .select("company", "ticker")
)

kaggle_companies = (
    kaggle_df.select(F.col("company").alias("company"))
    .distinct()
    .withColumn("ticker", company_to_ticker.getItem(F.col("company")))
    .select("company", "ticker")
)

print("yahoo schema:")
yahoo_companies.printSchema()
print("kaggle schema:")
kaggle_companies.printSchema()

combined = yahoo_companies.unionByName(kaggle_companies)
print("combined schema:")
combined.printSchema()
combined.show()

dim_company = combined.groupBy("company").agg(
    F.first(F.col("ticker"), ignorenulls=True).alias("ticker")
)
dim_company.show()

