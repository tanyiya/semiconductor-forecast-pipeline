from spark.spark_session import get_spark_session, stop_spark_session
from processing.kaggle_bronze_silver import run_kaggle_silver_transform
from processing.yahoo_bronze_silver import run_yahoo_silver_transform
from processing.trendforce_bronze_silver import run_trendforce_silver_transform

spark = get_spark_session()
try:
    run_kaggle_silver_transform(spark)
    run_yahoo_silver_transform(spark)
    run_trendforce_silver_transform(spark)
finally:
    stop_spark_session()