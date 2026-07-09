from spark.spark_session import get_spark_session, stop_spark_session
from config.config import GOLD_DIM_COMPANY_DIR, GOLD_DIM_COUNTRY_DIR

spark = get_spark_session()

print("===== DIM COMPANY =====")
df = spark.read.parquet(str(GOLD_DIM_COMPANY_DIR))
df.printSchema()
df.show(30, truncate=False)

print("===== DIM COUNTRY =====")
df = spark.read.parquet(str(GOLD_DIM_COUNTRY_DIR))
df.printSchema()
df.show(30, truncate=False)