from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length
spark = SparkSession.builder.master('local[*]').getOrCreate()

kaggle = spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/silver/kaggle')
print("Kaggle company count distinct:", kaggle.select("company").distinct().count())
print("Kaggle companies sample:")
kaggle.select("company").distinct().show(100, False)

print("Kaggle country sample:")
kaggle.select("country").distinct().show(100, False)

yahoo = spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/silver/yahoo')
print("Yahoo Tickers:", yahoo.select("Ticker").distinct().collect())
