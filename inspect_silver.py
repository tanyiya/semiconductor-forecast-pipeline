from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').getOrCreate()
yahoo = spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/silver/yahoo')
yahoo.show(5)
kaggle = spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/silver/kaggle')
kaggle.show(5)
