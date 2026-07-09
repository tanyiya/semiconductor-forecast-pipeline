import os
os.environ['PYSPARK_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe"
from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local[*]').getOrCreate()
print("DIM COMPANY:")
spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/gold/dim_company').show(100, False)
print("DIM COUNTRY:")
spark.read.parquet('d:/01_Bomi/03_Projects/semiconductor-forecast-pipeline/data/gold/dim_country').show(100, False)
