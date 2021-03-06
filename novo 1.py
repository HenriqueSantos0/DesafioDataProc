from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv("gs://stack-labs-minio-list/olist_customers_dataset.csv",header=True,sep="|")

df = spark.write.parquet.save("gs://stack-labs-minio-list/olist_customers_dataset.parquet",header=True,sep="|")