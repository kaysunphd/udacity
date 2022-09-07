# Databricks notebook source
# ingestion to Bronze
# read in payments.csv
df = spark.read.format('csv') \
            .option('inferSchema', 'true') \
            .option('header', 'false') \
            .option('sep', ',') \
            .load('/FileStore/project2/payments.csv') \
            .toDF('payment_id', 'date', 'amount', 'rider_id')

df = df.withColumn("date", df["date"].cast('date')) \
        .withColumn("amount", df["amount"].cast('float'))
df.printSchema()

# write payments to delta
df.write.format('delta') \
        .mode('overwrite') \
        .save('/delta/bronze_payments')

# COMMAND ----------

# read payments from delta
df = spark.read.format('delta') \
        .load('/delta/bronze_payments')

# convert payments to table
spark.sql("DROP TABLE IF EXISTS bronze_dimPayment")
df.write.format('delta') \
        .mode('overwrite') \
        .saveAsTable('bronze_dimPayment')

# COMMAND ----------

# ingestion to Bronze
# read in riders.csv
df = spark.read.format('csv') \
            .option('inferSchema', 'true') \
            .option('header', 'false') \
            .option('sep', ',') \
            .load('/FileStore/project2/riders.csv') \
            .toDF('rider_id', 'first', 'last', 'address', 'birthday', 'account_start_date', 'account_end_date', 'is_member')

df = df.withColumn("birthday", df["birthday"].cast('date')) \
        .withColumn("account_start_date", df["account_start_date"].cast('date')) \
        .withColumn("account_end_date", df["account_end_date"].cast('date'))
df.printSchema()

# write riders to delta
df.write.format('delta') \
        .mode('overwrite') \
        .save('/delta/bronze_riders')

# COMMAND ----------

# read riders from delta
df = spark.read.format('delta') \
        .load('/delta/bronze_riders')

# convert riders to table
spark.sql("DROP TABLE IF EXISTS gold_dimRider")
df.write.format('delta') \
        .mode('overwrite') \
        .saveAsTable('gold_dimRider')

# COMMAND ----------

# ingestion to Bronze
# read in trips.csv
df = spark.read.format('csv') \
            .option('inferSchema', 'true') \
            .option('header', 'false') \
            .option('sep', ',') \
            .load('/FileStore/project2/trips.csv') \
            .toDF('trip_id', 'rideable_type', 'started_at', 'ended_at', 'start_station_id', 'end_station_id', 'rider_id')

df.printSchema()

# write trips to delta
df.write.format('delta') \
        .mode('overwrite') \
        .save('/delta/bronze_trips')

# COMMAND ----------

# read trips from delta
df = spark.read.format('delta') \
        .load('/delta/bronze_trips')

# convert trips to table
spark.sql("DROP TABLE IF EXISTS bronze_dimTrip")
df.write.format('delta') \
        .mode('overwrite') \
        .saveAsTable('bronze_dimTrip')

# COMMAND ----------

# ingestion to Bronze
# read in stations.csv
df = spark.read.format('csv') \
            .option('inferSchema', 'true') \
            .option('header', 'false') \
            .option('sep', ',') \
            .load('/FileStore/project2/stations.csv') \
            .toDF('station_id', 'name', 'latitude', 'longitude')

df.printSchema()

# write riders to delta
df.write.format('delta') \
        .mode('overwrite') \
        .save('/delta/bronze_stations')

# COMMAND ----------

# read stations from delta
df = spark.read.format('delta') \
        .load('/delta/bronze_stations')

# convert station to tables
spark.sql("DROP TABLE IF EXISTS gold_dimStation")
df.write.format('delta') \
        .mode('overwrite') \
        .saveAsTable('gold_dimStation')

# COMMAND ----------

# Derivative of https://sparkbyexamples.com/pyspark/pyspark-sql-date-and-timestamp-functions/
# Gold business level
# create Date calendar dimension table
from pyspark.sql.functions import *

start,stop = ['2013-01-01', '2043-01-01']
interval=60*60*24
dt_col="date_time"

temp_df = spark.createDataFrame([(start, stop)], ("start", "stop"))
temp_df = temp_df.select([col(c).cast("timestamp") for c in ("start", "stop")])
temp_df = temp_df.withColumn("stop",F.date_add("stop",1).cast("timestamp"))
temp_df = temp_df.select([col(c).cast("long") for c in ("start", "stop")])
start, stop = temp_df.first()
df = spark.range(start,stop,interval).select(col("id").cast("timestamp").alias(dt_col))

df = df.select(to_date(col("date_time"), "yyyy-MM-dd").cast(StringType()).alias("date_id"),
     to_date(col("date_time"), "yyyy-MM-dd").alias("date"),
     year(col("date_time")).alias("year"), 
     month(col("date_time")).alias("month"), 
     weekofyear(col("date_time")).alias("weekofyear") ,
     dayofweek(col("date_time")).alias("dayofweek"), 
     dayofmonth(col("date_time")).alias("dayofmonth"), 
     dayofyear(col("date_time")).alias("dayofyear") 
  )

spark.sql("DROP TABLE IF EXISTS gold_dimTime")
df.write.format('delta') \
        .mode('overwrite') \
        .saveAsTable('gold_dimTime')

# COMMAND ----------

# Create factPayment fact table
spark.sql("DROP TABLE IF EXISTS factPayment")
spark.sql("CREATE TABLE factPayment ( \
           date_key STRING, \
           rider_key INT, \
           amount DOUBLE) \
          ")

spark.sql("INSERT INTO factPayment (date_key, rider_key, amount) \
            SELECT d.date_id            AS date_key, \
                   r.rider_id           AS rider_key, \
                   p.amount             AS amount \
            FROM bronze_dimPayment p \
            JOIN gold_dimRider r  ON (p.rider_id = r.rider_id) \
            JOIN gold_dimTime d ON (d.date = p.date) \
         ")

# COMMAND ----------

# Create factTrip fact table
spark.sql("DROP TABLE IF EXISTS factTrip")
spark.sql("CREATE TABLE factTrip ( \
          date_key STRING, \
          rider_key INT, \
          start_station_key STRING, \
          end_station_key STRING, \
          rider_age INT, \
          trip_duration INT) \
          ")

spark.sql("INSERT INTO factTrip (date_key, rider_key, start_station_key, end_station_key, rider_age, trip_duration) \
           SELECT d.date_id                                                                                                AS date_key, \
           r.rider_id                                                                                                      AS rider_key, \
           t.start_station_id                                                                                              AS start_station_key, \
           t.end_station_id                                                                                                AS end_station_key, \
           DATEDIFF(YEAR, r.birthday, t.started_at)                                  AS rider_age, \
           DATEDIFF(MINUTE, t.started_at, t.ended_at)  AS trip_duration \
           FROM bronze_dimTrip t \
           JOIN gold_dimRider r  ON (r.rider_id = t.rider_id) \
           JOIN bronze_dimPayment p  ON (p.rider_id = t.rider_id) \
           JOIN gold_dimTime d ON (d.date = p.date) \
         ")

# COMMAND ----------

# remove delta files
# dbutils.fs.rm('/delta/bronze_payments', recurse=True)
# dbutils.fs.rm('/delta/bronze_riders', recurse=True)
# dbutils.fs.rm('/delta/bronze_stations', recurse=True)
# dbutils.fs.rm('/delta/bronze_trips', recurse=True)
