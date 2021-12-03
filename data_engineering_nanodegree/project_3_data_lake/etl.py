import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Creates a Spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Spark ETL of song data.
    Reads in song_data from S3 in form of JSON and creates dimensions tables (songs_table and artists_table).
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_columns = ['song_id', 'title', 'artist_id','artist_name', 'year', 'duration']
    songs_table = df[songs_columns]
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs.parquet'), 'overwrite')

    # extract columns to create artists table
    artists_columns = ['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude']
    artists_table = df[artists_columns]
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    """
    Spark ETL of log data.
    Reads in log_data from S3 in form of JSON and creates dimensions tables (users_table and time_table), as well as fact table (songplays).
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_columns = ['userId', 'firstName', 'lastName', 'gender', 'level','ts']
    users_table = df[users_columns]
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(int(int(x)/1000)), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(int(int(x)/1000)), TimestampType())
    get_weekday = udf(lambda x: x.weekday())
    get_week = udf(lambda x: x.isocalendar()[1])
    get_hour = udf(lambda x: x.hour)
    get_day = udf(lambda x : x.day)
    get_year = udf(lambda x: x.year)
    get_month = udf(lambda x: x.month)
    
    df = df.withColumn('start_time', get_datetime(df.ts))
    df = df.withColumn('hour', get_hour(df.start_time))
    df = df.withColumn('day', get_day(df.start_time))
    df = df.withColumn('week', get_week(df.start_time))
    df = df.withColumn('month', get_month(df.start_time))
    df = df.withColumn('year', get_year(df.start_time))
    df = df.withColumn('weekday', get_weekday(df.start_time))
    
    # extract columns to create time table
    time_columns = ['start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday']
    time_table = df[time_columns]
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(os.path.join(output_data, 'time.parquet'), 'overwrite')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(os.path.join(output_data, 'songs.parquet'))

    # extract columns from joined song and log datasets to create songplays table
    df = df.join(song_df, (song_df.title == df.song) & (song_df.artist_name == df.artist))
    songplays_columns = ['start_time', 'userId', 'level', 'song_id', 'artist_id', 'sessionId', 'location', 'userAgent']
    songplays_table = df[songplays_columns]

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(os.path.join(output_data, 'songplays.parquet'), 'overwrite')


def print_parquet(spark, fullfilename):
    """
    Print out schema and head of saved parquet.
    """
    df = spark.read.parquet(fullfilename)
    df.printSchema()
    df.show(7)

    
def main():
    """
    Calls for creation of Spark session.
    Run ETL for song data and log data.
    Print out schema and first 7 rows of saved parquet.
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

    print_parquet(spark, os.path.join(output_data, 'songs.parquet'))
    print_parquet(spark, os.path.join(output_data, 'artists.parquet'))
    print_parquet(spark, os.path.join(output_data, 'users.parquet'))
    print_parquet(spark, os.path.join(output_data, 'time.parquet'))
    print_parquet(spark, os.path.join(output_data, 'songplays.parquet'))

    
if __name__ == "__main__":
    main()
