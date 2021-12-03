Purpose:
- Sparkify has a new music streaming app that want analysis done on the user activity and songs played to understand what songs their users are listening to.

Dataset:
- Sparkify's user activity is stored in a directory of JSON logs, while a directory with JSON metadata contains information on the songs in their app. Both are stored in the cloud, in AWS S3.

Database Schema:
- A star schema, consisting of a central fact table of song plays and surrounding dimension tables of users, songs, artists and time information, is selected for optimum queries on song play analyses.

    * Fact Table
songplays - records in log data associated with song plays and contains: \
songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent

    * Dimension Tables
users - users in the app and contains: \
user_id, first_name, last_name, gender, level

songs - songs in music database and contains: \
user_id, first_name, last_name, gender, level
song_id, title, artist_id, year, duration

artists - artists in music database and contains: \
user_id, first_name, last_name, gender, level
artist_id, name, location, latitude, longitude

time - timestamps of records in songplays broken down into specific units and contains: \
user_id, first_name, last_name, gender, level
start_time, hour, day, week, month, year, weekday

ETL Pipeline:
- Data are extracted from each JSON in songs and logs dataset in S3, which are then staged in AWS Redshift, and ultimately transformed into the fact and dimesion tables for Spariky's analytics team to explore and discover.

Files:
create_tables.py
- creates the database and drop tables if already exist.

sql_queries.py
- contains SQL queries to drop, create, insert and select staging, fact, and dimension tables.

etl.py
- extract data from dataset, stage, and insert into tables.

dwh.cfg
- configuration parameters to AWS warehouse with critical information scrubbed.

To Run:
0. start AWS cluster
1. create database
> python create_tables.py

2. stage and insert tables
> python etl.py

3. test by going into AWS Redshift Query Editor. add database, user, and password, start querying.
> SELECT * FROM songplays LIMIT 5;