Purpose:
- Sparkify has a new music streaming app that they want analysis done on the user activity and songs played to understand what songs their users are listening to.

Dataset:
- Sparkify's user activity is stored in a directory of JSON logs, while a directory with JSON metadata contains information on the songs in their app.

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
- Data are extracted from each JSON in songs and logs dataset to create and populate the fact and dimesion tables in Postgres using Python and SQL.

create_tables.py
- creates the database and drop tables if already exist.

sql_queries.py
- contains SQL queries to drop, create, insert and select tables.

etl.py
- extract data from dataset and populate tables.
etl.ipynb
- notebook to develop and test ETL process for each table.

test.ipynb
- notebook to test database and query tables.

To Run:
1. create database \
> python create_tables.py

2. Build ETL Pipeline \
> python etl.py

3. Sample query \
> %load_ext sql \
> %sql postgresql://student:student@127.0.0.1/sparkifydb \
> %sql SELECT * FROM songplays LIMIT 5;