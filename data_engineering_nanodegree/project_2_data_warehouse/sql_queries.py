import configparser


# CONFIG
config = configparser.ConfigParser()
config.read('dwh.cfg')

# DROP TABLES
"""
- SQL to drop tables
"""
staging_events_table_drop = "DROP TABLE IF EXISTS staging_events"
staging_songs_table_drop = "DROP TABLE IF EXISTS staging_songs"
songplay_table_drop = "DROP TABLE IF EXISTS songplays"
user_table_drop = "DROP TABLE IF EXISTS users"
song_table_drop = "DROP TABLE IF EXISTS songs"
artist_table_drop = "DROP TABLE IF EXISTS artists"
time_table_drop = "DROP TABLE IF EXISTS time"

# CREATE TABLES
"""
- SQL to create tables
"""
staging_events_table_create= ("""CREATE TABLE IF NOT EXISTS staging_events \
(artist varchar, \
auth varchar, \
firstName varchar, \
gender varchar, \
itemInSession int, \
lastName varchar, \
length double, \
level varchar, \
location varchar, \
method varchar, \
page varchar, \
registration double, \
session_id int, \
song varchar, \
status int, \
ts bigint, \
user_agent varchar, \
user_id varchar) \
;""")

staging_songs_table_create = ("""CREATE TABLE IF NOT EXISTS staging_songs \
(num_songs int, \
artist_id varchar, \
artist_latitude double, \
artist_location varchar, \
artist_longitude double, \
artist_name varchar, \
song_id varchar, \
title varchar, \
duration double, \
year int) \
;""")

songplay_table_create = ("""CREATE TABLE IF NOT EXISTS songplays \
(songplay_id int IDENTITY(0,1) PRIMARY KEY, \
start_time timestamp NOT NULL, \
user_id varchar NOT NULL, \
level varchar, \
song_id varchar, \
artist_id varchar, \
session_id int, \
location varchar, \
user_agent varchar) \
;""")

user_table_create = ("""CREATE TABLE IF NOT EXISTS users \
(user_id varchar PRIMARY KEY, \
first_name varchar, \
last_name varchar, \
gender varchar, \
level varchar) \
;""")

song_table_create = ("""CREATE TABLE IF NOT EXISTS songs \
(song_id varchar PRIMARY KEY, \
title varchar NOT NULL, \
artist_id varchar NOT NULL, \
year int, \
duration numeric) \
;""")

artist_table_create = ("""CREATE TABLE IF NOT EXISTS artists \
(artist_id varchar PRIMARY KEY, \
name varchar NOT NULL, \
location varchar, \
latitude numeric, \
longitude numeric) \
;""")

time_table_create = ("""CREATE TABLE IF NOT EXISTS time \
(start_time timestamp PRIMARY KEY, \
hour int, \
day int, \
week int, \
month int, \
year int, \
weekday int) \
;""")

# STAGING TABLES

staging_events_copy = ("""COPY staging_events \
(artist, \
auth, \
firstName, \
gender, \
itemInSession, \
lastName, \
length, \
level, \
location, \
method, \
page, \
registration, \
session_id, \
song, \
status, \
ts, \
user_agent, \
user_id) \
FROM {} \
iam_role {} \
FORMAT JSON AS {}; \
""").format(config.get('S3', 'LOG_DATA'), config.get('IAM_ROLE', 'ARN'), config.get('S3', 'LOG_JSONPATH'))

staging_songs_copy = ("""COPY staging_songs \
(num_songs, \
artist_id, \
artist_latitude, \
artist_location, \
artist_longitude, \
artist_name, \
song_id, \
title, \
duration, \
year) \
FROM {} \
iam_role {} \
FORMAT JSON AS 'auto'; \
""").format(config.get('S3', 'SONG_DATA'), config.get('IAM_ROLE', 'ARN'))

# FINAL TABLES
# add epoch to conver to modern timestamp
# https://stackoverflow.com/questions/39815425/how-to-convert-epoch-to-datetime-redshift
songplay_table_insert = ("""INSERT INTO songplays \
(start_time, \
user_id, \
level, \
song_id, \
artist_id, \
session_id, \
location, \
user_agent) \
SELECT \
timestamp 'epoch' + staging_events.ts * interval '0.001 seconds' as start_time,
staging_events.user_id, \
staging_events.level, \
staging_songs.song_id, \
staging_songs.artist_id, \
staging_events.session_id, \
staging_events.location, \
staging_events.user_agent \
FROM staging_events, staging_songs \
WHERE staging_events.artist = staging_songs.artist_name \
AND staging_events.song = staging_songs.title \
AND staging_events.page = 'NextSong' \
""")

user_table_insert = ("""INSERT INTO users \
(user_id, \
first_name, \
last_name, \
gender, \
level) \
SELECT DISTINCT \
user_id, \
firstName, \
lastName, \
gender, \
level \
FROM staging_events \
""")

song_table_insert = ("""INSERT INTO songs \
(song_id, \
title, \
artist_id, \
year, \
duration) \
SELECT DISTINCT \
song_id, \
title, \
artist_id, \
year, \
duration \
FROM staging_songs \
WHERE song_id IS NOT NULL \
""")

artist_table_insert = ("""INSERT INTO artists \
(artist_id, \
name, \
location, \
latitude, \
longitude) \
SELECT DISTINCT \
artist_id, \
artist_name as name, \
artist_location as location, \
artist_latitude as latitude, \
artist_longitude as longitude \
FROM staging_songs \
WHERE artist_id IS NOT NULL \
""")

time_table_insert = ("""INSERT INTO time \
(start_time, \
hour, \
day, \
week, \
month, \
year, \
weekday) \
SELECT DISTINCT \
start_time, \
EXTRACT(hour from start_time) as hour, \
EXTRACT(day from start_time) as day, \
EXTRACT(week from start_time) as week, \
EXTRACT(month from start_time) as month, \
EXTRACT(year from start_time) as year, \
EXTRACT(weekday from start_time) as weekday \
FROM songplays \
""")

# QUERY LISTS
"""
List of queries to create table, drop table, copy table, and insert table
"""
create_table_queries = [staging_events_table_create, staging_songs_table_create, songplay_table_create, user_table_create, song_table_create, artist_table_create, time_table_create]
drop_table_queries = [staging_events_table_drop, staging_songs_table_drop, songplay_table_drop, user_table_drop, song_table_drop, artist_table_drop, time_table_drop]
copy_table_queries = [staging_events_copy, staging_songs_copy]
insert_table_queries = [songplay_table_insert, user_table_insert, song_table_insert, artist_table_insert, time_table_insert]
