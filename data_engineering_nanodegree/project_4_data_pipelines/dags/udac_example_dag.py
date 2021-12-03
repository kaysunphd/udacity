from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.postgres_operator import PostgresOperator

from operators.stage_redshift import StageToRedshiftOperator
from operators.load_fact import LoadFactOperator
from operators.load_dimension import LoadDimensionOperator
from operators.data_quality import DataQualityOperator

# from airflow.operators import (StageToRedshiftOperator, LoadFactOperator, LoadDimensionOperator, DataQualityOperator)

from helpers import SqlQueries

# AWS_KEY = os.environ.get('')
# AWS_SECRET = os.environ.get('')

# Define arguments to DAG
# based on https://airflow.apache.org/docs/apache-airflow/stable/dag-run.html
default_args = {
    'owner': 'udacity',
    'start_date': datetime(2019, 1, 12),
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_retry': False
}

dag = DAG('udac_example_dag',
          default_args=default_args,
          description='Load and transform data in Redshift with Airflow',
          schedule_interval='0 * * * *',
          catchup=False
        )

# Dummy start task
start_operator = DummyOperator(task_id='Begin_execution',  dag=dag)

# Create tables (if not exist) task
create_tables = PostgresOperator(
    task_id="create_tables",
    dag=dag,
    postgres_conn_id="redshift",
    sql="create_tables.sql"
)

# Stage events to AWS Redshift task
stage_events_to_redshift = StageToRedshiftOperator(
    task_id='Stage_events',
    dag=dag,
    aws_credentials_id="aws_credentials",
    redshift_conn_id="redshift",
    table="staging_events",
    s3_bucket="udacity-dend",
    s3_key="log_data",
    json_path="s3://udacity-dend/log_json_path.json"
)

# Stage songs to AWS Redshift task
stage_songs_to_redshift = StageToRedshiftOperator(
    task_id='Stage_songs',
    dag=dag,
    aws_credentials_id="aws_credentials",
    redshift_conn_id="redshift",
    table="staging_songs",
    s3_bucket="udacity-dend",
    s3_key="song_data/A/A/A",
    json_path="auto"
)

# Load songplays table from staging task
load_songplays_table = LoadFactOperator(
    task_id='Load_songplays_fact_table',
    dag=dag,
    redshift_conn_id="redshift",
    table="songplays",
    sql_query=SqlQueries.songplay_table_insert
)

# Load users table from staging task
load_user_dimension_table = LoadDimensionOperator(
    task_id='Load_user_dim_table',
    dag=dag,
    redshift_conn_id="redshift",
    table="users",
    sql_query=SqlQueries.user_table_insert
)

# Load songs table from staging task
load_song_dimension_table = LoadDimensionOperator(
    task_id='Load_song_dim_table',
    dag=dag,
    redshift_conn_id="redshift",
    table="songs",
    sql_query=SqlQueries.song_table_insert
)

# Load artistis table from staging task
load_artist_dimension_table = LoadDimensionOperator(
    task_id='Load_artist_dim_table',
    dag=dag,
    redshift_conn_id="redshift",
    table="artists",
    sql_query=SqlQueries.artist_table_insert
)

# Load time table from staging task
load_time_dimension_table = LoadDimensionOperator(
    task_id='Load_time_dim_table',
    dag=dag,
    redshift_conn_id="redshift",
    table="time",
    sql_query=SqlQueries.time_table_insert
)

# Run data quality checks on dimension tables task
run_quality_checks = DataQualityOperator(
    task_id='Run_data_quality_checks',
    dag=dag,
    redshift_conn_id="redshift",
    list_tables=['songplays', 'users', 'songs', 'artists', 'time']
)

# Dummy end task
end_operator = DummyOperator(task_id='Stop_execution',  dag=dag)

# Define Airflow workflow
start_operator >> create_tables

create_tables >> stage_events_to_redshift
create_tables >> stage_songs_to_redshift

stage_events_to_redshift >> load_songplays_table
stage_songs_to_redshift >> load_songplays_table

load_songplays_table >> load_user_dimension_table
load_songplays_table >> load_song_dimension_table
load_songplays_table >> load_artist_dimension_table
load_songplays_table >> load_time_dimension_table

load_user_dimension_table >> run_quality_checks
load_song_dimension_table >> run_quality_checks
load_artist_dimension_table >> run_quality_checks
load_time_dimension_table >> run_quality_checks

run_quality_checks >> end_operator