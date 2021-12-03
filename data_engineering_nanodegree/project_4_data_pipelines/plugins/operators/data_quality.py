from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class DataQualityOperator(BaseOperator):
    """
    Run data quality checks on each of the dimension table. Each table and first row of table must not be empty.
    """

    ui_color = '#89DA59'

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="",
                 list_tables=[],
                 *args, **kwargs):

        super(DataQualityOperator, self).__init__(*args, **kwargs)
        self.list_tables = list_tables
        self.redshift_conn_id = redshift_conn_id

    def execute(self, context):
        redshift_hook = PostgresHook(self.redshift_conn_id)
        
        for table in self.list_tables:
            self.log.info(f"Running data quality checks on {table} table")
            
            records = redshift_hook.get_records(f"SELECT COUNT(*) FROM {table}")
        
            if len(records) < 1 or len(records[0]) < 1:
                raise ValueError(f"Data quality check failed. {table} returned no results")
            num_records = records[0][0]

            if num_records < 1:
                raise ValueError(f"Data quality check failed. {table} contained 0 rows")
                
            self.log.info(f"Data quality on table {table} check passed with {records[0][0]} records")