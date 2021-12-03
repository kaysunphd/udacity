from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class LoadFactOperator(BaseOperator):
    """
    Load data from staging tables to fact table in AWS Redshift.
    """

    ui_color = '#F98866'

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="",
                 table="",
                 sql_query="",
                 *args, **kwargs):

        super(LoadFactOperator, self).__init__(*args, **kwargs)
        self.table = table
        self.redshift_conn_id = redshift_conn_id
        self.sql_query = sql_query

    def execute(self, context):
        redshift_hook = PostgresHook(self.redshift_conn_id)
        self.log.info(f"Loading fact {self.table} table")
        
        redshift_hook.run("DELETE FROM {}".format(self.table))
        redshift_hook.run(f"Insert into {self.table} {self.sql_query}")