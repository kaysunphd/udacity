import configparser
import psycopg2
from sql_queries import copy_table_queries, insert_table_queries


def load_staging_tables(cur, conn):
    """
    Load data from S3 to staging tables.
    """
    for query in copy_table_queries:
#         print(query)
        cur.execute(query)
        conn.commit()


def insert_tables(cur, conn):
    """
    Insert data into tables from staging tables.
    """
    for query in insert_table_queries:
#         print(query)
        cur.execute(query)
        conn.commit()


def main():
    """
    - Parse configuration parameters to connect to AWS. 
    
    - Establishes connection with the sparkify database.  
    
    - Load data staging tables from S3.  
    
    - Insert data into tables. 
    
    - Finally, closes the connection. 
    """
    # parse configuration to AWS
    config = configparser.ConfigParser()
    config.read('dwh.cfg')

    # connect to database
    conn = psycopg2.connect("host={} dbname={} user={} password={} port={}".format(*config['CLUSTER'].values()))
    cur = conn.cursor()
    
    # load data into staging tables
    load_staging_tables(cur, conn)
    
    # insert data into tables
    insert_tables(cur, conn)

    # close connection
    conn.close()


if __name__ == "__main__":
    main()