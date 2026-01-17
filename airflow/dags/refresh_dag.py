from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import sys
import os

# Function to check for Kaggle credentials
def check_kaggle_creds():
    kaggle_key = os.environ.get("KAGGLE_KEY")
    home_kaggle = os.path.expanduser("~/.kaggle/kaggle.json")
    if not kaggle_key and not os.path.exists(home_kaggle):
        raise ValueError("Kaggle credentials not found in env or ~/.kaggle")

# Default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,  # ROBUSTNESS: Increased retries
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True, # EXPONENTIAL BACKOFF
}

with DAG(
    'movie_data_refresh',
    default_args=default_args,
    description='Daily refresh of movie recommendation data',
    schedule_interval='0 3 * * *',  # 3 AM daily
    catchup=False,
) as dag:

    # Task 1: Check Credentials
    t0_check_creds = PythonOperator(
        task_id='check_creds',
        python_callable=check_kaggle_creds,
    )

    # Task 2: Download Data (Using Kaggle CLI via Bash)
    # Ensure raw directory exists and clean old file
    # We CD into movie-rec first to keep paths relative
    download_cmd = """
    cd movie-rec
    mkdir -p data/raw
    kaggle datasets download -d alanvourch/tmdb-movies-daily-updates -p data/raw --unzip --force
    mv data/raw/TMDB_movie_dataset_*.csv data/raw/TMDB_all_movies.csv || true
    """
    
    t1_download = BashOperator(
        task_id='download_from_kaggle',
        bash_command=download_cmd,
    )

    # Task 3: Run Spark ETL
    # IDEMPOTENCY: Pass logical date {{ ds }} for partitioning
    t2_spark_etl = BashOperator(
        task_id='run_spark_etl',
        bash_command='cd movie-rec && python etl/pyspark_etl.py --date {{ ds }}',
    )
    
    # Task 4: Rebuild Index
    # Uses our consolidated pandas_etl.py just for indexing
    t3_index = BashOperator(
        task_id='rebuild_index',
        bash_command='cd movie-rec && python -m etl.pandas_etl --index-only',
    )

    # DAG Flow
    t0_check_creds >> t1_download >> t2_spark_etl >> t3_index
