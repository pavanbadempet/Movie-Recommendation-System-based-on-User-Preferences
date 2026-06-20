"""
Kafka and Spark Integration DAG for Movie Recommendation System

This DAG demonstrates how to integrate Kafka event streaming with Spark processing
in the movie recommendation pipeline using the new Kafka cluster for Bronze layer.
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow import DAG

# Import decoupled remote Spark operator
try:
    from operators.remote_spark import RemoteSparkSubmitOperator
except ImportError:
    try:
        from dags.operators.remote_spark import RemoteSparkSubmitOperator
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from operators.remote_spark import RemoteSparkSubmitOperator

# Default args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


def check_kafka_connection():
    """Check if Kafka cluster is available and accessible"""
    from kafka import KafkaAdminClient
    from kafka.errors import KafkaError

    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers="kafka-1:9092,kafka-2:9092,kafka-3:9092", client_id="airflow-connection-test"
        )
        topics = admin_client.list_topics()
        admin_client.close()
        return f"Kafka cluster connection successful. Topics: {topics}"
    except KafkaError as e:
        raise ValueError(f"Kafka cluster connection failed: {e}")


def check_spark_connection():
    """Check if Spark is available and accessible"""
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName("AirflowConnectionTest")
            .master("spark://spark:7077")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )

        spark_version = spark.version
        spark.stop()
        return f"Spark connection successful. Version: {spark_version}"
    except Exception as e:
        raise ValueError(f"Spark connection failed: {e}")


with DAG(
    "kafka_spark_integration",
    default_args=default_args,
    description="Kafka and Spark integration for movie recommendation pipeline (Bronze layer)",
    schedule="@daily",
    catchup=False,
    tags=["integration", "kafka", "spark", "data-pipeline", "bronze-layer"],
) as dag:
    # Task 1: Check Kafka connection
    t0_check_kafka = PythonOperator(
        task_id="check_kafka_connection",
        python_callable=check_kafka_connection,
    )

    # Task 2: Check Spark connection
    t1_check_spark = PythonOperator(
        task_id="check_spark_connection",
        python_callable=check_spark_connection,
    )

    # Task 3: Create Kafka topic for movie events (Bronze layer)
    t2_create_topic = BashOperator(
        task_id="create_movie_events_topic",
        bash_command="""
        kafka-topics.sh --bootstrap-server kafka-1:9092 \
          --create --if-not-exists \
          --topic raw_movie_events \
          --partitions 3 \
          --replication-factor 2
        """,
    )

    # Task 4: Simulate producing movie events to Kafka (Bronze layer)
    t3_produce_events = BashOperator(
        task_id="produce_movie_events",
        bash_command="""
        cd /opt/airflow/movie-rec
        python -c "
        from kafka import KafkaProducer
        import json
        import random
        import time

        producer = KafkaProducer(
            bootstrap_servers='kafka-1:9092,kafka-2:9092,kafka-3:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Sample movie events
        movies = [
            {'id': 1, 'title': 'The Shawshank Redemption', 'event_type': 'view', 'user_id': 101},
            {'id': 2, 'title': 'The Godfather', 'event_type': 'rating', 'user_id': 102, 'rating': 5},
            {'id': 3, 'title': 'Pulp Fiction', 'event_type': 'view', 'user_id': 103},
            {'id': 4, 'title': 'The Dark Knight', 'event_type': 'rating', 'user_id': 104, 'rating': 4},
            {'id': 5, 'title': 'Inception', 'event_type': 'view', 'user_id': 105}
        ]

        for movie in movies:
            producer.send('raw_movie_events', movie)
            print(f'Sent event: {movie}')
            time.sleep(0.5)

        producer.flush()
        producer.close()
        print('Finished producing events')
        "
        """,
    )

    # Task 5: Process events with Spark (Bronze layer)
    t4_process_events = BashOperator(
        task_id="process_events_with_spark",
        bash_command="""
        cd /opt/airflow/movie-rec
        python -c "
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, from_json, lower, trim
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType

        # Create Spark session
        spark = SparkSession.builder \\
            .appName('KafkaEventProcessing') \\
            .master('spark://spark:7077') \\
            .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \\
            .getOrCreate()

        # Define schema for movie events
        schema = StructType([
            StructField('id', IntegerType(), True),
            StructField('title', StringType(), True),
            StructField('event_type', StringType(), True),
            StructField('user_id', IntegerType(), True),
            StructField('rating', IntegerType(), True)
        ])
        allowed_event_types = ['view', 'click', 'rating', 'search', 'recommendation_request', 'recommendation_impression']

        # Read from Kafka (Bronze layer topic)
        df = spark.readStream \\
            .format('kafka') \\
            .option('kafka.bootstrap.servers', 'kafka-1:9092,kafka-2:9092,kafka-3:9092') \\
            .option('subscribe', 'raw_movie_events') \\
            .option('startingOffsets', 'earliest') \\
            .option('maxOffsetsPerTrigger', '1000') \\
            .option('failOnDataLoss', 'false') \\
            .load()

        # Parse JSON, normalize event type, and bound state cardinality
        parsed_df = df.select(
            from_json(col('value').cast('string'), schema).alias('data')
        ).select('data.*')
        validated_df = parsed_df.withColumn(
            'event_type', lower(trim(col('event_type')))
        ).filter(col('event_type').isin(allowed_event_types))

        # Process data
        processed_df = validated_df.groupBy('event_type').count()

        # Write to console (for demo purposes)
        query = processed_df.writeStream \\
            .outputMode('complete') \\
            .format('console') \\
            .start()

        # Wait for processing to finish
        query.awaitTermination(30)  # Run for 30 seconds
        spark.stop()
        print('Spark processing completed')
        "
        """,
    )

    # Task 6: Run Spark ETL with Delta Lake support
    # Decoupled via RemoteSparkSubmitOperator (routes to remote cluster or local fallback)
    t5_spark_etl = RemoteSparkSubmitOperator(
        task_id="run_spark_etl_with_delta",
        bash_command=(
            "cd /opt/airflow/movie-rec && python etl/pyspark_etl.py "
            '--date {{ ds }} --run-id "{{ run_id }}" --sink delta '
            '--tenant-id "${NOVA_TENANT_ID:-demo-media-co}" '
            '--catalog-id "${NOVA_CATALOG_ID:-tmdb-movies}" '
            "--source-system tmdb_kaggle"
        ),
    )

    # DAG Flow
    t0_check_kafka >> t1_check_spark >> t2_create_topic >> t3_produce_events >> t4_process_events >> t5_spark_etl
