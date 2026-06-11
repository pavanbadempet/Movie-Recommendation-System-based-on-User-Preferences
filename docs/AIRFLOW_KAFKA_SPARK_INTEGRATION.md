# Airflow, Kafka, and Spark Integration

This document explains how Airflow, Kafka, and Spark are integrated in the Movie Recommendation System to create a robust data pipeline orchestration platform.

## Overview

The system now includes:

- **Apache Airflow**: For workflow orchestration and scheduling
- **Apache Kafka**: For event streaming and real-time data processing
- **Apache Spark**: For distributed data processing and analytics
- **Delta Lake**: For reliable data lake storage

## Architecture

```
[Data Sources] → [Kafka] → [Spark] → [Delta Lake/Parquet] → [Recommendation Engine]
       ↑
[Airflow DAGs] → Orchestrates the entire pipeline
```

## Services Added

### Kafka Service

- **Image**: `bitnami/kafka:3.6`
- **Port**: `9092` (accessible within Docker network)
- **Configuration**:
  - Single node with controller and broker roles
  - Auto-creation of topics enabled
  - Persistent volume for data storage

### Spark Service

- **Image**: `bitnami/spark:3.5`
- **Ports**:
  - `8081`: Spark UI
  - `7077`: Spark master port
- **Configuration**:
  - Standalone mode
  - Shared volumes for data and models
  - Access to the movie recommendation codebase

## Airflow Configuration

Airflow has been configured with the necessary dependencies:

- `kafka-python`: For Kafka integration
- `pyspark`: For Spark integration
- `delta-spark`: For Delta Lake support

These dependencies are installed in all Airflow components:
- Airflow Webserver
- Airflow Scheduler
- Airflow Init

## DAGs

### 1. Kafka Spark Integration DAG (`kafka_spark_integration`)

This DAG demonstrates the complete integration flow:

1. **Check Kafka Connection**: Verifies Kafka is accessible
2. **Check Spark Connection**: Verifies Spark is accessible
3. **Create Kafka Topic**: Creates a `movie_events` topic
4. **Produce Movie Events**: Simulates user events (views, ratings)
5. **Process Events with Spark**: Reads from Kafka, processes with Spark
6. **Run Spark ETL with Delta**: Processes data using Delta Lake format

### 2. Movie Data Refresh DAG (`movie_data_refresh`)

The main ETL pipeline now includes:

1. **Download from Kaggle**: Gets raw movie data
2. **Run Spark ETL**: Processes data with Spark (Parquet format)
3. **Run Spark ETL with Delta**: Optional Delta Lake format processing
4. **Rebuild Index**: Updates recommendation indexes

## Usage

### Starting the Services

```bash
docker-compose up -d
```

### Accessing Services

- **Airflow UI**: http://localhost:8080
- **Spark UI**: http://localhost:8081

### Running the Integration DAG

1. Access the Airflow UI
2. Enable the `kafka_spark_integration` DAG
3. Trigger the DAG manually or wait for the scheduled run

## Development

### Adding New Kafka Topics

```python
# In your DAG file
create_topic_task = BashOperator(
    task_id='create_new_topic',
    bash_command="""
    kafka-topics.sh --bootstrap-server kafka:9092 \
      --create --if-not-exists \
      --topic new_topic \
      --partitions 3 \
      --replication-factor 1
    """,
)
```

### Producing Events to Kafka

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

producer.send('movie_events', {'id': 1, 'title': 'Movie Title', 'event_type': 'view'})
producer.flush()
producer.close()
```

### Processing with Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('KafkaProcessing') \
    .master('spark://spark:7077') \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'kafka:9092') \
    .option('subscribe', 'movie_events') \
    .load()
```

## Best Practices

1. **Error Handling**: Always include proper error handling for Kafka and Spark connections
2. **Resource Management**: Monitor Spark resource usage and adjust configurations as needed
3. **Data Quality**: Implement data quality checks in your Spark jobs
4. **Idempotency**: Design DAGs to be idempotent for reliable retries
5. **Monitoring**: Use Airflow's built-in monitoring and Spark UI for performance tracking

## Troubleshooting

### Kafka Connection Issues

- Verify Kafka is running: `docker-compose ps`
- Check Kafka logs: `docker-compose logs kafka`
- Test Kafka connection manually: `docker exec -it kafka kafka-topics.sh --bootstrap-server localhost:9092 --list`

### Spark Connection Issues

- Verify Spark is running: `docker-compose ps`
- Check Spark logs: `docker-compose logs spark`
- Test Spark connection manually: `docker exec -it spark spark-submit --version`

### Airflow Dependency Issues

- Check Airflow logs for missing dependencies: `docker-compose logs airflow-webserver`
- Rebuild Airflow containers if dependencies are missing: `docker-compose up -d --build`

## Future Enhancements

1. **Multi-node Kafka cluster** for higher availability
2. **Spark cluster mode** for better scalability
3. **Kafka Connect** for easier data integration
4. **Schema Registry** for better data governance
5. **Airflow Providers** for native Kafka and Spark operators
