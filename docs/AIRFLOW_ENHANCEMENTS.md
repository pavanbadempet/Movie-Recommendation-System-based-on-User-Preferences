# Airflow Integration Enhancements

This document describes the enhancements made to the Airflow integration in the Movie Recommendation System to ensure proper interaction with Kafka and Spark services.

## Summary of Changes

### 1. Docker Compose Configuration Updates

**Enhanced Airflow Services:**
- Added `spark-sql-kafka-0-10_2.12:3.5.0` package to all Airflow containers (init, webserver, scheduler)
- This package enables Spark to read from Kafka topics within Airflow tasks

**Network Configuration:**
- Added explicit `networks` configuration to all services
- Created a dedicated `default` network for cross-service communication
- Services can now communicate using service names as hostnames (e.g., `kafka:9092`, `spark://spark:7077`)

**Kafka Service Enhancements:**
- Added additional Kafka configuration parameters:
  - `KAFKA_CFG_BROKER_ID=0`
  - `KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT`
  - `KAFKA_CFG_INTER_BROKER_LISTENER_NAME=PLAINTEXT`
- Added network configuration for proper service discovery

**Spark Service Enhancements:**
- Added additional Spark configuration parameters:
  - `SPARK_MASTER_HOST=spark`
  - `SPARK_MASTER_WEBUI_PORT=8080`
- Added dependency on Kafka service
- Added network configuration for proper service discovery

### 2. Package Installation

The following packages are now installed in all Airflow containers:

```bash
pip install kaggle faiss-cpu scikit-learn pandas pyarrow pandera joblib pyspark kafka-python delta-spark spark-sql-kafka-0-10_2.12:3.5.0
```

Additional packages for the scheduler:
```bash
pip install sentence-transformers httpx
```

### 3. Service Discovery

All services can now communicate using these connection strings:

- **Kafka**: `kafka:9092`
- **Spark**: `spark://spark:7077`
- **PostgreSQL**: `postgres:5432`

### 4. Integration Testing

A new test script `tests/test_airflow_integration.py` was created to verify the integration:

- Tests Kafka connection from Airflow context
- Tests Spark connection from Airflow context
- Tests Kafka message production
- Tests Spark Kafka connector availability

## Verification

To verify the integration works correctly:

1. Start the services:
```bash
docker-compose up -d
```

2. Run the integration test:
```bash
docker-compose exec airflow-webserver python /opt/airflow/movie-rec/tests/test_airflow_integration.py
```

3. Check the Airflow UI at http://localhost:8080 and run the `kafka_spark_integration` DAG

## Benefits

1. **Seamless Integration**: Airflow can now fully orchestrate pipelines that involve both Kafka and Spark
2. **Reliable Service Discovery**: Services can communicate using consistent hostnames
3. **Enhanced Capabilities**: Spark jobs can read from Kafka topics directly
4. **Improved Monitoring**: All services are properly connected and can be monitored
5. **Scalable Architecture**: The foundation is now in place for more complex data pipelines

## Usage Examples

### Spark Reading from Kafka in Airflow Task

```python
from pyspark.sql import SparkSession

def process_kafka_events():
    spark = SparkSession.builder \
        .appName("KafkaEventProcessing") \
        .master("spark://spark:7077") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "movie_events") \
        .option("startingOffsets", "earliest") \
        .load()

    # Process data...
    spark.stop()
```

### Kafka Producer in Airflow Task

```python
from kafka import KafkaProducer
import json

def produce_movie_events():
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    producer.send('movie_events', {'id': 1, 'title': 'Movie Title', 'event_type': 'view'})
    producer.flush()
    producer.close()