# Kafka Cluster Setup for Medallion Architecture

This document explains the Kafka cluster setup for the Movie Recommendation System, designed to support streaming ingestion for the Bronze layer of the Medallion Architecture.

## Overview

The Kafka cluster provides a fault-tolerant, scalable streaming platform for ingesting raw data into the Bronze layer. This setup replaces the previous single-node Kafka instance with a 3-broker cluster for improved availability and data durability.

## Architecture

### Kafka Cluster Configuration

- **3 Kafka brokers** (kafka-1, kafka-2, kafka-3) for fault tolerance
- **KRaft mode** (no ZooKeeper dependency)
- **Replication factor 2** for all topics
- **3 partitions** per topic for parallel processing
- **Kafka UI** for monitoring and management

### Bronze Layer Topics

The following topics are configured for raw data ingestion:

| Topic Name            | Description                                  | Partitions | Replication | Retention  | Use Case                          |
|-----------------------|----------------------------------------------|------------|-------------|------------|-----------------------------------|
| `raw_movie_events`    | User interactions (views, ratings, etc.)     | 3          | 2           | 7 days     | User behavior tracking            |
| `raw_movie_metadata`  | Movie information updates                    | 3          | 2           | 30 days    | Movie data ingestion              |
| `raw_user_data`       | User profile updates                         | 3          | 2           | 30 days    | User profile management           |
| `raw_system_logs`     | System monitoring and logs                   | 3          | 2           | 30 days    | System health monitoring          |

## Setup Instructions

### 1. Start the Kafka Cluster

```bash
docker-compose -f docker-compose.kafka-cluster.yml up -d
```

### 2. Set Up Bronze Layer Topics

```bash
# Make the script executable
chmod +x scripts/setup_kafka_topics.sh

# Run the topic setup script
./scripts/setup_kafka_topics.sh
```

### 3. Verify the Setup

```bash
# Check running containers
docker-compose -f docker-compose.kafka-cluster.yml ps

# Check Kafka cluster health
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 --describe

# Access Kafka UI at http://localhost:8082
```

## Integration with Medallion Architecture

### Bronze Layer Integration

The Kafka cluster is designed to feed data into the Bronze layer of the Medallion Architecture:

1. **Data Ingestion**: Raw data is produced to Kafka topics
2. **Stream Processing**: Spark consumes data from Kafka topics
3. **Storage**: Processed data is stored in the Bronze layer (`data/bronze/`)
4. **Pipeline Orchestration**: Airflow manages the data pipeline workflows

### Spark Integration

Spark is configured to consume from the Kafka cluster:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName('BronzeLayerProcessing') \
    .master('spark://spark:7077') \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0') \
    .getOrCreate()

# Read from Kafka topic
df = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'kafka-1:9092,kafka-2:9092,kafka-3:9092') \
    .option('subscribe', 'raw_movie_events') \
    .load()
```

### Airflow Integration

Airflow DAGs can manage Kafka topic creation and data processing:

```python
from airflow.operators.bash import BashOperator

create_topic_task = BashOperator(
    task_id='create_bronze_topic',
    bash_command="""
    docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 \
      --create --if-not-exists \
      --topic raw_movie_events \
      --partitions 3 \
      --replication-factor 2
    """,
)
```

## Monitoring and Management

### Kafka UI

Access the Kafka UI at `http://localhost:8082` to:
- Monitor cluster health
- View and manage topics
- Inspect messages
- Monitor consumer groups

### Health Checks

Each Kafka broker includes health checks that verify:
- Broker responsiveness
- Topic listing capability
- Network connectivity

## Best Practices

1. **Data Retention**: Configure appropriate retention policies based on data importance
2. **Monitoring**: Regularly monitor cluster health and disk usage
3. **Scaling**: Add more brokers as data volume grows
4. **Security**: Implement authentication and encryption for production use
5. **Backup**: Regularly back up Kafka data volumes

## Troubleshooting

### Common Issues

**1. Broker connection issues**
```bash
# Check if brokers are running
docker-compose -f docker-compose.kafka-cluster.yml ps

# Check broker logs
docker-compose -f docker-compose.kafka-cluster.yml logs kafka-1
docker-compose -f docker-compose.kafka-cluster.yml logs kafka-2
docker-compose -f docker-compose.kafka-cluster.yml logs kafka-3
```

**2. Topic creation failures**
```bash
# Manually create a topic
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 --create --topic test_topic --partitions 1 --replication-factor 1

# List existing topics
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 --list
```

**3. Cluster formation issues**
- Ensure all brokers can communicate on ports 9092 (data) and 9093 (controller)
- Check network connectivity between containers
- Verify volume mounts for data persistence

## Future Enhancements

1. **Schema Registry**: Add Confluent Schema Registry for data governance
2. **Kafka Connect**: Implement Kafka Connect for easier data integration
3. **Monitoring**: Add Prometheus and Grafana for advanced monitoring
4. **Security**: Implement SASL/SSL for secure communication
5. **Scaling**: Add more brokers for increased capacity
