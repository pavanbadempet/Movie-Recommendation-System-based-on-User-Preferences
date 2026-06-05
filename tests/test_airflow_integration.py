#!/usr/bin/env python3
"""
Test script to verify Airflow can interact with Kafka and Spark services.
This script tests the connectivity between services in the Docker Compose setup.
"""

import logging
import os
import sys

import pytest

if os.getenv("RUN_INFRA_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Live Kafka/Spark integration tests require RUN_INFRA_INTEGRATION_TESTS=1",
        allow_module_level=True,
    )

kafka = pytest.importorskip("kafka")
KafkaAdminClient = kafka.KafkaAdminClient
KafkaProducer = kafka.KafkaProducer
KafkaError = pytest.importorskip("kafka.errors").KafkaError
SparkSession = pytest.importorskip("pyspark.sql").SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_kafka_connection():
    """Test Kafka connection from Airflow context"""
    try:
        admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092", client_id="integration-test")
        topics = admin_client.list_topics()
        admin_client.close()
        logger.info(f"✓ Kafka connection successful. Topics: {topics}")
        return True
    except KafkaError as e:
        logger.error(f"✗ Kafka connection failed: {e}")
        return False


def test_spark_connection():
    """Test Spark connection from Airflow context"""
    try:
        spark = (
            SparkSession.builder.appName("IntegrationTest")
            .master("spark://spark:7077")
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )

        spark_version = spark.version
        spark.stop()
        logger.info(f"✓ Spark connection successful. Version: {spark_version}")
        return True
    except Exception as e:
        logger.error(f"✗ Spark connection failed: {e}")
        return False


def test_kafka_producer():
    """Test producing messages to Kafka"""
    try:
        producer = KafkaProducer(bootstrap_servers="kafka:9092", value_serializer=lambda v: str(v).encode("utf-8"))

        # Create test topic if it doesn't exist
        admin_client = KafkaAdminClient(bootstrap_servers="kafka:9092", client_id="integration-test")

        # Test producing a message
        producer.send("test_topic", value="test_message")
        producer.flush()
        producer.close()
        admin_client.close()

        logger.info("✓ Kafka message production successful")
        return True
    except Exception as e:
        logger.error(f"✗ Kafka message production failed: {e}")
        return False


def test_spark_kafka_integration():
    """Test Spark reading from Kafka"""
    try:
        spark = (
            SparkSession.builder.appName("KafkaIntegrationTest")
            .master("spark://spark:7077")
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
            .getOrCreate()
        )

        # Test reading from Kafka (this will fail if the connector is not available)
        df = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", "kafka:9092")
            .option("subscribe", "test_topic")
            .option("startingOffsets", "earliest")
            .load()
        )

        # Just test the schema to verify the connector works
        _ = df.schema
        spark.stop()

        logger.info("✓ Spark Kafka connector available and working")
        return True
    except Exception as e:
        logger.error(f"✗ Spark Kafka integration failed: {e}")
        return False


def main():
    """Run all integration tests"""
    logger.info("Starting Airflow-Kafka-Spark integration tests...")

    tests = [
        ("Kafka Connection", test_kafka_connection),
        ("Spark Connection", test_spark_connection),
        ("Kafka Producer", test_kafka_producer),
        ("Spark Kafka Integration", test_spark_kafka_integration),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n=== Integration Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        logger.info("\n✓ All integration tests passed!")
        return 0
    else:
        logger.error("\n✗ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
