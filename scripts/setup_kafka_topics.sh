#!/bin/bash
#
# Script to set up Kafka topics for Bronze layer in Medallion Architecture
# Run this script after starting the Kafka cluster

echo "Setting up Kafka topics for Bronze layer..."

# Wait for Kafka brokers to be ready
echo "Waiting for Kafka brokers to start..."
sleep 30

# Create topics for Bronze layer
echo "Creating Bronze layer topics..."

# Raw movie events topic (user interactions like views, ratings, etc.)
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 \
  --create --if-not-exists \
  --topic raw_movie_events \
  --partitions 3 \
  --replication-factor 2 \
  --config retention.ms=604800000 \  # 7 days retention
  --config cleanup.policy=delete

# Raw movie metadata topic (movie information updates)
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 \
  --create --if-not-exists \
  --topic raw_movie_metadata \
  --partitions 3 \
  --replication-factor 2 \
  --config retention.ms=2592000000 \  # 30 days retention
  --config cleanup.policy=delete

# Raw user data topic (user profile updates)
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 \
  --create --if-not-exists \
  --topic raw_user_data \
  --partitions 3 \
  --replication-factor 2 \
  --config retention.ms=2592000000 \  # 30 days retention
  --config cleanup.policy=delete

# Raw system logs topic (system monitoring and logs)
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 \
  --create --if-not-exists \
  --topic raw_system_logs \
  --partitions 3 \
  --replication-factor 2 \
  --config retention.ms=2592000000 \  # 30 days retention
  --config cleanup.policy=delete

# Verify topics were created
echo "Verifying created topics..."
docker exec kafka-1 kafka-topics.sh --bootstrap-server kafka-1:9092 --list

echo "Kafka Bronze layer topics setup completed successfully!"