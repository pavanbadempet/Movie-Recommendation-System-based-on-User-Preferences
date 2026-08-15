"""
Spark Structured Streaming ingestion for product behavior events.

Catalog refresh is batch because the source changes daily. Product behavior is
the real streaming use case: searches, views, clicks, ratings, and impressions
arrive continuously and become ranking/personalization signals.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    coalesce,
    col,
    concat_ws,
    current_timestamp,
    from_json,
    lit,
    sha2,
    to_date,
    to_timestamp,
    udf,
)
from pyspark.sql.types import BooleanType, FloatType, StringType, StructField, StructType

from etl.config import paths
from etl.delta_lakehouse import get_delta_table
from etl.pyspark_etl import create_spark_session

logger = logging.getLogger(__name__)

EVENT_PAYLOAD_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("event_ts", StringType(), True),
        StructField("event_type", StringType(), False),
        StructField("tenant_id", StringType(), True),
        StructField("catalog_id", StringType(), True),
        StructField("content_id", StringType(), True),
        StructField("source_content_id", StringType(), True),
        StructField("movie_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("request_id", StringType(), True),
        StructField("rating", FloatType(), True),
        StructField("query_text", StringType(), True),
        StructField("source", StringType(), True),
    ]
)


class LocalSchemaRegistryClient:
    """
    Mock/Local Confluent Schema Registry Client.
    Validates streaming records against defined JSON schemas to prevent schema drift
    before Delta Lake appends.
    """

    def __init__(self, schemas_dir: Path | None = None):
        if schemas_dir is None:
            self.schemas_dir = Path(__file__).resolve().parent.parent / "contracts"
        else:
            self.schemas_dir = schemas_dir
        self._schemas: dict[str, dict[str, Any]] = {}

    def get_latest_schema(self, subject: str) -> dict:
        """Retrieve the latest schema definition for a subject/topic."""
        if subject not in self._schemas:
            schema_path = self.schemas_dir / f"{subject}.schema.json"
            if not schema_path.exists():
                schema_path = self.schemas_dir / "raw_events.schema.json"
            with open(schema_path, encoding="utf-8") as f:
                self._schemas[subject] = json.load(f)
        return self._schemas[subject]

    def validate_record(self, record: dict, subject: str) -> bool:
        """Validate an individual record dictionary against the registered schema."""
        try:
            schema = self.get_latest_schema(subject)
            required = schema.get("required_columns", [])
            for col in required:
                if col not in record or record[col] is None:
                    return False

            columns = schema.get("columns", {})
            for col_name, rules in columns.items():
                if col_name not in record or record[col_name] is None:
                    continue
                val = record[col_name]
                expected_type = rules.get("type")
                if expected_type == "string" and not isinstance(val, str):
                    return False
                elif expected_type == "integer" and not isinstance(val, (int, float)):
                    return False
                elif expected_type == "number" and not isinstance(val, (int, float)):
                    return False
                elif expected_type == "boolean" and not isinstance(val, bool):
                    return False
            return True
        except Exception:
            return False


# Global client instance
schema_registry = LocalSchemaRegistryClient()


@udf(BooleanType())
def validate_event_payload(payload_str: str, subject: str = "raw_events") -> bool:
    """Spark UDF to validate event payload string against the schema registry contract."""
    if not payload_str:
        return False
    try:
        record = json.loads(payload_str)
        return schema_registry.validate_record(record, subject)
    except Exception:
        return False


def parse_kafka_event_stream(raw_stream: DataFrame) -> DataFrame:
    """Parse Kafka JSON values into the canonical tenant-aware event fact schema."""
    # Enforce Schema Registry contract validation filter
    compliant_stream = raw_stream.filter(validate_event_payload(col("value").cast("string"), lit("raw_events")))

    parsed = compliant_stream.select(
        from_json(col("value").cast("string"), EVENT_PAYLOAD_SCHEMA).alias("event"),
        col("timestamp").alias("kafka_timestamp"),
    ).select("event.*", "kafka_timestamp")

    event_ts = coalesce(to_timestamp(col("event_ts")), col("kafka_timestamp"), current_timestamp())
    source_content_id = coalesce(col("source_content_id"), col("movie_id").cast("string"))
    content_id = coalesce(
        col("content_id"),
        sha2(
            concat_ws(
                "||",
                coalesce(col("tenant_id"), lit("demo-media-co")),
                coalesce(col("catalog_id"), lit("tmdb-movies")),
                source_content_id,
            ),
            256,
        ),
    )

    return parsed.select(
        coalesce(col("tenant_id"), lit("demo-media-co")).alias("tenant_id"),
        coalesce(col("catalog_id"), lit("tmdb-movies")).alias("catalog_id"),
        coalesce(
            col("event_id"),
            sha2(concat_ws("||", col("event_type"), event_ts.cast("string"), content_id, col("user_id")), 256),
        ).alias("event_id"),
        event_ts.alias("event_ts"),
        col("event_type"),
        content_id.alias("content_id"),
        source_content_id.alias("source_content_id"),
        col("user_id"),
        col("session_id"),
        col("request_id"),
        col("rating").cast("float"),
        col("query_text"),
        col("source"),
        to_date(event_ts).cast("string").alias("event_date"),
    )


def start_kafka_events_to_delta(
    spark: SparkSession,
    kafka_bootstrap_servers: str,
    topic: str = "nova.content_events",
    checkpoint_location: str | None = None,
    processing_time: str = "30 seconds",
):
    """Start a streaming query from Kafka behavior events into Delta Lake."""
    table = get_delta_table("gold.fact_content_event")
    if checkpoint_location is None:
        checkpoint_location = (
            f"{paths.logs.rstrip('/')}/checkpoints/content_events"
            if isinstance(paths.logs, str)
            else str(paths.logs / "checkpoints" / "content_events")
        )

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    events_df = parse_kafka_event_stream(raw_stream)
    logger.info("Starting Kafka -> Delta event stream: topic=%s output=%s", topic, table.path)

    return (
        events_df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_location)
        .partitionBy(*table.partition_columns)
        .queryName("nova_content_event_ingest")
        .trigger(processingTime=processing_time)
        .start(table.path)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream product behavior events from Kafka into Delta.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="nova.content_events")
    parser.add_argument("--checkpoint-location", default=None)
    parser.add_argument("--processing-time", default="30 seconds")
    parser.add_argument("--await-seconds", type=int, default=0, help="0 means run until terminated.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session(app_name="NovaContentEventStreaming", enable_delta=True)
    query = start_kafka_events_to_delta(
        spark,
        kafka_bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        checkpoint_location=args.checkpoint_location,
        processing_time=args.processing_time,
    )

    if args.await_seconds > 0:
        query.awaitTermination(args.await_seconds)
        query.stop()
        spark.stop()
    else:
        query.awaitTermination()


if __name__ == "__main__":
    main()
