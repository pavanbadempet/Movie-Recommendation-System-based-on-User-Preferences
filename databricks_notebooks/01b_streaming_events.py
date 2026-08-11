# Databricks notebook source
# MAGIC %md
# MAGIC # 01b - Real-Time Streaming Ingest (Auto Loader & Micro-Batching)
# MAGIC
# MAGIC ## Overview & Message Provenance Architecture
# MAGIC
# MAGIC ### 📥 WHO BRINGS THE MESSAGES & HOW HUGGING FACE SPACES CONNECTS:
# MAGIC 1. **Message Source (Hugging Face Spaces UI):** When a user clicks, rates, or searches for a movie on HF Spaces, the frontend web server (FastAPI/Node) creates a JSON event payload:
# MAGIC    `{"user_id": "usr_9921", "movie_id": "101", "interaction_type": "click", "timestamp": "2026-08-11T13:25:00Z"}`
# MAGIC 2. **Zero-Broker Volume Ingestion (REST API):** HF Spaces calls the Databricks Files REST API (`POST /api/2.0/fs/files/Volumes/apex/default/secrets/events_raw/event_{uuid}.json`) to drop the JSON event into Unity Catalog Volume storage without requiring an expensive Kafka cluster.
# MAGIC 3. **Auto Loader Ingestion (`cloudFiles`):** This notebook's Auto Loader stream automatically detects new `.json` files in `/Volumes/apex/default/secrets/events_raw/` and appends them to `apex.default.user_events`.
# MAGIC
# MAGIC ### Core Streaming Design:
# MAGIC 1. **Databricks Auto Loader (`cloudFiles`):** Natively discovers and ingests new files as they arrive with automatic schema inference.
# MAGIC 2. **Checkpointing:** Tracks processed file offsets in `/Volumes/apex/default/secrets/checkpoints/` ensuring **exactly-once processing semantics**.
# MAGIC 3. **Delta Micro-Batching (`availableNow=True`):** Processes incoming interaction logs into the Silver table `apex.default.user_events` and exits cleanly.

# COMMAND ----------
# MAGIC %run ./doppler_config

# COMMAND ----------
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType, IntegerType, LongType, DoubleType
from pyspark.sql.functions import col, from_json, current_timestamp

try:
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    dbutils.widgets.text("ENVIRONMENT", "dev", "Deployment Environment (dev, stg, prd)")
    env = dbutils.widgets.get("ENVIRONMENT")
    secrets = load_centralized_doppler_secrets(dbutils=dbutils, env=env)
    events_db_url = secrets.get("DATABASE_URL_EVENTS")
except Exception:
    events_db_url = None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Schema & Directory Setup

# COMMAND ----------
raw_events_path = "/Volumes/apex/default/secrets/events_raw/"
checkpoint_path = "/Volumes/apex/default/secrets/checkpoints/events_silver/"
silver_table_name = "apex.default.user_events"

# EDGE CASE 1: Directory Existence Check
# Ensures Volume event directory and checkpoint directory exist before stream starts
os.makedirs(raw_events_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

# Define expected schema from FastAPI / Cloudflare Worker webhook payload
# 📥 STREAM INPUT EXAMPLE (JSON):
# {"user_id": "usr_9921", "movie_id": "101", "interaction_type": "click", "timestamp": "2026-08-10T22:00:00Z"}
event_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("movie_id", StringType(), True),
    StructField("interaction_type", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("metadata", StringType(), True)  # JSON string payload
])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Start Auto Loader Stream

# COMMAND ----------
print(f"Starting Auto Loader Stream on {raw_events_path}...")

# 1. Read Stream using Databricks Auto Loader (cloudFiles)
streaming_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path + "schema")
    .schema(event_schema)
    .load(raw_events_path)
    .withColumn("_ingested_at", current_timestamp())
)

# 2. Write Micro-Batch Stream to Delta Lake Managed Table
def merge_microbatch(microBatchDF, batchId):
    """
    Processes each micro-batch stream partition cleanly.

    📌 EDGE CASE 2: Empty Micro-Batch Check
    - IF microBatchDF is empty (0 new JSON events dropped): Skip transaction write to avoid creating empty 0-record Delta commits.
    - ELSE: Append non-empty event batch to Silver Delta table 'apex.default.user_events'.
    """
    if microBatchDF.limit(1).count() == 0:
        print(f"Micro-batch {batchId}: 0 new events. Skipping write.")
        return

    print(f"Micro-batch {batchId}: Appending incoming streaming interaction events to {silver_table_name}...")
    microBatchDF.write.format("delta").mode("append").saveAsTable(silver_table_name)

    # Optional Sync to Neon Clickstream DB (Account 2)
    if events_db_url:
        try:
            print(f"Micro-batch {batchId}: Syncing events to Neon Clickstream DB...")
            import psycopg2
            from psycopg2.extras import execute_values
            rows = microBatchDF.select("user_id", "movie_id", "interaction_type", "timestamp", "metadata").collect()
            if rows:
                conn = psycopg2.connect(events_db_url)
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_events (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100),
                        movie_id VARCHAR(100),
                        interaction_type VARCHAR(100),
                        timestamp VARCHAR(100),
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                insert_query = """
                    INSERT INTO user_events (user_id, movie_id, interaction_type, timestamp, metadata)
                    VALUES %s;
                """
                data_tuples = [(r.user_id, r.movie_id, r.interaction_type, r.timestamp, r.metadata) for r in rows]
                execute_values(cur, insert_query, data_tuples)
                conn.commit()
                cur.close()
                conn.close()
                print(f"Synced {len(data_tuples)} events to Neon Clickstream DB.")
        except Exception as e:
            print(f"Warning: Could not sync events to Neon: {e}")

query = (
    streaming_df.writeStream
    .foreachBatch(merge_microbatch)
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)  # Databricks Serverless native batch trigger
    .start()
)

print(f"Streaming job initialized successfully. Auto Loader is watching {raw_events_path}...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Visual Verification & Querying (Silver Streaming Events Table)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT _ingested_at, user_id, movie_id, interaction_type, timestamp
# MAGIC FROM apex.default.user_events
# MAGIC ORDER BY _ingested_at DESC
# MAGIC LIMIT 10;
