# Databricks notebook source
# MAGIC %md
# MAGIC # 01b - Real-Time Streaming Ingest (Auto Loader & Micro-Batching)
# MAGIC
# MAGIC ## 📌 Overview & Streaming Architecture
# MAGIC This notebook runs 24/7 (or on micro-batch schedule) using **Spark Structured Streaming**.
# MAGIC It watches Unity Catalog Volume storage (`/Volumes/apex/default/secrets/events_raw/`) for JSON event logs dropped by external APIs/webhooks (e.g. Cloudflare Workers / FastAPI).
# MAGIC
# MAGIC ### 💡 Core Streaming Patterns:
# MAGIC 1. **Databricks Auto Loader (`cloudFiles`):** Natively discovers and ingests new files as they arrive with automatic schema inference and evolution.
# MAGIC 2. **Checkpointing:** Tracks processed offsets in `/Volumes/apex/default/secrets/checkpoints/` ensuring **exactly-once processing semantics** across cluster restarts.
# MAGIC 3. **Delta Micro-Batching:** Streams incoming interaction logs directly into the managed Delta table `apex.default.user_events`.

# COMMAND ----------
# COMMAND ----------
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType
from pyspark.sql.functions import col, from_json, current_timestamp

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
    Processes each 10-second micro-batch stream partition.

    📌 EDGE CASE 2: Empty Micro-Batch Check
    - IF microBatchDF is empty (0 new JSON events dropped): Skip transaction write to avoid creating empty 0-record Delta commits.
    - ELSE: Append non-empty event batch to Silver Delta table 'apex.default.user_events'.
    """
    if microBatchDF.rdd.isEmpty():
        print(f"Micro-batch {batchId}: 0 new events. Skipping write.")
        return

    print(f"Micro-batch {batchId}: Appending incoming streaming interaction events to {silver_table_name}...")
    microBatchDF.write.format("delta").mode("append").saveAsTable(silver_table_name)

query = (
    streaming_df.writeStream
    .foreachBatch(merge_microbatch)
    .option("checkpointLocation", checkpoint_path)
    .trigger(processingTime="10 seconds")  # Real-time 10-second micro-batches
    .start()
)

print(f"Streaming job initialized successfully. Auto Loader is watching {raw_events_path}...")
# Note: Do not awaitTermination() here if running as a Databricks Job, 
# unless the job is specifically configured as a Continuous cluster.
