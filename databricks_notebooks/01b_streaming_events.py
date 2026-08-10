# Databricks notebook source
# MAGIC %md
# MAGIC # 01b - Real-Time Streaming Ingest (Auto Loader)
# MAGIC This notebook runs 24/7 (or micro-batch) using Spark Structured Streaming.
# MAGIC It watches the raw events storage for the JSON files dropped by the Zerobus API (Cloudflare Workers) and merges them into the Silver interaction tables instantly.

# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, MapType
from pyspark.sql.functions import col, from_json

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Schema & Paths

# COMMAND ----------
# The location where Cloudflare Workers drops the JSON event logs
raw_events_path = "/Volumes/apex/default/secrets/events_raw/"
checkpoint_path = "/Volumes/apex/default/secrets/checkpoints/events_silver/"
silver_table_name = "apex.default.user_events"

# Define the schema expected from the FastAPI/Cloudflare webhook
event_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("movie_id", StringType(), True),
    StructField("interaction_type", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("metadata", StringType(), True)  # JSON string
])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Start Auto Loader Stream

# COMMAND ----------
print(f"Starting Auto Loader Stream on {raw_events_path}...")

# 1. Read Stream using Auto Loader (cloudFiles)
# This perfectly replaces Kafka, achieving Zerobus real-time streaming
streaming_df = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", checkpoint_path + "schema")
    .schema(event_schema)
    .load(raw_events_path)
)

# 2. Write Stream to Delta Lake (Silver Layer)
def merge_microbatch(microBatchDF, batchId):
    # Append incoming real-time interaction events to the Delta Managed Table
    microBatchDF.write.format("delta").mode("append").saveAsTable(silver_table_name)
    
query = (
    streaming_df.writeStream
    .foreachBatch(merge_microbatch)
    .option("checkpointLocation", checkpoint_path)
    .trigger(processingTime="10 seconds") # Real-time micro-batches
    .start()
)

print(f"Streaming job initialized. Waiting for events...")
# Note: Do not awaitTermination() here if running as a Databricks Job, 
# unless the job is specifically configured as a Continuous cluster.
