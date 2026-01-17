# Enterprise Deployment Guide 🚀

This project is architected to be "Cloud-Agnostic". While it runs locally with Docker/Airflow, it is designed for a seamless "Lift & Shift" to Enterprise Data Platforms like **Databricks** or **Snowflake**.

## 1. Databricks (The Lakehouse)

### Why?
Databricks offers managed Spark, optimized runtime (Photon), and Delta Lake for ACID transactions.

### Deployment Steps
1.  **Repo Integration**: Connect your Databricks Workspace to this GitHub repository.
2.  **Cluster Setup**:
    - Runtime: **14.3 LTS ML** (Includes PyTorch, Sentence-Transformers pre-installed).
    - Instance: `i3.xlarge` (AWS) or `Standard_DS3_v2` (Azure).
3.  **Job Configuration**:
    - Type: `Python Script`
    - Path: `etl/pyspark_etl.py`
    - Parameters: `["--sink", "delta", "--date", "{{job.start_date}}"]`

### Code Adaptation
The code already supports Delta:
```python
# etl/pyspark_etl.py
if format_type == "delta":
    writer.format("delta").save("s3://my-datalake/movies_delta")
```

---

## 2. Snowflake (The Data Cloud)

### Why?
Snowflake provides zero-maintenance data warehousing and is increasingly supporting Python/Spark workloads via **Snowpark**.

### Deployment Steps (Snowpark)
1.  **Upload Code**: Upload `etl/` folder to a Snowflake Stage `@MY_STAGE`.
2.  **Create Stored Procedure**:
    ```sql
    CREATE OR REPLACE PROCEDURE RUN_MOVIE_ETL()
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = ('snowflake-snowpark-python', 'sentence-transformers')
    IMPORTS = ('@MY_STAGE/etl/pyspark_etl.py')
    HANDLER = 'etl.pyspark_etl.run_spark_etl';
    ```
3.  **Execute**:
    ```sql
    CALL RUN_MOVIE_ETL();
    ```

### Legacy Integration (Spark Connector)
If running Spark externally (EMR/Glue) and writing to Snowflake:
- The code supports `sink_format="snowflake"`.
- Ensure `sfUrl`, `sfUser`, `sfPassword` are passed via environment variables or Secrets Manager.

---

## 3. AWS Architecture (Hybrid)

- **Orchestration**: Managed Airflow (MWAA).
- **Processing**: EMR Serverless (running `pyspark_etl.py`).
- **Storage**: S3 (Parquet/Iceberg).

**Configuration**:
 Simply update `.env` or `config.py` environment variables:
```bash
RAW_DATA_PATH=s3://my-corp-bucket/raw
PROCESSED_DATA_PATH=s3://my-corp-bucket/processed
```
