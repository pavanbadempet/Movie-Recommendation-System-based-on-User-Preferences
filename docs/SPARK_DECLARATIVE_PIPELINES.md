# Apache Spark Declarative Pipelines (SDP) Architecture

The **AI Recommendation System** leverages **Apache Spark Declarative Pipelines (SDP)** (compatible with Spark 4.1 & Delta Live Tables) to define data flow transitions declaratively through configuration specifications without imperative glue code.

---

## ⚡ Declarative Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                SPARK DECLARATIVE PIPELINE (SDP)                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Declarative Spec: `config/spark_declarative_pipeline.yaml` defines catalog, target schema, │
│     Medallion tables (Bronze -> Silver -> Gold), and data quality expectations.                 │
│  2. Pipeline Executor: `etl/spark_declarative_pipeline.py` parses spec, compiles topological   │
│     DAG execution plan, and enforces schema expectations.                                      │
│  3. Multi-Layer Transformations:                                                               │
│     • Bronze: Raw JSON/CSV file ingestion.                                                     │
│     • Silver: Cleansed, validated, and SCD Type 2 dimension history.                           │
│     • Gold: Aggregated feature embeddings with Z-Ordering compaction.                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 Declarative Spec Example (`config/spark_declarative_pipeline.yaml`)

```yaml
pipeline_id: "apex_movie_recommendation_pipeline"
version: "1.0"
catalog: "main"
target_schema: "recommendations"

tables:
  - name: "bronze_raw_movies"
    layer: "BRONZE"
    format: "delta"
    source:
      format: "parquet"
      path: "data/bronze/movies_raw"

  - name: "silver_curated_movies"
    layer: "SILVER"
    format: "delta"
    source_table: "bronze_raw_movies"
    expectations:
      - name: "valid_movie_id"
        expr: "movie_id IS NOT NULL AND movie_id > 0"
        action: "FAIL"
    scd_type: 2

  - name: "gold_movie_features"
    layer: "GOLD"
    format: "delta"
    source_table: "silver_curated_movies"
    z_order_by: ["movie_id", "genres"]
```

---

## 🧪 Verification

```bash
$ python -m pytest tests/test_spark_declarative_pipeline.py
============================== 4 passed in 0.25s ==============================
```
