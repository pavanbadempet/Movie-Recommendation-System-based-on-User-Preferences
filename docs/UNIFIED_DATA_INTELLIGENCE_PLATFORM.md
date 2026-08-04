# Unified Data Intelligence Platform Architecture

The **APEX Movie Recommendation System** implements an open, vendor-neutral **Unified Data & AI Intelligence Platform** matching Databricks platform capabilities 1-to-1 using open enterprise standards: **Apache Spark 4.2**, **Delta Lake 4.3**, **Lakeflow Declarative Ingestion**, **OpenLineage**, and a **Multi-Agent AI Architecture**.

---

## 🏛️ Databricks 1-to-1 Platform Mapping Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               UNIFIED DATA & AI PLATFORM MAPPING                                │
├───────────────────────────┬─────────────────────────────────┬───────────────────────────────────┤
│ DATABRICKS COMPONENT      │ APEX OPEN PLATFORM ENGINE       │ ENTERPRISE SPECIFICATION          │
├───────────────────────────┼─────────────────────────────────┼───────────────────────────────────┤
│ 1. DBSQL / Warehousing    │ Apache Spark 4.2 SQL (AQE)      │ Vectorized Parquet & AQE 2.0      │
│ 2. Lakebase (Serverless)  │ Serverless Postgres / DuckDB    │ Dynamic branching & scaling       │
│ 3. Lakeflow Ingestion     │ Lakeflow Ingestion Engine       │ `etl/lakeflow_pipeline.py`        │
│ 4. Open Table Format      │ Delta Lake 4.3 (`delta-spark`)  │ ACID logs & Z-Ordering            │
│ 5. Data Catalog           │ Unity Catalog Metastore         │ `catalog.schema.table` 3-level    │
│ 6. AgentBricks / Agentic  │ Multi-Agent Orchestrator        │ `backend/agents/` ReAct agents    │
│ 7. AI & BI (Agentic BI)   │ Agentic BI Query Engine         │ Natural language to SQL engine    │
│ 8. Data & AI Web Apps     │ Modern Bun 1.2 + React 19 UI    │ React 19 + TypeScript + Wasm      │
└───────────────────────────┴─────────────────────────────────┴───────────────────────────────────┘
```

---

## 🌊 Delta Lake Medallion Architecture

The data pipeline organizes raw inputs into cleaned and enriched Delta Lake tables across three medallion tiers:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       DELTA LAKE MEDALLION PIPELINE ARCHITECTURE                        │
├─────────────────────────┬─────────────────────────────┬─────────────────────────────────┤
│    BRONZE LAYER         │        SILVER LAYER         │           GOLD LAYER            │
│  (Raw Ingestion)        │  (Cleaned & Enriched)       │    (Aggregated Features)        │
├─────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ • Ingest raw JSON/Kafka │ • Deduplicate clickstreams  │ • User Interaction Sequences    │
│ • TMDB & ALS Raw Feeds  │ • Standardize ratings       │ • Item Similarity Matrices      │
│ • Variant Schema Store  │ • Join TMDB Metadata        │ • ALS 16d-64d Vectors & FAISS   │
└─────────────────────────┴─────────────────────────────┴─────────────────────────────────┘
```

1. **Bronze Layer**: Raw clickstreams and JSON event logs ingested asynchronously via `etl/lakeflow_pipeline.py`.
2. **Silver Layer**: Cleaned, deduplicated interaction logs with standardized timestamps and user-item keys.
3. **Gold Layer**: Aggregated user interaction sequences, item co-occurrence matrices, and ALS vector embeddings formatted for FAISS and PyTorch models.

---

## 📄 Spark Declarative Pipeline (SDP) Specification

The system uses YAML declarations (`config/spark_declarative_pipeline.yaml`) executed by `etl/spark_declarative_pipeline.py`:

```yaml
pipeline_id: "apex_unified_data_intelligence_v1"
target_schema: "apex_recommendations"
tables:
  - table_name: "bronze_user_events"
    layer: "bronze"
    format: "delta"
    source: "data/raw/user_events.json"

  - table_name: "silver_interactions"
    layer: "silver"
    format: "delta"
    depends_on: ["bronze_user_events"]

  - table_name: "gold_user_embeddings"
    layer: "gold"
    format: "delta"
    depends_on: ["silver_interactions"]
```

---

## 🌟 Key Architectural Advantages

1. **Zero Vendor Lock-In**: Built 100% on open formats (Delta Lake, Apache Spark, OpenLineage) so you can run standalone locally or seamlessly deploy to Databricks when required.
2. **Legal & Open Compliance**: Uses standard open-source Python, PySpark, Rust, and SQL components without proprietary lock-in.
3. **Agentic Business Intelligence**: Natural language to SQL translation, executing metric aggregations over Delta Lake tables.

---

## 🧪 Verification & Test Commands

```bash
# Execute PySpark Declarative Pipeline
python etl/pyspark_etl.py --declarative

# Run pipeline unit tests
python -m pytest tests/test_spark_declarative_pipeline.py -v
```
