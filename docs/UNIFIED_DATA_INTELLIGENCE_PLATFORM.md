# Unified Data & AI Intelligence Platform

The **APEX Movie Recommendation System** implements an open, vendor-neutral **Unified Data & AI Intelligence Platform** matching Databricks platform capabilities 1-to-1 using open standards (Apache Spark, Delta Lake, Serverless Postgres, OpenLineage, and Multi-Agent AI).

---

## 🏛️ Databricks 1-to-1 Mapping Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 UNIFIED DATA & AI PLATFORM MAPPING                              │
├───────────────────────────┬─────────────────────────────────┬───────────────────────────────────┤
│ DATABRICKS COMPONENT      │ APEX OPEN PLATFORM ENGINE       │ ENTERPRISE SPECIFICATION          │
├───────────────────────────┼─────────────────────────────────┼───────────────────────────────────┤
│ 1. DBSQL / Warehousing    │ Apache Spark 4.1 SQL (AQE)      │ Vectorized Parquet & AQE 2.0      │
│ 2. Lakebase (Serverless)  │ Neon Serverless Postgres        │ Dynamic branching & scaling       │
│ 3. Lakeflow Ingestion     │ Lakeflow Ingestion Engine       │ `etl/lakeflow_pipeline.py`        │
│ 4. Open Table Format      │ Delta Lake 4.3 (`delta-spark`)  │ ACID logs & Z-Ordering            │
│ 5. Data Catalog           │ Unity Catalog Metastore         │ `catalog.schema.table` 3-level    │
│ 6. AgentBricks / Agentic  │ Multi-Agent Orchestrator        │ `backend/agents/` ReAct agents    │
│ 7. AI & BI (Agentic BI)   │ Agentic BI Query Engine         │ `backend/intelligence/agentic_bi` │
│ 8. Data & AI Web Apps     │ Modern Glassmorphic React 19 UI │ React 19 + Vite 6 + Wasm          │
└───────────────────────────┴─────────────────────────────────┴───────────────────────────────────┘
```

---

## 🌟 Key Architecture Advantages

1. **Zero Vendor Lock-In**: Built 100% on open formats (Delta Lake, Apache Spark, OpenLineage, PyO3 Rust) so you can deploy standalone locally or seamlessly migrate to Databricks when needed.
2. **Legal & Open Compliance**: Uses standard open-source Python, PySpark, Rust, and SQL components without proprietary lock-in.
3. **Agentic Business Intelligence**: Natural language to SQL translation, executing metric aggregations over Delta Lake and Neon tables.

---

## 🧪 Verification

```bash
$ python -m pytest tests/test_unified_data_intelligence.py
============================== 3 passed in 0.25s ==============================
```
