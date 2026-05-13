# Data Engineering System Design and Tradeoffs

This document explains the "why" behind the main data engineering choices in Nova. The goal is not to use every tool. The goal is to choose the smallest architecture that satisfies the product requirement, then know when to evolve it.

## Decision Principle

Every component must answer three questions:

1. What requirement does it solve?
2. What simpler option did we reject?
3. What operational cost does it add?

If a component cannot answer those questions, it does not belong in the core architecture.

## Current Core vs Conditional Extensions

| Area | Current Core | Add Only When |
| --- | --- | --- |
| Catalog ingestion | Batch ETL | Source updates become near-real-time |
| Processing | PySpark-first batch ETL with local Pandas fallback | Data volume exceeds free/local compute or needs managed distributed jobs |
| Storage | Parquet serving artifacts | Delta/Iceberg when ACID, MERGE, time travel, or concurrent writes matter |
| Orchestration | CLI/CI locally, Airflow as scheduled pipeline option | Multiple dependent jobs, retries, backfills, SLAs |
| Streaming | Not required for static catalog refresh | User behavior events need replay, backpressure, and consumer groups |
| Data model | Current movie metadata for serving | SCD2/star schema when analytics need history |
| Vector search | FAISS | Distributed vector DB when index size, filtering, tenancy, or uptime exceeds single-node limits |

## Target System

Nova can be explained in four planes. Not every plane needs every enterprise component in the local MVP:

- Data plane: raw catalog data, curated metadata, optional user events and search logs.
- Processing plane: PySpark for canonical batch/lakehouse runs, with Pandas only for small local fallback and deterministic tests.
- Serving plane: SBERT embeddings, FAISS index, FastAPI recommendation/search endpoints, React/Streamlit UIs.
- Operations plane: CI tests, Docker, health checks, optional Airflow orchestration, data quality reports, artifact manifests, monitoring.

## Batch vs Streaming

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| Batch ETL | Daily catalog refresh, embedding rebuild, backfills | Simple, deterministic, easy to rerun | Not real-time |
| Kafka streaming | User views, ratings, clicks, search events | Captures behavior as it happens | Requires brokers, consumer offsets, replay handling |
| Hybrid | Catalog in batch, user behavior in stream | Common real-world design | More moving parts |

Recommended Nova design:

- Catalog metadata is batch because TMDB/Kaggle-style catalog updates do not need millisecond freshness.
- User events should be streaming only if the product uses views, ratings, clicks, or searches for personalization or analytics.
- Embeddings can be rebuilt batch-first; later, incremental embedding updates can be added for changed movies only.

## PySpark vs Pandas

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| PySpark | Canonical batch pipeline, distributed joins, partitioned lakehouse writes, SCD updates | Scales horizontally and matches production DE work | More setup, slower local iteration |
| Pandas | Small local fallback, unit tests, artifact inspection | Fast developer loop, simple debugging | Single-machine memory limit; not the main DE story |
| Databricks Spark | Managed production lakehouse | Jobs, clusters, Delta, governance | Cloud cost and platform dependency |

Recommended Nova design:

- Treat `etl/pyspark_etl.py` and `etl/delta_lakehouse.py` as the canonical DE implementation.
- Keep Pandas helpers for deterministic local fallback, tests, and lightweight artifact inspection.
- Ensure the Spark and fallback paths produce compatible serving artifacts.

## Parquet vs Delta Lake

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| Parquet | Portable columnar storage, model metadata, small local demo | Simple, efficient, universally supported | No ACID transactions or time travel by itself |
| Delta Lake | Bronze/Silver/Gold tables, backfills, SCD, incremental merges | ACID, schema evolution, MERGE, time travel | Requires Delta runtime/JARs |
| Iceberg/Hudi | Large open lakehouse alternatives | Strong table formats | More catalog setup |

Recommended Nova design:

- Use Parquet for local serving artifacts and CI fixtures.
- Use Delta for cloud/lakehouse tables only where MERGE, time travel, concurrent writes, and schema evolution matter.
- Explain that Parquet is a file format, while Delta is a transaction layer on top of Parquet.

## Bronze, Silver, Gold

| Layer | Purpose | Example |
| --- | --- | --- |
| Bronze | Raw, minimally changed ingestion | Original TMDB rows partitioned by run date |
| Silver | Cleaned and validated data | Deduped movies, normalized fields, quality checks |
| Gold | Business-ready serving features | Movie text for embeddings, popularity features, SCD dimensions |
| Serving | Optimized online artifacts | FAISS index, SBERT vectors, compact Parquet metadata |

Product/system-design explanation:

> Bronze preserves source truth, Silver makes the data trustworthy, Gold makes it useful, and Serving makes it fast.

## SCD Type 1 vs Type 2

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| SCD Type 1 | Correcting attributes where history does not matter | Simple overwrite | Loses history |
| SCD Type 2 | Tracking changes to catalog attributes over time | Preserves history for analytics and audits | More storage and query complexity |

Nova does not need SCD Type 2 for the online recommendation API. It is useful for the analytical warehouse path if stakeholders need to answer historical questions such as "what genre/director/cast metadata was current when this recommendation was served?"

Implementation in repo:

- `etl/pyspark_etl.py` contains Spark SCD Type 2 helpers and an upsert path for `gold.dim_movie_scd`.
- `etl/delta_lakehouse.py` defines the Delta table contracts, schemas, time-travel helpers, and restore/history utilities.
- `etl/scd.py` remains a small deterministic fallback used by local tests and manifest-backed inspection utilities.
- `tests/test_pyspark_scd.py` verifies Spark SCD change tracking and the Parquet fallback path.
- `sql/movie_recommendation_star_schema.sql` contains the analytical dimension/fact model.

## SQL vs NoSQL

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| SQL warehouse | Analytics, BI, data quality, reporting | Joins, window functions, governance | Less suited for very low-latency key-value lookups |
| Object storage | Lakehouse raw/curated data | Cheap, scalable, decoupled compute/storage | Requires table format/catalog for governance |
| NoSQL/key-value | API cache, session state, hot recommendations | Very low-latency lookups | Harder ad hoc analytics |
| Vector index | Semantic nearest-neighbor retrieval | Fast similarity search | Not a replacement for warehouse truth |

Recommended Nova design:

- Use SQL/lakehouse for source of truth and analytics.
- Use FAISS/vector DB for semantic retrieval.
- Use Redis or DynamoDB only for cache/hot lookup if needed.
- Do not put analytical truth directly in NoSQL unless query patterns demand it.

## FAISS vs Vector Database

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| FAISS | Local demo, batch-built vector index, low cost | Fast, mature, offline, simple deployment | Single-node unless custom sharding |
| Pinecone/Milvus/Weaviate/OpenSearch | Distributed vector serving | Scaling, metadata filters, managed operations | Cost, vendor/platform dependency |

Recommended Nova design:

- Use FAISS for portfolio MVP and cost-efficient deployment.
- Explain that at 10M+ items or multi-tenant workloads, you would evaluate a distributed vector DB.

## Airflow vs Cron vs Step Functions

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| Cron | Simple scheduled command | Minimal setup | Weak dependency management and observability |
| Airflow | ETL DAGs, dependencies, retries, backfills | Industry standard orchestration | Operational overhead |
| Step Functions | Serverless AWS workflows | Managed, visual, integrates with Lambda/Glue/EMR | AWS-specific |
| Databricks Jobs | Lakehouse-native jobs | Strong Spark integration | Databricks-specific |

Recommended Nova design:

- Use a CLI/manual run for local development.
- Use Airflow when refresh has multiple dependent steps, retries, backfills, and operational ownership.
- Explain MWAA or Databricks Jobs for production depending on platform.

## Kafka vs Direct API Writes

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| Direct API write | Simple logging to database/object storage | Easy for MVP | Tight coupling and weaker replay |
| Kafka | User event streams and replayable ingestion | Durable event log, consumer groups, backpressure | More infra and operational complexity |

Recommended Nova design:

- Use Kafka for user behavior events, not for static catalog ingestion.
- Do not add Kafka to the core recommender unless there is a product requirement for event replay, near-real-time engagement features, or downstream consumers.
- If Kafka is introduced, store events in Bronze first, then aggregate engagement features in Silver/Gold.

## Databricks vs Self-Managed Spark

| Choice | Use It For | Why | Tradeoff |
| --- | --- | --- | --- |
| Local Spark | Learning, tests, small batch | Free and controllable | Not production-like |
| Self-managed Spark on Kubernetes | Custom infra, cost control | Flexible | Requires ops maturity |
| Databricks | Production lakehouse | Managed Spark, Delta, Jobs, MLflow, Unity Catalog | Cost and vendor dependency |
| EMR/Glue | AWS-native Spark | Integrates with S3/IAM | More AWS-specific tuning |

Product/system-design explanation:

> I would use local Spark/Docker for reproducible development, Databricks or EMR Serverless for production-scale ETL, and keep storage in open formats like Parquet/Delta so compute remains replaceable.

## Data Quality Strategy

Minimum checks:

- Required columns exist.
- Primary keys are not null.
- Movie IDs are unique per snapshot.
- Title and overview completeness above threshold.
- Vote averages within 0 to 10.
- Record counts do not drop unexpectedly.
- Embedding count equals curated movie count.
- FAISS index count equals embedding count.

Artifacts to add:

- `data/quality/run_date=.../metrics.json`
- `data/manifests/run_date=.../pipeline_manifest.json`
- `data/lineage/run_date=.../lineage.json`

## Idempotency Strategy

Use:

- Run date as a partition key.
- Atomic writes to temp paths followed by rename/commit.
- Deterministic output paths for each run.
- Pipeline manifests with source file checksums.
- SCD hashes to detect actual attribute changes.
- Backfill mode that can rerun a date without duplicating facts.

## Serving Tradeoff

Online recommendation calls should not compute embeddings from scratch for every movie. Nova precomputes:

- Movie metadata.
- Dense embeddings.
- FAISS index.
- Re-ranking attributes.

Only query-time text search/chat should compute a query embedding or call an LLM.

## What To Build Next, Based On Need

1. Needed now: persist data quality metrics and pipeline manifests for every ETL run.
2. Needed now: add deterministic sample data so reviewers can run the project quickly.
3. Needed now: add API latency/load smoke test because serving speed is part of the product claim.
4. Needed if analytics is a goal: write recommendation impression/search logs into fact tables.
5. Needed if catalog history matters at production scale: run the Spark/Delta SCD path in managed CI or Databricks/EMR with real Delta runtime.
6. Needed if user behavior matters: add Kafka event ingestion and Gold engagement features.
7. Needed if cloud deployment matters: add a Databricks or AWS job config, not both unless both are actually targeted.
