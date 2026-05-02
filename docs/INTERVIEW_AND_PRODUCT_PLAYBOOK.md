# Nova Interview and Product Playbook

This project should be presented as an AI-powered data product, not only as a movie recommender.

The strongest positioning is:

> Nova is a batch-first semantic discovery platform that ingests raw catalog data, validates and curates it, generates embeddings, indexes them with FAISS, and serves low-latency recommendations through FastAPI and Streamlit. Lakehouse, orchestration, streaming, and warehouse patterns are introduced only when the requirement justifies them.

## What This Project Proves

For data engineering interviews, Nova should prove that you can:

- Build an end-to-end data pipeline from raw ingestion to serving.
- Design batch ETL with data quality gates and idempotent reruns.
- Explain Spark transformations, partitioning, file formats, and pipeline orchestration.
- Explain when lakehouse concepts such as Bronze, Silver, Gold, and Delta Lake are worth the operational cost.
- Serve ML/AI artifacts as a product through APIs.
- Add observability, tests, CI, Docker, and deployment discipline.
- Discuss tradeoffs between local files, object storage, warehouses, vector indexes, and distributed systems.

For startup/product interviews, Nova should prove that you can:

- Turn a dataset into a working AI product.
- Build a useful demo with search, recommendations, chat, and monitoring.
- Explain monetizable use cases beyond movies: e-commerce search, content discovery, enterprise knowledge search.
- Make pragmatic architecture choices instead of overengineering everything.

## Architecture Story

Use this concise explanation in interviews:

1. Raw movie data is ingested as a batch source because catalog freshness does not require streaming.
2. The ETL validates schema, filters low-quality records, deduplicates IDs, and curates metadata.
3. The local path writes Parquet serving artifacts; the lakehouse path can map the same logic to Bronze/Silver/Gold when Delta is available.
4. Gold/serving features include searchable text, ranking attributes, and curated movie metadata.
5. SBERT converts movie text into dense embeddings.
6. FAISS indexes normalized vectors for fast nearest-neighbor retrieval.
7. FastAPI serves search, recommendation, health, and chat endpoints.
8. Streamlit provides the product UI and monitoring dashboard.
9. CI runs compile, lint, and tests; optional infra tests are explicitly gated so local checks stay deterministic.

## Interview Answer: Why This Is Data Engineering

This is not just a machine learning demo. The core data engineering work is:

- Ingestion: raw source data is fetched and normalized.
- Data quality: null checks, vote thresholds, schema validation, and curated output.
- Storage design: Parquet for the current serving artifacts, with Delta Lake reserved for production lakehouse requirements.
- Processing: Pandas for local reliability and PySpark for scale.
- Orchestration: Airflow when scheduled refresh, retries, backfills, and dependency ownership are required.
- Serving: precomputed embeddings and FAISS index avoid expensive online computation.
- Reliability: tests, health checks, retries, Docker, and safe artifact loading.
- Observability: Streamlit monitoring page plus backend health endpoint.

## Strong Resume Bullets

Use truthful bullets that match the repository:

- Built Nova, an AI-powered semantic discovery platform that processes TMDB catalog data through Pandas/PySpark ETL, validates data quality, generates SBERT embeddings, indexes vectors with FAISS, and serves low-latency recommendations through FastAPI and Streamlit.
- Designed a batch-first data pipeline with Parquet serving artifacts and a documented lakehouse extension path for Bronze/Silver/Gold and Delta Lake when ACID MERGE, time travel, and backfills are required.
- Implemented semantic search and recommendation APIs with FAISS vector retrieval, metadata re-ranking, and MMR-style diversification to improve recommendation relevance and reduce duplicate-style results.
- Added CI-quality checks with pytest, compile validation, linting, optional Airflow/Kafka/Spark integration tests, and safer artifact loading to prevent import-time network calls or tracked data rewrites.

Shorter version for a one-page resume:

- Built an AI semantic recommendation platform using Pandas/PySpark ETL, Parquet serving artifacts, SBERT embeddings, FAISS vector search, FastAPI, Streamlit, Docker, and pytest.
- Designed data quality checks, idempotent refresh flow, model artifact generation, API serving, and documented tradeoffs for Delta, Kafka, Airflow, Databricks, SQL/NoSQL, and SCD Type 2.

## Claims To Avoid

Avoid claims that are easy for an interviewer to challenge:

- Do not say it is production-scale unless deployed with production infra and load testing.
- Do not say 100 percent uptime unless you have SLOs, monitoring, replicas, failover, and incident history.
- Do not say it processes billions of records unless the repo demonstrates that scale.
- Do not say cosine similarity if the current implementation uses SBERT and FAISS.
- Do not say Kafka/Spark streaming is fully productionized until the live infra path is tested end to end.

Better wording:

- "Production-style" instead of "production-grade."
- "Designed for cloud/lakehouse deployment" instead of "enterprise deployed."
- "Source dataset supports large-scale ingestion; curated serving layer is optimized for low latency" instead of vague scale claims.

## Questions You Should Be Ready For

### ETL and Spark

- Why use Spark instead of Pandas?
- How would you partition the data?
- What happens if the pipeline fails halfway?
- How do you make reruns idempotent?
- How do you handle schema drift?
- How do you validate data quality?
- What would you store in Bronze, Silver, and Gold?

### Serving and Search

- Why precompute embeddings?
- Why FAISS instead of Postgres full-text search?
- What is approximate nearest-neighbor search?
- How do you keep vectors and metadata in sync?
- How would you update the index incrementally?
- How do you evaluate recommendation quality?

### System Design

- How would you scale this to 100 million items?
- How would you deploy it on AWS?
- How would you monitor freshness and failures?
- What are the bottlenecks?
- How would you recover from corrupted model artifacts?
- How would you handle secrets?

### Product

- Who is the customer?
- What problem does semantic discovery solve?
- How would you price it?
- How would you adapt it from movies to e-commerce or enterprise documents?
- What metrics prove the product is working?

## Best Demo Script

Use this sequence in a live demo:

1. Start with the product: "Search for 'time travel heist' and show semantic results."
2. Show the API docs: FastAPI endpoints and health endpoint.
3. Show the pipeline: ETL modules, config, data quality checks, artifacts.
4. Show the architecture docs: Bronze/Silver/Gold and serving layer.
5. Show tests and CI: pytest, lint, optional infra tests.
6. End with scale discussion: what changes for 10M, 100M, and distributed vector search.

## Productization Roadmap

### Tier 1: Interview Ready

- Keep tests, lint, and compile passing.
- Make README and architecture docs consistent with actual implementation.
- Add a clear architecture diagram.
- Add one deterministic sample dataset for local demo.
- Add `.env.example` with all required variables.
- Document optional test flags: `RUN_LLM_TESTS`, `RUN_INFRA_INTEGRATION_TESTS`, `FORCE_MODEL_REFRESH`.

### Tier 2: Strong Data Engineering Signal

- Add explicit data contracts for raw, silver, and gold tables.
- Persist data quality metrics as JSON/Parquet artifacts.
- Add idempotency markers for pipeline runs.
- Add partitioned outputs by `run_date`.
- Add a small lineage manifest for every pipeline run.
- Add Airflow DAG tests that run in CI with a pinned Airflow dependency.
- Use the SCD Type 2 implementation in `etl/scd.py` to explain historical dimension handling.
- Use `sql/movie_recommendation_star_schema.sql` to discuss dimensions, facts, ranking analytics, and SQL tradeoffs.

### Tier 3: Product Signal

- Add query analytics: searches, clicks, recommendations served.
- Add offline evaluation: precision@k, diversity, coverage, latency.
- Add an admin dashboard for data freshness and index status.
- Add Docker Compose quickstart for API + UI + optional Airflow.
- Add load testing for API latency.

### Tier 4: FAANG-Level Discussion

- Explain how to move from FAISS local index to distributed vector search.
- Explain eventual consistency between data lake, embeddings, index, and API.
- Explain backfills, incremental updates, and data retention.
- Explain cost tradeoffs across S3, Snowflake, Databricks, Redis, and vector databases.
- Explain monitoring through SLIs/SLOs: freshness, success rate, latency, quality.

## Target Architecture For Cloud

AWS version:

- S3 for Bronze/Silver/Gold.
- Glue or EMR Serverless for Spark ETL.
- MWAA for orchestration.
- ECR/ECS or Lambda container for FastAPI.
- CloudWatch for logs and alerts.
- Secrets Manager for API keys.
- OpenSearch/Milvus/Pinecone if vectors outgrow local FAISS.

Databricks version:

- Delta Lake tables for Bronze/Silver/Gold.
- Databricks Jobs for orchestration.
- MLflow for embedding/index artifact tracking.
- Model Serving or containerized FastAPI for API.
- Unity Catalog for governance.

Startup MVP version:

- Docker Compose.
- Local Parquet.
- FAISS.
- FastAPI.
- Streamlit.
- GitHub Actions.

## Deep-Dive References

- `docs/DATA_ENGINEERING_SYSTEM_DESIGN.md`: Spark, Delta, Kafka, Airflow, Databricks, SQL/NoSQL, SCD, and serving tradeoffs.
- `etl/scd.py`: deterministic SCD Type 2 helper for movie dimension history.
- `sql/movie_recommendation_star_schema.sql`: SCD2 dimension, recommendation/search/user-event facts, and analytical SQL examples.

## One-Minute Pitch

Nova is an AI discovery platform I built to demonstrate end-to-end data engineering and AI serving. It ingests raw movie catalog data, validates and curates it into lakehouse-style layers, generates SBERT embeddings, builds a FAISS vector index, and serves semantic recommendations through FastAPI and Streamlit. The important part is not movies; the same architecture applies to e-commerce products, articles, support tickets, or enterprise knowledge bases. I focused on idempotent ETL, data quality, artifact management, low-latency serving, and CI-tested reliability so it behaves like a real data product rather than a notebook demo.
