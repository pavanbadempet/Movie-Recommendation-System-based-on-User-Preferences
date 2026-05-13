---
title: "Nova: Recommendation Intelligence Platform"
emoji: "🎬"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Nova: Recommendation Intelligence Platform

Nova is a B2B recommendation and semantic discovery platform for content businesses. It is not a streaming-app clone. It is the infrastructure layer a media company, OTT startup, education platform, publisher, or catalog marketplace could use to make its own catalog searchable, recommendable, measurable, and AI-ready.

The current demo vertical is movies using TMDB/Kaggle data, but the architecture is intentionally tenant-aware: customers own catalogs, catalogs contain content items, content items become semantic features, and user behavior becomes ranking intelligence.

## What Nova Does

- Ingests content catalogs through a PySpark batch pipeline.
- Stores raw, validated, and curated data in a Delta Lake medallion model.
- Preserves catalog history with SCD Type 2 dimensions and Delta time travel.
- Quarantines bad records instead of silently dropping them.
- Keeps long-tail and obscure catalog items searchable; weak metadata is scored, not deleted.
- Builds semantic embeddings and FAISS search artifacts for low-latency serving.
- Serves hybrid AI search with sparse recall, optional dense query recall, ranking features, optional cross-encoder reranking, and MMR diversity.
- Personalizes recommendations from user behavior events using implicit feedback.
- Trains an optional learned ranking model from behavior events and catalog quality, then loads it during serving.
- Creates incremental embedding jobs only for new or changed content.
- Captures product events such as views, clicks, searches, ratings, and recommendation impressions.
- Provides FastAPI endpoints for search, recommendations, events, and behavior features.
- Protects product APIs with optional tenant API keys while keeping the public demo free.
- Measures recommendation artifact quality with label-free coverage checks plus human-labeled search, semantic, and item-to-item benchmark gates.
- Includes a Streamlit Nova Console for API context, usage, AI quality, event testing, and integration snippets.
- Onboards customer catalogs through CSV preview, column mapping, quality profiling, and raw upload manifests.
- Publishes model/artifact outputs through Hugging Face for lightweight serving on Render and Streamlit.

## Product Architecture

```mermaid
flowchart LR
    A["Customer catalog batch source"] --> B["Bronze Delta: raw content snapshots"]
    B --> C["Silver Delta: validated content + quarantine"]
    C --> D["Gold Delta: content features + SCD history"]
    D --> E["Embedding jobs for changed content"]
    E --> F["FAISS + model artifacts"]
    F --> G["FastAPI recommendation service"]
    G --> H["Customer app or Streamlit demo"]
    H --> I["Behavior events API"]
    I --> J["Kafka / Spark Structured Streaming"]
    J --> K["Gold Delta: content event facts"]
    K --> L["Behavior aggregates"]
    L --> G
```

## Data Engineering Model

Nova uses two related data models:

- **Movie vertical model:** keeps the existing TMDB-powered demo simple and functional.
- **Platform content model:** adds `tenant_id`, `catalog_id`, `content_id`, `source_system`, and `source_content_id` so the same system can support many customer catalogs.

Core Delta tables include:

- `bronze.movies`
- `silver.movies`
- `silver.movies_quarantine`
- `gold.movies_features`
- `gold.dim_movie_scd`
- `gold.movie_embedding_jobs`
- `gold.pipeline_run`
- `gold.data_quality_observation`
- `silver.content_items`
- `gold.content_features`
- `gold.fact_content_event`
- `gold.content_behavior_daily`
- `gold.dim_content_scd`

## Why These Choices

- **Batch for catalog refresh:** movie/content metadata changes daily or hourly, not every millisecond.
- **Streaming for product behavior:** views, clicks, searches, ratings, and impressions are continuous and useful for fresh ranking.
- **Delta over raw Parquet for lakehouse tables:** ACID commits, MERGE, Change Data Feed, time travel, restore, and auditable history matter for a customer-facing data product.
- **Parquet/NumPy/FAISS for serving artifacts:** small, portable, cheap to host, and fast enough for the current demo scale.
- **PySpark over Pandas for the canonical ETL path:** distributed transformations, schema contracts, and batch reliability belong in Spark. Pandas remains useful for lightweight local fallback and tests.
- **Coverage-first catalog filtering:** missing IDs/titles are invalid, but low votes, low popularity, or short overviews are long-tail signals. Nova keeps those items and assigns metadata completeness, quality buckets, and recommendability flags.
- **GitHub Actions/Kaggle today, Airflow later:** scheduled notebook execution is enough for the current hosted demo; Airflow becomes valuable when retries, dependency graphs, SLAs, and multiple customers matter.
- **Kafka only for events:** adding Kafka to a static daily catalog source would be theater. Product events are the correct streaming boundary.

## Repository Map

- `backend/` - FastAPI serving, recommendation endpoints, event capture.
- `backend/auth.py` - optional API-key tenant context for product/customer mode.
- `backend/evaluation.py` - free-tier-safe recommendation quality metrics.
- `backend/search_benchmark.py`, `backend/semantic_benchmark.py`, and `backend/recommendation_benchmark.py` - human-labeled serving quality gates.
- `backend/catalogs.py` - customer CSV preview, quality profiling, and local upload manifests.
- `backend/recommender.py` - hybrid AI search, dense item recommendations, reranking, behavior-aware personalization.
- `backend/ranker.py` and `backend/ranker_training.py` - learned ranker artifact loading, training, and offline metrics.
- `etl/pyspark_etl.py` - canonical PySpark batch pipeline.
- `etl/delta_lakehouse.py` - Delta schemas, table contracts, time travel, CDF, audit helpers.
- `etl/streaming_events.py` - Kafka to Delta Structured Streaming ingestion for behavior events.
- `data/evaluation/` - benchmark labels for search relevance, semantic similarity, and recommendation product quality.
- `notebooks/kaggle_etl_pipeline.py` - hosted Kaggle execution path for daily artifact refresh.
- `airflow/dags/` - orchestration examples for mature deployments.
- `docs/PRODUCT_DATA_PLATFORM_BLUEPRINT.md` - product and data platform architecture.

## Quick Start

Install serving dependencies:

```bash
python manage.py setup
```

Run the API and Streamlit app locally:

```bash
python manage.py run
```

Optional API-key mode:

```bash
NOVA_API_KEYS=secret-key:demo-media-co:tmdb-movies:free
NOVA_API_KEY=secret-key
```

Optional durable event store:

```bash
NOVA_EVENT_STORE=postgres
NOVA_EVENT_DATABASE_URL=postgresql://user:password@host:5432/dbname
NOVA_EVENT_TABLE=nova_content_events
```

Use `NOVA_EVENT_STORE=dual` during demos when you want local JSONL plus durable Postgres writes. Without a database URL, Nova keeps using local JSONL so the free demo remains easy to run.

Run the canonical Spark/Delta ETL path in an ETL-capable environment:

```bash
pip install -r requirements-etl.txt
python etl/pyspark_etl.py --sink delta --tenant-id demo-media-co --catalog-id tmdb-movies
```

Run Kafka event streaming into Delta:

```bash
python etl/streaming_events.py --bootstrap-servers localhost:9092 --topic nova.content_events
```

For Spark Kafka integration, submit with the matching `spark-sql-kafka-0-10` package for your Spark/Scala runtime.

Train the optional learned ranker:

```bash
python scripts/train_ranker.py --events data/events/movie_events.jsonl --output models/nova_ranker.joblib
```

Train the ranker from the latest Hugging Face movie artifact and upload the refreshed ranker:

```bash
python scripts/train_ranker.py --download-movies-from-hf --upload-to-hf --hf-repo pavanbadempet/movie-recs-models
```

Promotion-gated training:

```bash
python scripts/train_ranker.py --promotion-gate --output models/nova_ranker.candidate.joblib --production-output models/nova_ranker.joblib
```

The daily GitHub Actions refresh runs the Kaggle artifact build first, then trains a candidate ranker, compares it against baseline/current metrics, and uploads `nova_ranker.joblib` only when the promotion gate passes. Set `NOVA_EVENTS_URL` as a GitHub secret when a persistent JSONL event export is available; otherwise the scheduled ranker uses catalog-bootstrap labels until real feedback exists.

For durable behavior feedback, set `NOVA_EVENT_DATABASE_URL` as a GitHub secret. The daily ranker refresh will read the Postgres event store directly and switch from catalog-bootstrap labels to implicit-feedback labels once enough content has real events.

Experiment assignment and outcome metrics:

```bash
NOVA_EXPERIMENT_VARIANTS=control:50,personalized_v2:50
```

Use `/v1/experiments/assignment` for deterministic variant assignment and `/v1/experiments/metrics` for impression, click, and rating outcomes from behavior events.

## Deployment Artifact Reload

The scheduled artifact refresh can update the deployed backend without waiting for a cold start.

Set the same secret value in both places:

- Render environment variable: `NOVA_ADMIN_TOKEN`
- GitHub Actions repository secret: `NOVA_ADMIN_TOKEN`

If the backend URL changes, also set `NOVA_RENDER_API_URL` as a GitHub Actions secret. After the Kaggle/Hugging Face artifact workflow succeeds, GitHub Actions calls:

```bash
POST /v1/artifacts/reload?force_download=true&load=true
X-Nova-Admin-Token: <NOVA_ADMIN_TOKEN>
```

This refreshes the pipeline manifest, downloads changed serving artifacts, validates artifact health, and swaps the in-memory recommender instance.

## Commercial Direction

Nova can become a product for smaller media/catalog companies before it can serve huge enterprises. The believable first customers are teams that have content but do not have recommendation infrastructure:

- regional OTT platforms
- education video libraries
- creator/course marketplaces
- digital publishers
- internal media archives
- niche streaming catalogs

The product promise is simple: bring your catalog and behavior events; Nova gives you semantic search, recommendations, quality-controlled data pipelines, catalog history, behavior analytics, and serving APIs.

## Engineering References

- [Delta Lake Change Data Feed](https://docs.delta.io/delta-change-data-feed.html)
- [Apache Spark Structured Streaming](https://spark.apache.org/docs/3.5.7/structured-streaming-programming-guide.html)
- [Spark Structured Streaming Kafka Integration](https://spark.apache.org/docs/_site/streaming/structured-streaming-kafka-integration.html)
- [Apache Kafka Design](https://kafka.apache.org/41/design/design/)
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
