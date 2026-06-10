# Data Contracts

This document defines the current data contracts for the AI + Data Engineering
path in this repository. It is intentionally grounded in the code that exists
today, primarily:

- `etl/pandas_etl.py`
- `etl/delta_lakehouse.py`
- `etl/pyspark_etl.py`
- `backend/serving/artifact_validator.py`

## Purpose

These contracts establish a shared definition for:

- required datasets
- schema ownership
- keys and partitions
- quality expectations
- downstream consumers

The goal is to make batch processing, model training, and serving artifacts
easier to validate and safer to evolve.

## Current Sources Of Truth

### Local fallback ETL

The Pandas pipeline in `etl/pandas_etl.py` currently defines the most explicit
local input schema via `MOVIE_SCHEMA` and already performs:

- schema validation
- duplicate ID tracking
- vote range checks
- metadata completeness scoring
- content quality scoring

### Canonical DE layer

The Spark and lakehouse contracts in `etl/delta_lakehouse.py` are the canonical
batch definitions for:

- `bronze_movies`
- `silver_movies`
- `gold_movies_features`
- `dim_movie_scd`
- `fact_movie_event`
- `movie_embedding_jobs`
- `pipeline_runs`
- `quarantine_movies`
- tenant catalog tables

## Contract Template

Each dataset contract should define:

- owner
- layer
- schema version
- primary key
- partition key
- required columns
- nullable columns
- freshness expectation
- quality gates
- downstream consumers

## Dataset Contracts

### Raw Movie Input

- Owner: Data ingestion pipeline
- Layer: source
- Schema version: `1`
- Current implementation: `MOVIE_SCHEMA` in `etl/pandas_etl.py`
- Primary key: `id`
- Partition key: none at source level
- Freshness expectation: loaded per batch run from the configured raw dataset
- Required columns:
  - `id`
  - `title`
- Important optional columns:
  - `overview`
  - `genres`
  - `vote_average`
  - `vote_count`
  - `popularity`
  - `release_date`
  - `poster_path`
  - `keywords`
  - `production_companies`
  - `cast`
  - `director`
- Quality gates:
  - `id` must be coercible to integer
  - `title` must be present
  - `vote_average`, when present, must be in `[0, 10]`
  - duplicate `id` values are tracked and should be removed before curated outputs
- Downstream consumers:
  - Pandas ETL transforms
  - Spark bronze ingest
  - metadata completeness and quality scoring

### Bronze Movies

- Owner: batch ETL
- Layer: bronze
- Schema version: `1`
- Current implementation: `BRONZE_MOVIES_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `id`
- Partition key:
  - `run_date`
- Required columns:
  - `id`
  - `run_date`
  - `run_id`
  - `ingestion_ts`
- Quality gates:
  - ingestion metadata must always be populated
  - each write must be attributable to a single `run_id`
  - source rows that cannot be parsed cleanly should not silently disappear
- Downstream consumers:
  - silver transforms
  - quarantine logic
  - pipeline run accounting

### Silver Movies

- Owner: curated batch ETL
- Layer: silver
- Schema version: `1`
- Current implementation: `SILVER_MOVIES_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `id`
- Partition key:
  - `run_date`
- Required columns:
  - `id`
  - `title`
  - `overview`
  - `tags`
  - `run_date`
  - `run_id`
  - `ingestion_ts`
- Important derived columns:
  - `release_year`
  - `metadata_completeness`
  - `content_quality_score`
  - `quality_bucket`
  - `searchable`
  - `recommendable`
  - `is_adult_content`
  - `public_demo_eligible`
- Quality gates:
  - one deterministic current row per movie ID
  - text fields required for downstream feature creation must be populated
  - quality columns must be generated consistently from the curated row
- Downstream consumers:
  - gold feature generation
  - embedding job generation
  - public catalog and demo eligibility logic

### Gold Movie Features

- Owner: feature generation pipeline
- Layer: gold
- Schema version: `1`
- Current implementation: `GOLD_MOVIES_FEATURES_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `id`
- Partition key:
  - `run_date`
- Required columns:
  - `id`
  - `title`
  - `overview`
  - `tags`
  - `run_date`
  - `run_id`
  - `ingestion_ts`
- Important feature columns:
  - `vector`
  - `popularity_score`
  - `quality_score`
  - `engagement_score`
  - `metadata_completeness`
  - `content_quality_score`
  - `quality_bucket`
  - `searchable`
  - `recommendable`
  - `is_popular`
  - `is_high_rated`
  - `is_recent`
- Quality gates:
  - feature rows must align one-to-one with curated movies
  - vector dimensions must be consistent within a run
  - derived features must be reproducible from silver inputs
- Downstream consumers:
  - model training
  - ANN index generation
  - serving artifact creation

### Movie Dimension SCD

- Owner: dimensional modeling pipeline
- Layer: gold or serving-adjacent warehouse
- Schema version: `1`
- Current implementation: `DIM_MOVIE_SCD_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key:
  - business key: `id`
  - version tracking: `record_hash`, `effective_start_at`
- Partition key: none currently defined in schema
- Quality gates:
  - one `is_current = true` record per movie ID
  - `record_hash` changes only when tracked attributes change
- Downstream consumers:
  - historical analysis
  - auditability of movie metadata changes

### Fact Movie Events

- Owner: event ingestion
- Layer: fact
- Schema version: `1`
- Current implementation: `FACT_MOVIE_EVENT_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `event_id`
- Partition key:
  - `event_date`
- Required columns:
  - `event_id`
  - `event_ts`
  - `event_type`
  - `event_date`
- Quality gates:
  - event IDs must be unique
  - `event_date` must match the event timestamp day convention
  - event type taxonomy must remain controlled
- Downstream consumers:
  - behavior aggregation
  - experimentation and usage reporting
  - future feature generation

### Embedding Jobs

- Owner: embedding refresh pipeline
- Layer: operational
- Schema version: `1`
- Current implementation: `MOVIE_EMBEDDING_JOBS_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `job_id`
- Partition key: none currently defined in schema
- Quality gates:
  - `movie_id` must exist in curated movie data
  - `tags_hash` should change only when embedding inputs change
  - status transitions should be explicit
- Downstream consumers:
  - embedding refresh orchestration
  - operational monitoring

### Pipeline Runs

- Owner: orchestrator and ETL
- Layer: operational
- Schema version: `1`
- Current implementation: `PIPELINE_RUN_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: `run_id`
- Partition key:
  - `run_date`
- Quality gates:
  - every batch run records a final status
  - input and output row counts are captured when available
  - failures preserve error context
- Downstream consumers:
  - run auditing
  - observability
  - recovery decisions

### Quarantine Movies

- Owner: data quality controls
- Layer: quarantine
- Schema version: `1`
- Current implementation: `QUARANTINE_MOVIES_SCHEMA` in `etl/delta_lakehouse.py`
- Primary key: no strict primary key today
- Partition key:
  - `run_date`
- Required columns:
  - `failure_reason`
  - `run_date`
  - `run_id`
  - `quarantined_at`
- Quality gates:
  - every quarantined row must include a human-readable failure reason
  - quarantine writes must not overwrite evidence of prior failures silently
- Downstream consumers:
  - debugging
  - backfill and recovery workflows

## Serving Artifact Contract

The current serving contract is enforced operationally through artifact
validation rather than a warehouse schema. The key source of truth is
`backend/serving/artifact_validator.py`.

The serving artifact contract currently assumes:

- artifacts are checksum validated by SHA-256
- artifact manifests are JSON objects keyed by artifact path or file name
- embedding arrays must align with movie rows
- validation should degrade gracefully for unknown manifest entries, but this
  should become stricter over time

Expected serving artifact families include:

- ANN indexes
- embedding arrays
- metadata tables
- ONNX or model export files
- manifest files

## Quality Enforcement Roadmap

The codebase already has the foundations for contract enforcement, but the
following improvements should be implemented next:

1. Add machine-readable schema files under `contracts/`.
2. Enforce the same required columns in both Pandas and Spark paths.
3. Convert warnings on invalid data into explicit failures for governed outputs.
4. Add duplicate key checks before bronze, silver, and gold writes.
5. Define controlled enums for fields like `quality_bucket`, `event_type`, and
   pipeline status.
6. Add CI tests for schema drift and invalid partition behavior.

## Definition Of Done

This document becomes operationally complete when:

- every major dataset has a machine-readable contract
- ETL fails fast on broken contracts
- contract checks run in automated tests
- serving artifacts have versioned metadata in addition to checksums
