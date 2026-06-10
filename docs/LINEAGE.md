# Lineage

This document describes the current lineage model for data, features, training
outputs, and serving artifacts in the repository. It is based on the concrete
assets that already exist in:

- `etl/pandas_etl.py`
- `etl/delta_lakehouse.py`
- `backend/serving/mlops_engine.py`
- `backend/serving/artifact_validator.py`
- `scripts/train_apex_models.py`
- `scripts/rebuild_serving_artifacts.py`

## Goal

Provide a simple answer to the question:

"Where did this artifact come from, and can we reproduce it?"

## Current Lineage Model

The current lineage story already has useful building blocks:

- dataset schemas and table definitions in `etl/delta_lakehouse.py`
- run-level metadata fields like `run_id` and `run_date`
- serving contract versioning in `etl/pandas_etl.py`
- artifact checksums in `backend/serving/artifact_validator.py`
- run registry and promotion status in `backend/serving/mlops_engine.py`

The missing piece is one unified, explicit chain from source data to active
serving artifacts.

## End-To-End Flow

### 1. Source data

Source movie data enters through raw files, currently represented by the movie
input schema in `etl/pandas_etl.py`.

Lineage fields that should be captured:

- source file path
- source file hash
- source acquisition timestamp
- raw schema version

### 2. Bronze ingest

Raw source rows are normalized into the bronze layer defined by
`BRONZE_MOVIES_SCHEMA` in `etl/delta_lakehouse.py`.

Current lineage fields already implied by the schema:

- `run_date`
- `run_id`
- `ingestion_ts`

### 3. Silver curation

Curated movie rows are produced in the silver layer, where content quality,
searchability, recommendation eligibility, and tags are defined.

Lineage should connect silver outputs to:

- source bronze run
- transform version
- quality rules applied
- quarantine exclusions, if any

### 4. Gold feature generation

Feature rows and vectors are produced in the gold layer via
`GOLD_MOVIES_FEATURES_SCHEMA`.

Lineage should connect gold outputs to:

- source silver run
- feature generation code version
- vector model name
- vector dimension
- feature statistics summary

### 5. Training and model generation

Training scripts and model export steps consume curated inputs and emit model
artifacts or indexes.

Lineage should connect training outputs to:

- source dataset version or run window
- model name and training config
- code commit
- metric summary
- output artifact paths

### 6. Serving artifact build

Serving artifacts are rebuilt, validated, and published through scripts like
`scripts/rebuild_serving_artifacts.py` and validated by
`backend/serving/artifact_validator.py`.

Lineage should connect each serving artifact to:

- source gold dataset run
- model or export version
- checksum
- manifest version
- row alignment status

### 7. Runtime promotion

The lightweight promotion and drift status registry in
`backend/serving/mlops_engine.py` can already mark runs as:

- `promoted`
- `needs_review`

Lineage should connect promotion status to:

- drift analysis results
- benchmark summary
- prior active version
- rollback target

## Recommended Lineage Keys

The following keys should become standard across data, training, and artifact
metadata:

- `run_id`
- `run_date`
- `pipeline_name`
- `pipeline_version`
- `schema_version`
- `source_dataset_version`
- `source_run_id`
- `git_commit`
- `training_config_version`
- `artifact_name`
- `artifact_version`
- `artifact_sha256`
- `promotion_status`

## Minimal Metadata Records

### Training run record

A training run record should capture:

- run identifier
- training window
- input dataset references
- model family
- config summary
- benchmark summary
- output artifact list

### Artifact manifest record

An artifact manifest should capture:

- artifact file name
- checksum
- originating run ID
- schema version
- source dataset version
- build timestamp
- validation status

## Lineage Table

| Stage | Primary identifier | Minimum lineage fields | Current code anchor |
| --- | --- | --- | --- |
| Source input | source file or batch | source path, hash, raw schema version | `etl/pandas_etl.py` |
| Bronze | `run_id` + `run_date` | ingest timestamp, source reference | `BRONZE_MOVIES_SCHEMA` |
| Silver | `run_id` + `run_date` | transform version, quality outputs | `SILVER_MOVIES_SCHEMA` |
| Gold | `run_id` + `run_date` | feature version, vector stats | `GOLD_MOVIES_FEATURES_SCHEMA` |
| Training | training run ID | config, dataset refs, metrics | `scripts/train_apex_models.py` |
| Serving artifact | artifact name or version | checksum, source run, validation result | `artifact_validator.py` |
| Promotion | promoted run ID | drift summary, status, rollback target | `mlops_engine.py` |

## Current Gaps

The repository has partial lineage today, but it is not yet unified. The main
gaps are:

1. No single artifact manifest standard that includes dataset and config lineage.
2. No standard training metadata file emitted on every successful run.
3. No explicit link between artifact validation and promotion decisions.
4. No standard rollback metadata for restoring the last known-good serving set.

## Next Implementation Steps

1. Add `metadata/artifact_manifest.example.json`.
2. Add `metadata/training_run.example.json`.
3. Extend `backend/serving/mlops_engine.py` to store dataset and artifact lineage.
4. Extend `backend/serving/artifact_validator.py` to validate manifest metadata,
   not just checksums.
5. Make rebuild scripts emit lineage metadata as part of artifact generation.

## Definition Of Done

Lineage is operationally credible when:

- every important artifact has a checksum and source run reference
- every training run records its input dataset and config
- promotion decisions reference a known artifact version
- rollback targets are identifiable without manual digging
