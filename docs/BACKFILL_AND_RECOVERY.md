# Backfill And Recovery

This document defines the intended operating model for reruns, backfills, and
recovery in the current repository. It is based on the batch and artifact paths
already present in:

- `etl/pipeline.py`
- `etl/pandas_etl.py`
- `etl/pyspark_etl.py`
- `scripts/rebuild_serving_artifacts.py`
- `scripts/backfill_serving_metadata_artifacts.py`
- `scripts/validate_serving_artifacts.py`
- `backend/serving/artifact_validator.py`

## Goals

- make reruns safe
- make failures diagnosable
- make recovery steps explicit
- recreate serving artifacts from governed outputs instead of ad hoc state

## Operating Principles

### Prefer deterministic reruns

Every rerun should produce the same output for the same input window and
configuration, except when the upstream source itself has changed.

### Keep recovery bounded

Recover the smallest affected unit first:

- one artifact
- one partition
- one run window
- full rebuild only as a last resort

### Validate before promotion

Any rebuilt serving artifact should be validated before it is considered ready
for the API or deployment workflow.

## Current Recovery Surfaces

### Batch ETL

The current codebase supports both:

- local fallback ETL via `etl/pandas_etl.py`
- canonical Spark and lakehouse processing via `etl/pyspark_etl.py` and
  `etl/delta_lakehouse.py`

The batch recovery story should be centered on:

- `run_id`
- `run_date`
- partition-aware outputs
- row counts and quality metrics
- quarantine outputs for bad rows

### Serving artifacts

The current artifact workflow already includes:

- manifest creation
- checksum validation
- row alignment validation
- rebuild scripts
- metadata backfill scripts

These form the basis of a reproducible serving recovery path.

## Failure Modes

### Bad source batch

Symptoms:

- schema drift
- null explosion in required columns
- duplicate movie IDs
- vote or metadata quality fields outside expected ranges

Initial action:

- stop promotion of downstream outputs
- preserve the failed run metadata
- quarantine invalid rows when supported

Preferred recovery:

1. fix the source or mapping rule
2. rerun the affected batch window
3. compare row counts and quality metrics with the previous successful run

### Partial batch write

Symptoms:

- missing files in a target layer
- incomplete partition output
- run metadata says started but not finished

Initial action:

- mark the run as failed or incomplete
- remove or isolate the partial output if the write path is not atomic

Preferred recovery:

1. rebuild only the affected partition or run window
2. validate row counts and partition completeness
3. only then continue to downstream feature and artifact generation

### Artifact checksum mismatch

Symptoms:

- `backend/serving/artifact_validator.py` raises checksum mismatch
- startup validation fails
- serving lineage and checksum metadata no longer agree

Preferred recovery:

1. identify the affected artifact
2. rebuild it from the latest valid curated dataset or gold output
3. regenerate the manifest
4. rerun checksum and row-alignment validation

### Artifact row-alignment mismatch

Symptoms:

- embeddings and movie metadata have different row counts
- ANN index is built from a different movie ordering than serving metadata

Preferred recovery:

1. rebuild the aligned metadata and embedding outputs from the same source run
2. validate row alignment before publishing
3. avoid mixing files from different run IDs

### Drift or quality regression after a successful build

Symptoms:

- drift checks indicate promotion should pause
- benchmark quality falls below expectations
- live serving checks show degraded readiness or poor output quality

Preferred recovery:

1. keep the previous known-good artifact active
2. mark the new run as `needs_review`
3. inspect source deltas, feature generation, and artifact lineage
4. either patch forward or rollback to the prior promoted version

## Recovery Units

### Artifact-only rebuild

Use when:

- curated datasets are valid
- only serving files or manifests are bad

Target scripts:

- `scripts/rebuild_serving_artifacts.py`
- `scripts/validate_serving_artifacts.py`
- `scripts/backfill_serving_metadata_artifacts.py`

### Partition or run-window backfill

Use when:

- one or more batch dates are invalid
- curated tables need correction

Preferred future command shape:

```bash
python -m etl.pipeline backfill --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

### Full rebuild from curated gold

Use when:

- serving artifacts are inconsistent
- feature generation logic changed
- checksum lineage is no longer trusted

Preferred future command shape:

```bash
python scripts/rebuild_from_gold.py --run-date YYYY-MM-DD
```

## Standard Recovery Playbooks

### Playbook 1: Rebuild serving artifacts from a known-good curated dataset

1. Identify the last valid run ID and run date.
2. Confirm curated gold outputs are present and complete.
3. Rebuild serving artifacts.
4. Regenerate or refresh the manifest.
5. Run artifact validation.
6. Only then update the active serving set.

### Playbook 2: Repair a bad batch window

1. Identify the affected date range.
2. Confirm whether the issue started in raw ingest, bronze, silver, or gold.
3. Correct the source mapping or transform logic.
4. Reprocess the smallest affected window.
5. Recompute downstream artifacts if gold changed.
6. Compare counts, quality metrics, and drift checks against the last good run.

### Playbook 3: Recover from partial output

1. Detect incomplete writes using missing files, incomplete partitions, or run
   metadata.
2. Isolate or delete only the incomplete target output.
3. Rerun the exact same window and configuration.
4. Validate outputs before allowing downstream tasks to continue.

## Required Metadata For Recovery

Every governed batch run should record:

- `run_id`
- `run_date`
- pipeline name and version
- status
- started and finished timestamps
- input and output row counts
- artifact paths or hashes when applicable
- error message on failure

The current `PIPELINE_RUN_SCHEMA` in `etl/delta_lakehouse.py` already provides a
strong starting point for this metadata.

## Gaps To Close

The repository already has the right building blocks, but the following work is
still needed:

1. Add a dedicated backfill CLI for bounded date windows.
2. Add a rebuild-from-gold script instead of only general artifact rebuilds.
3. Make write semantics and idempotency guarantees explicit in code and docs.
4. Add automated tests for partial-output recovery and rerun safety.
5. Tighten artifact validation so metadata lineage is checked along with checksums.

## Definition Of Done

Recovery is credible when:

- a bounded backfill can be run intentionally
- serving artifacts can be rebuilt from curated data
- validation runs before promotion
- failed runs leave enough metadata to support debugging
- playbooks are specific enough for another engineer to follow
