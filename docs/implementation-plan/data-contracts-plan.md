# Data Contracts Plan

## Objective

Formalize machine-readable dataset contracts and enforce them in both Pandas and
Spark ETL paths.

## Why This Matters

This is the fastest way to improve AI + DE credibility because it proves the
project has governed data rather than ad hoc pipelines.

## Files To Add

- `docs/DATA_CONTRACTS.md`
- `contracts/raw_events.schema.json`
- `contracts/bronze_movies.schema.json`
- `contracts/silver_movies.schema.json`
- `contracts/gold_training_set.schema.json`
- `tests/test_data_contracts.py`

## Files To Modify

- `etl/pandas_etl.py`
- `etl/pyspark_etl.py`

## Contract Requirements

Each dataset contract should define:

- schema version
- owner
- primary key
- partition key
- required columns
- nullable columns
- allowed enums or ranges
- freshness expectation
- downstream consumers

## Implementation Steps

1. Create human-readable contract docs in `docs/DATA_CONTRACTS.md`.
2. Add JSON schema files for the main datasets.
3. Add validation helpers shared by ETL code.
4. Validate required columns, types, nulls, and duplicate keys before writes.
5. Fail fast with actionable error messages.
6. Add tests for pass and fail cases.

## Acceptance Criteria

- ETL fails on schema drift.
- ETL fails on duplicate primary keys.
- ETL fails on nulls in required columns.
- Spark and Pandas paths enforce the same expectations.
- Tests prove the failure modes.
