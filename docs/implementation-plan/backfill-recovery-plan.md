# Backfill And Recovery Plan

## Objective

Make the ETL and artifact workflow idempotent, recoverable, and easy to
operate.

## Why This Matters

FAANG-style AI + DE review strongly favors systems that can be repaired and
rerun safely after partial failure or bad data.

## Files To Add

- `docs/BACKFILL_AND_RECOVERY.md`
- `scripts/backfill_gold_window.py`
- `scripts/rebuild_from_gold.py`
- `tests/test_backfill_recovery.py`

## Files To Modify

- `etl/pipeline.py`
- `scripts/rebuild_serving_artifacts.py`

## Recovery Scenarios To Cover

- rerun the last successful window
- rebuild one bad partition
- rebuild a bounded date range
- recreate serving artifacts from gold data
- recover from partial output writes
- validate outputs before promotion

## Implementation Steps

1. Document common failure modes and their recovery paths.
2. Add a deterministic backfill CLI for date or partition windows.
3. Add a rebuild-from-gold path for serving artifacts.
4. Expose standard run, backfill, and recovery flows from one entrypoint.
5. Add tests for rerun safety and artifact rebuild correctness.

## Acceptance Criteria

- Rebuilding the same window twice produces consistent outputs.
- Recovering from a bad partition is documented and testable.
- Serving artifacts can be reproduced from governed gold data.
- Operators can follow a single runbook without tribal knowledge.
