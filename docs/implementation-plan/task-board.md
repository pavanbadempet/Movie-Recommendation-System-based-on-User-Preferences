# AI + DE Task Board

## P0

### Data Contracts

- Add `docs/DATA_CONTRACTS.md`
- Add `contracts/*.schema.json`
- Enforce contracts in `etl/pandas_etl.py`
- Enforce contracts in `etl/pyspark_etl.py`
- Add `tests/test_data_contracts.py`

Outcome: governed datasets with enforceable schemas.

## P1

### Backfill And Recovery

- Add `docs/BACKFILL_AND_RECOVERY.md`
- Add `scripts/backfill_gold_window.py`
- Add `scripts/rebuild_from_gold.py`
- Update `etl/pipeline.py`
- Add `tests/test_backfill_recovery.py`

Outcome: pipelines are idempotent and recoverable.

### Lineage And Governance

- Add `docs/LINEAGE.md`
- Add `docs/MODEL_REGISTRY.md`
- Add `metadata/artifact_manifest.example.json`
- Add `metadata/training_run.example.json`
- Update `backend/serving/mlops_engine.py`
- Update `backend/serving/artifact_validator.py`
- Update `scripts/train_apex_models.py`

Outcome: artifacts and training runs are traceable.

## P2

### Feature Consistency

- Add `docs/FEATURE_CATALOG.md`
- Update `backend/serving/feature_store.py`
- Add `tests/test_offline_online_feature_parity.py`

Outcome: key serving features are aligned with training semantics.

### Orchestration Credibility

- Add `docs/ORCHESTRATION_SLA.md`
- Update `airflow/dags/refresh_dag.py`
- Update `airflow/dags/kafka_spark_integration_dag.py`
- Update `docker-compose.yml`

Outcome: orchestration assumptions are explicit and consistent.

## P3

### Platform Proof

- Add `docs/EVALUATION_METHODOLOGY.md`
- Add `docs/OPERATIONS_SCORECARD.md`
- Update `README.md`

Outcome: the repo communicates measurable platform maturity.

## Best First Sprint

1. Create `docs/DATA_CONTRACTS.md`.
2. Add `contracts/*.schema.json`.
3. Wire contract validation into ETL.
4. Add `tests/test_data_contracts.py`.
5. Create `docs/BACKFILL_AND_RECOVERY.md`.
