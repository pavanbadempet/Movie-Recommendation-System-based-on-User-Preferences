# Lineage And Artifact Governance Plan

## Objective

Make every important model or serving artifact traceable back to its source
datasets, schema version, code version, and training configuration.

## Why This Matters

Lineage is a strong AI platform signal because it proves the project supports
reproducibility, debugging, and safe promotion.

## Files To Add

- `docs/LINEAGE.md`
- `docs/MODEL_REGISTRY.md`
- `metadata/artifact_manifest.example.json`
- `metadata/training_run.example.json`

## Files To Modify

- `backend/serving/mlops_engine.py`
- `backend/serving/artifact_validator.py`
- `scripts/train_apex_models.py`

## Metadata To Record

- dataset version
- schema version
- code commit
- training config
- metrics summary
- artifact checksum
- promotion status
- serving version

## Implementation Steps

1. Define a simple artifact manifest format.
2. Define a simple training run metadata format.
3. Persist lineage fields during training and artifact generation.
4. Validate manifests alongside file integrity checks.
5. Document the raw -> curated -> training -> artifact -> serving path.

## Acceptance Criteria

- Any artifact can be traced back to its dataset and config.
- Drift and evaluation outputs reference concrete versions.
- Artifacts without valid metadata fail validation.
- The lineage chain is documented in a single place.
