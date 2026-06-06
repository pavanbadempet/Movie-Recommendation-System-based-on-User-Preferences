# Implementation Plan: TurboVec Migration

## Overview

Replace all FAISS imports and index artifact references with TurboVec (`turbovec.TurboQuantIndex`)
equivalents across every pipeline, serving, training, validation, and test layer. The migration
proceeds in isolation per layer — ETL first, then serving infrastructure, then training scripts,
then utilities, and finally the test suite — ensuring the artifact contract flows cleanly
top-to-bottom with no serving disruption.

## Tasks

- [x] 1. Update Pandas ETL pipeline to build and persist TurboVec index
  - [x] 1.1 Implement `build_turbovec_index` and `atomic_write_turbovec_index` in `etl/pandas_etl.py`
    - Remove `import faiss`, `build_faiss_index`, and `atomic_write_faiss_index`
    - Add `from turbovec import TurboQuantIndex`
    - Implement `build_turbovec_index(vectors: np.ndarray) -> TurboQuantIndex` with `bit_width=4`, `index.add()`, and `ntotal` verification
    - Implement `atomic_write_turbovec_index` using write-to-temp-then-rename pattern
    - Update `build_index` caller to use the new functions and write to `models/turbovec.tq`
    - Update `build_serving_contract` and `assert_batch_invariants` to use `turbovec_index_size` key
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 1.2 Write property test for build preserves vector count (P1)
    - **Property 1: Index Build Preserves Vector Count**
    - Use `st.integers(1, 500)` for n and `st.integers(16, 1024)` for dim
    - Assert `index.ntotal == n` for all valid inputs
    - Place in `tests/test_turbovec_etl.py`
    - **Validates: Requirements 1.1, 1.6, 1.7, 2.1, 6.1, 7.1**

  - [ ]* 1.3 Write property test for write-then-load round trip (P2)
    - **Property 2: Write-then-Load Round Trip Preserves ntotal**
    - Use same generators as P1; write index to a `tmp_path`, load it, assert `ntotal == n`
    - Place in `tests/test_turbovec_etl.py`
    - **Validates: Requirements 1.3, 2.3, 6.3, 6.4, 7.3**

  - [ ]* 1.4 Write property test for manifest turbovec_index_size (P3)
    - **Property 3: Manifest Turbovec Index Size Matches ntotal**
    - Assert manifest contains `turbovec_index_size == n` and does NOT contain `faiss_index_size` or `faiss_index`
    - Mock filesystem; place in `tests/test_turbovec_etl.py`
    - **Validates: Requirements 2.5, 3.1, 3.4**

  - [ ]* 1.5 Write unit tests for Pandas ETL atomic write and empty-batch edge cases
    - Test `test_atomic_write_no_temp_file_left`: successful write leaves no `.tmp` file
    - Test empty vector batch builds and persists without error
    - _Requirements: 1.4, 1.7_

- [x] 2. Update PySpark ETL pipeline to build and persist TurboVec index
  - [x] 2.1 Replace FAISS indexing block with TurboVec in `etl/pyspark_etl.py`
    - Remove `import faiss`, `faiss.index_factory`, `index.train()`, `faiss.write_index`
    - Add `from turbovec import TurboQuantIndex`; build index with `bit_width=4`
    - Write index to `models/turbovec.tq` via `index.write()`
    - Update manifest `artifacts` map to `"turbovec_index": "turbovec.tq"`
    - Update manifest `serving_contract` to `"turbovec_index_size": int(index.ntotal)`
    - Remove any `"faiss_index"` and `"faiss_index_size"` keys from manifest output
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.4_

  - [ ]* 2.2 Write unit tests for PySpark ETL manifest schema correctness
    - Verify manifest contains `turbovec_index` and `turbovec_index_size` and not FAISS keys
    - Test zero-sized index (no vectors) builds and writes without error
    - _Requirements: 2.1, 2.4, 2.5, 3.4_

- [x] 3. Checkpoint — Ensure all ETL tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update Pipeline Manifest schema and Model Loader
  - [x] 4.1 Update `backend/models/model_loader.py` to register `turbovec.tq` and remove `faiss.index`
    - Remove `"faiss.index"` entry from `MODEL_FILES`
    - Add `"turbovec.tq"` entry with `url` from `TURBOVEC_INDEX_URL` env var
    - Update `_manifest_contract_matches` to handle `turbovec.tq`: load via `TurboQuantIndex.load`, compare `ntotal` against `turbovec_index_size` (fallback: `faiss_index_size`)
    - Update `_load_manifest_contract` fallback loop to include `"turbovec_index_size"`
    - Update `default_artifacts_for_serving_profile`: include `turbovec.tq` in full profile, exclude from lite/low-memory profile
    - Log warning and return `False` for `turbovec.tq` download when `TURBOVEC_INDEX_URL` is not set
    - _Requirements: 3.3, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 4.2 Write unit tests for Model Loader TurboVec behavior
    - Test `test_missing_url_returns_false`: `TURBOVEC_INDEX_URL` not set → result is `False`, warning logged
    - Test `test_lite_profile_excludes_turbovec`: lite mode excludes `turbovec.tq`
    - Test `test_manifest_fallback_key`: manifest with only `faiss_index_size` is read by loader without error
    - Test `test_backward_compat_manifest`: manifest with `faiss_index_size` passes validation without error
    - _Requirements: 3.3, 3.5, 4.5, 4.7_

- [x] 5. Update Recommender to load and search TurboVec index
  - [x] 5.1 Replace FAISS vector artifact loading in `backend/pipeline/recommender_core.py`
    - Remove `import faiss` and `faiss.read_index`
    - Add `from turbovec import TurboQuantIndex`
    - Replace vector artifact loading block: load from `models/turbovec.tq` via `TurboQuantIndex.load`
    - Implement fallback when `turbovec.tq` absent and `faiss.index` present: log warning with `"turbovec.tq not found"` and expected path, set `vector_artifacts_ready = False`
    - Validate `index.ntotal == len(catalog)`; call `_disable_vector_artifacts` if mismatch
    - Update `ensure_model_files` call to use `"turbovec.tq"` instead of `"faiss.index"`
    - Update ANN search call to `index.search(query_embedding, k)` returning `(scores, indices)`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 8.1, 8.2, 8.3_

  - [ ]* 5.2 Write property test for missing turbovec.tq disables vector artifacts (P5)
    - **Property 5: Missing turbovec.tq Disables Vector Artifacts**
    - Use filesystem fixture with no `turbovec.tq`; assert `vector_artifacts_ready == False` and no index load attempted
    - Place in `tests/test_recommender_core.py`
    - **Validates: Requirements 5.4, 8.1, 8.2, 8.3**

  - [ ]* 5.3 Write property test for catalog row count invariant (P6)
    - **Property 6: Catalog Row Count Invariant**
    - Generate random `ntotal` and random catalog size; when they differ assert `_disable_vector_artifacts` is called
    - Place in `tests/test_recommender_core.py`
    - **Validates: Requirements 5.5, 5.6**

  - [ ]* 5.4 Write integration test for recommender loading turbovec.tq
    - Test `test_recommender_loads_turbovec_tq`: create `turbovec.tq` in temp dir; Recommender loads it; dense retrieval enabled
    - _Requirements: 5.1, 5.5_

- [~] 6. Checkpoint — Ensure all serving layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Update Multimodal Fusion module
  - [x] 7.1 Replace FAISS flat index with TurboVec in `backend/intelligence/multimodal_fusion.py`
    - Remove `import faiss`, `faiss.IndexFlatIP`, `faiss.write_index`, `faiss.read_index`
    - Add `from turbovec import TurboQuantIndex`
    - In `build_fusion_index`: build `TurboQuantIndex(self.total_dim, bit_width=4)`, call `index.add(fused_vectors)`, write to `models/multimodal_turbovec.tq`
    - In `load_fusion_index`: load from `models/multimodal_turbovec.tq` via `TurboQuantIndex.load`
    - `search()` call site unchanged — `index.search(fused_query, top_k)` returns same `(distances, indices)` shape
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 7.2 Write unit tests for Multimodal Fusion TurboVec index
    - Test build, persist, load, and search round trip with a small embedding matrix
    - _Requirements: 6.1, 6.3, 6.4, 6.5_

- [x] 8. Update Two-Tower training script
  - [x] 8.1 Replace `export_to_faiss` with `export_to_turbovec` in `scripts/train_two_tower.py`
    - Remove `import faiss`, `faiss.IndexFlatIP`, `faiss.write_index`
    - Add `from turbovec import TurboQuantIndex`
    - Rename `export_to_faiss` → `export_to_turbovec`; build `TurboQuantIndex(dim, bit_width=4)`, call `index.add()`, write to `models/two_tower_turbovec.tq`
    - Update `main()` output path to `MODELS_DIR / "two_tower_turbovec.tq"`
    - Log TurboVec index path and `index.ntotal` row count in summary
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 8.2 Write unit tests for Two-Tower export
    - Test that `export_to_turbovec` writes `two_tower_turbovec.tq` and logs correct row count
    - _Requirements: 7.3, 7.4_

- [x] 9. Create migration utility and recall evaluation script
  - [x] 9.1 Create `scripts/migrate_faiss_to_turbovec.py`
    - Implement `migrate(faiss_path, output_path)`: load FAISS index via `faiss.read_index`, extract vectors via `reconstruct_n`, build `TurboQuantIndex(dim, bit_width=4)`, add vectors, verify `ntotal` match, write `turbovec.tq`
    - Print summary: source file size, output file size, row count
    - Exit with error if row count mismatch
    - _Requirements: 8.4, 8.5, 8.6_

  - [x] 9.2 Create `scripts/evaluate_turbovec_recall.py`
    - Implement `evaluate_recall(turbovec_path, embeddings_path, n_queries=1000, k_values=(10, 50))`
    - Load production index, sample 1,000 query vectors (seed=42), compute TurboVec and brute-force inner-product results, calculate Recall@K
    - Log all K values; exit with code 1 if Recall@10 < 0.90
    - _Requirements: 9.3, 9.4_

  - [ ]* 9.3 Write property test for migration preserves row count (P7)
    - **Property 7: Migration Preserves Row Count**
    - Use `st.integers(1, 200)` for n; build a synthetic FAISS index, run migration, assert `turbovec ntotal == n`
    - Place in `tests/test_migrate_script.py`
    - **Validates: Requirements 8.5**

  - [ ]* 9.4 Write property test for recall threshold (P8)
    - **Property 8: Recall@K Meets Quality Threshold**
    - Fixed seed; 1,000 queries from SBERT corpus; assert Recall@10 ≥ 0.92 and Recall@50 ≥ 0.97
    - Place in `tests/test_turbovec_recall.py`
    - **Validates: Requirements 9.1, 9.2**

  - [ ]* 9.5 Write unit tests for migration and recall scripts
    - Test `test_migration_summary_output`: migration prints source/output sizes and row count
    - Test `test_recall_script_exits_nonzero`: when Recall@10 < 0.90, script exits with code 1
    - _Requirements: 8.6, 9.4_

- [x] 10. Update CI/CD and artifact validation scripts
  - [x] 10.1 Update `scripts/validate_serving_artifacts.py`
    - Replace `"faiss.index"` with `"turbovec.tq"` in `REQUIRED_FILES`
    - Replace `"faiss_index_size": "faiss.index"` with `"turbovec_index_size": "turbovec.tq"` in `HEAVY_ARTIFACT_CONTRACTS`
    - Update `validate()` to use `turbovec_index_size` (fallback: `faiss_index_size`) for `expected_index_size` lookup
    - _Requirements: 11.1, 11.2_

  - [x] 10.2 Update `scripts/backfill_serving_metadata_artifacts.py`
    - Remove `describe_faiss_artifact` and `import faiss`
    - Add `describe_turbovec_artifact` using `TurboQuantIndex.load` to read `ntotal`
    - Update `build_backfill_artifacts` to use `turbovec_path` parameter and write `"turbovec_index": "turbovec.tq"` and `"turbovec_index_size"` in manifest
    - _Requirements: 11.3_

  - [x] 10.3 Update `scripts/rebuild_serving_artifacts.py`
    - Replace `atomic_write_faiss_index` / `build_faiss_index` imports with `atomic_write_turbovec_index` / `build_turbovec_index` from `etl.pandas_etl`
    - Replace `faiss_path` local variable with `turbovec_path`
    - Update `build_backfill_artifacts` call to pass `turbovec_path=turbovec_path`
    - _Requirements: 11.5_

  - [x] 10.4 Update `manage.py` required artifacts list
    - Replace `"models/faiss.index"` with `"models/turbovec.tq"` in artifacts list
    - _Requirements: 11.4_

  - [ ]* 10.5 Write integration test for rebuild and backfill scripts
    - Test `test_rebuild_produces_turbovec_tq`: `rebuild_serving_artifacts` writes `turbovec.tq` and correct manifest
    - Test `test_backfill_records_turbovec_key`: `build_backfill_artifacts` with turbovec_path writes `turbovec_index_size`
    - _Requirements: 11.3, 11.5_

- [x] 11. Update test suite — replace MockFaissIndex with MockTurboVecIndex
  - [x] 11.1 Create `MockTurboVecIndex` and update all test files
    - Define `MockTurboVecIndex` class with `ntotal`, `_dim`, and `search(query, k) -> (scores, indices)` returning shapes `(1, k)`
    - Replace every `MockFaissIndex` reference in the test suite with `MockTurboVecIndex`
    - Remove `import faiss` from all test files
    - Update test Pipeline_Manifests to use `"turbovec_index_size"` and `"turbovec.tq"`
    - Update integration tests to build and persist index as `turbovec.tq` in temp directories
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 11.2 Update `retrieval_source` generator in property-based tests
    - Replace or update `RETRIEVAL_SOURCES` strategy to include `"turbovec"` as a valid value
    - Place updated generator in the relevant property-based test modules
    - _Requirements: 10.6_

  - [ ]* 11.3 Write property test for search output shape (P4)
    - **Property 4: Search Returns Correct Output Shape**
    - Use `st.integers(1, 100)` for k; assert both `scores` and `indices` have shape `(1, k)`
    - Apply to both `MockTurboVecIndex` and any integration with real `TurboQuantIndex`
    - Place in `tests/test_turbovec_mock.py`
    - **Validates: Requirements 5.2, 5.3, 6.5, 10.1, 10.2**

  - [ ]* 11.4 Write property test for retrieval source generator includes TurboVec (P9)
    - **Property 9: Retrieval Source Generator Includes TurboVec**
    - Assert `"turbovec"` is reachable within the `st.sampled_from(RETRIEVAL_SOURCES)` strategy
    - Place in `tests/test_retrieval_generators.py`
    - **Validates: Requirements 10.6**

- [x] 12. Smoke-test static correctness checks
  - [x] 12.1 Add smoke tests asserting no `import faiss` in production modules
    - Assert `etl/pandas_etl.py`, `etl/pyspark_etl.py`, `backend/pipeline/recommender_core.py`, `backend/intelligence/multimodal_fusion.py`, `scripts/train_two_tower.py` contain no `import faiss` statement
    - Assert `MODEL_FILES` in `model_loader.py` does not contain `"faiss.index"`
    - Assert `REQUIRED_FILES` in `validate_serving_artifacts.py` contains `"turbovec.tq"`
    - Assert `scripts/migrate_faiss_to_turbovec.py` and `scripts/evaluate_turbovec_recall.py` exist on disk
    - _Requirements: 1.5, 2.6, 4.6, 5.7, 6.6, 7.5_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for full traceability
- The migration is strictly layered: ETL → serving infrastructure → training/utilities → test suite
- Property 8 (Recall@K) requires the production SBERT embedding corpus; it may be skipped in environments without the full dataset
- The migration utility (`scripts/migrate_faiss_to_turbovec.py`) is the only file that intentionally retains a `faiss` import — it is a one-time transition tool, not a production module
- Backward-compatibility fallbacks (`faiss_index_size` → `turbovec_index_size`) are read-only; no newly written manifests should contain FAISS keys

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4", "1.5", "2.2"] },
    { "id": 2, "tasks": ["4.1", "7.1", "8.1", "9.1", "9.2", "10.4"] },
    { "id": 3, "tasks": ["4.2", "5.1", "7.2", "8.2", "9.3", "9.4", "9.5", "10.1", "10.2", "10.3"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "10.5", "11.1", "11.2"] },
    { "id": 5, "tasks": ["11.3", "11.4", "12.1"] }
  ]
}
```
