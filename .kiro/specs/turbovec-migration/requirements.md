# Requirements Document

## Introduction

This feature migrates the Movie Recommendation System's vector index from FAISS to TurboVec
(`turbovec.TurboQuantIndex`). TurboVec is a Rust-based ANN library with Python bindings,
built on Google Research's TurboQuant algorithm (ICLR 2026). It provides a drop-in inner-product
search interface with no training step, 2–4 bits-per-dimension compression, and hand-written
SIMD kernels that match or beat FAISS performance.

The migration touches every layer that currently imports or references FAISS: the Pandas ETL
pipeline, the PySpark ETL pipeline, the serving-time recommender, the multimodal fusion module,
the two-tower training script, the model-loader artifact contract, and all related tests and
manifest schemas. The primary goals are: replace all FAISS imports with TurboVec equivalents,
preserve the existing artifact contract and serving quality, and provide a clean backward-
compatibility bridge so existing `faiss.index` files on disk are either migrated or gracefully
handled during transition.

---

## Glossary

- **TurboVec**: The `turbovec` Python package (`TurboQuantIndex` class), the replacement vector index.
- **TurboVec_Index**: An instance of `turbovec.TurboQuantIndex` used as the ANN index.
- **FAISS_Index**: A legacy FAISS index file (extension `.faiss.index` or `faiss.index`) produced by
  previous pipeline runs.
- **TurboVec_Index_File**: A serialized TurboVec index file written with `index.write(path)`,
  using the `.tq` extension (e.g., `turbovec.tq`).
- **Pipeline_Manifest**: The `pipeline_manifest.json` artifact that records the serving contract
  (row counts, checksums, artifact names) consumed by the model loader and health checks.
- **Serving_Contract**: The portion of the Pipeline_Manifest that the serving layer validates
  before trusting vector artifacts.
- **ETL_Pipeline**: Either `etl/pandas_etl.py` or `etl/pyspark_etl.py` — the batch process that
  produces embeddings and the vector index.
- **Recommender**: `backend/pipeline/recommender.py` — the serving-time class that loads the
  vector index and executes ANN search at request time.
- **Model_Loader**: `backend/models/model_loader.py` — the artifact downloader and validator that
  checks vector index integrity before serving.
- **Multimodal_Fusion**: `backend/intelligence/multimodal_fusion.py` — the multimodal retrieval
  module that builds and searches a fused text+vision vector index.
- **Two_Tower_Script**: `scripts/train_two_tower.py` — exports trained item embeddings into a
  separate vector index used for candidate generation evaluation.
- **Backward_Compatibility_Window**: The transition period during which both `faiss.index` (legacy)
  and `turbovec.tq` (new) may exist on disk; the system must handle both.
- **Recall@K**: The fraction of true top-K nearest neighbours correctly returned by the index
  for a given query.
- **bit_width**: The TurboVec quantization precision parameter (2 or 4 bits per dimension).
  4-bit provides higher recall; 2-bit provides higher compression.

---

## Requirements

### Requirement 1: Replace FAISS Index Build in Pandas ETL

**User Story:** As a data engineer, I want the Pandas ETL pipeline to build and persist a
TurboVec index instead of a FAISS index, so that downstream serving uses the new engine
without changing the pipeline invocation contract.

#### Acceptance Criteria

1. WHEN the Pandas ETL indexing stage runs, THE ETL_Pipeline SHALL build a `TurboQuantIndex`
   with `dim` equal to the embedding dimensionality and `bit_width=4`.
2. WHEN the TurboVec index is built, THE ETL_Pipeline SHALL add all normalized embedding
   vectors to the index via `index.add(vectors)` without a separate training step.
3. WHEN the indexing stage completes successfully, THE ETL_Pipeline SHALL write the TurboVec
   index to `models/turbovec.tq` using `index.write(path)`.
4. WHEN writing the TurboVec index, THE ETL_Pipeline SHALL use an atomic write pattern
   (write to a temp file then rename) to prevent serving from reading a partially written file.
5. WHEN the indexing stage runs, THE ETL_Pipeline SHALL NOT import or call any `faiss` module
   symbols.
6. THE ETL_Pipeline SHALL export a `build_turbovec_index(vectors: np.ndarray) -> TurboQuantIndex`
   function as the public indexing entry point, replacing `build_faiss_index`.
7. WHEN the index is built with one or more vectors, THE ETL_Pipeline SHALL verify that
   `index.ntotal` equals the number of vectors added and SHALL raise `ValueError` if the
   counts differ; WHEN no vectors are provided, THE ETL_Pipeline SHALL allow the resulting
   empty index to be built and persisted without raising an error.

---

### Requirement 2: Replace FAISS Index Build in PySpark ETL

**User Story:** As a data engineer, I want the PySpark ETL pipeline to produce a TurboVec
index instead of a FAISS SQfp16 index, so that both ETL pipelines produce the same artifact
format.

#### Acceptance Criteria

1. WHEN the PySpark ETL indexing stage runs, THE ETL_Pipeline SHALL build a `TurboQuantIndex`
   with `bit_width=4` in place of the `faiss.index_factory(d, "SQfp16", ...)` call; zero-sized
   indexes (when no vectors are available) SHALL be allowed to be built and written without error.
2. WHEN the PySpark ETL indexing stage runs, THE ETL_Pipeline SHALL NOT call `index.train()`
   because TurboVec requires no training step.
3. WHEN the PySpark ETL indexing stage completes, THE ETL_Pipeline SHALL write the index to
   `models/turbovec.tq` using `index.write(path)`.
4. WHEN the index is written, THE ETL_Pipeline SHALL update the Pipeline_Manifest `artifacts`
   map to record `"turbovec_index": "turbovec.tq"` instead of `"faiss_index": "faiss.index"`.
5. WHEN the index is written, THE ETL_Pipeline SHALL record `turbovec_index_size` (equal to
   `index.ntotal`) in the Serving_Contract section of the Pipeline_Manifest.
6. THE ETL_Pipeline SHALL NOT import or call any `faiss` module symbols in the indexing stage.

---

### Requirement 3: Update Serving_Contract Schema in Pipeline_Manifest

**User Story:** As a platform engineer, I want the Pipeline_Manifest to track the TurboVec
index artifact instead of the FAISS index, so that the model loader and health checks validate
the correct file.

#### Acceptance Criteria

1. WHEN either ETL pipeline writes the Pipeline_Manifest, THE ETL_Pipeline SHALL include
   `"turbovec_index_size"` in the `serving_contract` object with a value equal to `index.ntotal`.
2. WHEN either ETL pipeline writes the Pipeline_Manifest, THE ETL_Pipeline SHALL include
   `"turbovec_index"` in the `artifacts` map with value `"turbovec.tq"`.
3. WHERE a legacy Pipeline_Manifest contains only `"faiss_index_size"`, THE Model_Loader SHALL
   read `"faiss_index_size"` as a fallback for `"turbovec_index_size"` to preserve backward
   compatibility during the transition.
4. THE ETL_Pipeline SHALL NOT write `"faiss_index_size"` or `"faiss_index"` keys to newly
   generated Pipeline_Manifest files, including during any transitional state where both
   old and new artifact formats coexist on disk.
5. WHEN the Serving_Contract is validated, THE Model_Loader SHALL use the key
   `"turbovec_index_size"` (with `"faiss_index_size"` as a fallback) when checking the index
   row count.

---

### Requirement 4: Update Model Loader for TurboVec Artifact

**User Story:** As a platform engineer, I want the Model Loader to download, cache, and
validate the `turbovec.tq` artifact instead of `faiss.index`, so that serving environments
always have the correct index file available.

#### Acceptance Criteria

1. THE Model_Loader SHALL register `"turbovec.tq"` as a required artifact in `MODEL_FILES`
   with a configurable download URL via the `TURBOVEC_INDEX_URL` environment variable.
2. WHEN `turbovec.tq` is present locally and its row count matches `turbovec_index_size` in
   the Serving_Contract, THE Model_Loader SHALL NOT re-download the artifact.
3. WHEN `turbovec.tq` is present locally but its row count differs from the Serving_Contract,
   THE Model_Loader SHALL re-download `turbovec.tq`.
4. WHEN validating `turbovec.tq` row count, THE Model_Loader SHALL load the index via
   `TurboQuantIndex.load(path)` and compare `index.ntotal` against the manifest value.
5. WHERE the environment variable `TURBOVEC_INDEX_URL` is not set, THE Model_Loader SHALL
   log a warning and set the download result for `"turbovec.tq"` to `False`.
6. THE Model_Loader SHALL remove `"faiss.index"` from the `MODEL_FILES` registry so it is
   no longer downloaded or validated in new deployments.
7. WHEN the lite/low-memory serving profile is active, THE Model_Loader SHALL exclude
   `"turbovec.tq"` from the required artifact set, consistent with how `"faiss.index"` was
   previously excluded.

---

### Requirement 5: Update Recommender to Load and Search TurboVec Index

**User Story:** As a backend engineer, I want the Recommender to load the TurboVec index
at startup and use it for ANN search, so that recommendations are served from the new engine
with the same latency and recall profile as FAISS.

#### Acceptance Criteria

1. WHEN the Recommender loads vector artifacts, THE Recommender SHALL load the index from
   `models/turbovec.tq` using `TurboQuantIndex.load(path)`.
2. WHEN executing an ANN search, THE Recommender SHALL call `index.search(query_embedding, k)`
   where `query_embedding` is a 2-D float32 array of shape `(1, dim)` and `k` equals `fetch_k`.
3. WHEN `index.search` returns results, THE Recommender SHALL interpret the returned
   `(scores, indices)` tuple identically to the former FAISS `(distances, indices)` tuple.
4. WHEN `turbovec.tq` is absent and `faiss.index` is present on disk, THE Recommender SHALL
   log a warning and fall back to the metadata-only content retrieval path rather than
   attempting to load the FAISS file directly.
5. WHEN the TurboVec index is loaded, THE Recommender SHALL validate that `index.ntotal`
   equals the number of rows in the serving catalog before enabling dense retrieval.
6. IF `index.ntotal` does not equal the catalog row count, THEN THE Recommender SHALL call
   `_disable_vector_artifacts` with a descriptive reason string.
7. THE Recommender SHALL NOT import or reference `faiss` at serving time.

---

### Requirement 6: Update Multimodal Fusion Module

**User Story:** As a backend engineer, I want the Multimodal Fusion module to build and serve
a TurboVec-backed fused index instead of a FAISS flat index, so that multimodal retrieval
uses the consistent index engine.

#### Acceptance Criteria

1. WHEN `MultiModalFusionIndex.build_fusion_index()` runs, THE Multimodal_Fusion SHALL build
   a `TurboQuantIndex` with the fused vector dimensionality (1280) and `bit_width=4`.
2. WHEN the fusion index is built, THE Multimodal_Fusion SHALL add all fused vectors via
   `index.add(fused_vectors)` without a separate training step.
3. WHEN the fusion index is persisted, THE Multimodal_Fusion SHALL write it to
   `models/multimodal_turbovec.tq` using `index.write(path)`.
4. WHEN `MultiModalFusionIndex.load_fusion_index()` is called, THE Multimodal_Fusion SHALL
   load the index from `models/multimodal_turbovec.tq` using `TurboQuantIndex.load(path)`.
5. WHEN `MultiModalFusionIndex.search()` is called, THE Multimodal_Fusion SHALL call
   `index.search(fused_query, top_k)` and interpret the returned `(scores, indices)` tuple
   the same way the former FAISS search results were interpreted.
6. THE Multimodal_Fusion SHALL NOT directly import or reference `faiss`; indirect
   transitive dependencies through other libraries are permitted.

---

### Requirement 7: Update Two-Tower Training Script

**User Story:** As an ML engineer, I want the Two-Tower training script to export item
embeddings to a TurboVec index, so that the evaluation and candidate generation pipeline
is consistently built on the new engine.

#### Acceptance Criteria

1. WHEN `export_to_turbovec` (replacing `export_to_faiss`) is called, THE Two_Tower_Script
   SHALL build a `TurboQuantIndex` with the item embedding dimensionality and `bit_width=4`.
2. WHEN the index is built, THE Two_Tower_Script SHALL add all item embeddings via `index.add()`.
3. WHEN the index is persisted, THE Two_Tower_Script SHALL write it to
   `models/two_tower_turbovec.tq` using `index.write(path)`.
4. WHEN the script logs its summary, THE Two_Tower_Script SHALL log the TurboVec index path
   and `index.ntotal` row count.
5. THE Two_Tower_Script SHALL NOT import or reference `faiss`.

---

### Requirement 8: Backward Compatibility — Graceful Handling of Legacy faiss.index Files

**User Story:** As a DevOps engineer, I want the system to gracefully handle environments
where only the old `faiss.index` artifact exists on disk, so that live deployments are not
broken during the transition period before the new `turbovec.tq` artifact is deployed.

#### Acceptance Criteria

1. WHEN `turbovec.tq` is absent on disk and `faiss.index` is present, THE Recommender SHALL
   log a structured warning message that includes the string `"turbovec.tq not found"` and
   the path where it was expected.
2. WHEN `turbovec.tq` is absent, THE Recommender SHALL set `vector_artifacts_ready` to `False`
   in `_artifact_status` and SHALL serve all requests via the metadata-only fallback path.
3. WHEN `turbovec.tq` is absent, THE Recommender SHALL NOT attempt to load, parse, or import
   the `faiss.index` file.
4. THE System SHALL provide a one-time migration utility (`scripts/migrate_faiss_to_turbovec.py`)
   that reads an existing `faiss.index` file, extracts raw vectors, builds a `TurboQuantIndex`,
   and writes `turbovec.tq` to the same directory.
5. WHEN the migration utility runs, THE Migration_Script SHALL verify that the row count of
   the produced `turbovec.tq` matches the row count of the source `faiss.index`.
6. WHEN the migration utility completes, THE Migration_Script SHALL print a summary that
   includes source file size, output file size, and row count.

---

### Requirement 9: Preserve Serving Recall — No Quality Regression

**User Story:** As a product manager, I want the recommendation quality (recall) to not
regress after the FAISS-to-TurboVec migration, so that users continue to receive relevant
recommendations.

#### Acceptance Criteria

1. WHEN a TurboVec index is built with `bit_width=4`, THE TurboVec_Index SHALL achieve
   Recall@10 of at least 0.92 on the SBERT embedding corpus (768-dimensional vectors),
   measured against brute-force inner product search on a 1,000-query sample.
2. WHEN a TurboVec index is built with `bit_width=4`, THE TurboVec_Index SHALL achieve
   Recall@50 of at least 0.97 on the same corpus and query sample.
3. THE System SHALL include a `scripts/evaluate_turbovec_recall.py` script that computes
   Recall@K for the production index and logs the results, allowing CI/CD to gate deployments.
4. WHEN the recall evaluation script runs, THE System SHALL compare TurboVec recall against
   a brute-force numpy inner product baseline and SHALL exit with a non-zero status code if
   Recall@10 falls below 0.90.

---

### Requirement 10: Update Tests — Replace FAISS Mock with TurboVec Mock

**User Story:** As a software engineer, I want the existing property-based and integration tests
to use a TurboVec-compatible mock index instead of the FAISS mock, so that tests continue to
validate retrieval pipeline correctness after the migration.

#### Acceptance Criteria

1. WHEN property-based tests construct a mock ANN index, THE Test_Suite SHALL use a mock
   object that exposes the `search(query, k) -> (scores, indices)` interface matching
   `TurboQuantIndex`, replacing the current `MockFaissIndex`.
2. WHEN the mock index's `search` method is called, THE Mock_Index SHALL return a tuple of
   `(scores_array, indices_array)` with shapes `(1, k)` consistent with both TurboVec's
   and the expected retrieval pipeline output format; either a TurboVec mock or a FAISS-style
   mock is acceptable as long as the returned tuple satisfies this shape contract.
3. THE Test_Suite SHALL NOT directly import `faiss` in any test file; transitive dependencies
   through other libraries are permitted.
4. WHEN integration tests create a local test index for the recommender, THE Test_Suite SHALL
   build the index using `TurboQuantIndex` and persist it as `turbovec.tq` in the test's
   temporary directory.
5. WHEN test Pipeline_Manifests are created, THE Test_Suite SHALL use `"turbovec_index_size"`
   as the manifest key and `"turbovec.tq"` as the artifact file name.
6. FOR ALL property-based tests that generate arbitrary retrieval results, the `retrieval_source`
   field SHALL include `"turbovec"` as a valid source label in the sampled set, replacing or
   supplementing `"faiss"`.

---

### Requirement 11: Update CI/CD and Artifact Validation Scripts

**User Story:** As a DevOps engineer, I want all CI/CD workflows and artifact validation
scripts to reference `turbovec.tq` instead of `faiss.index`, so that the deployment pipeline
is consistent with the new artifact format.

#### Acceptance Criteria

1. WHEN `scripts/validate_serving_artifacts.py` runs, THE Validation_Script SHALL check for
   the presence and non-zero size of `turbovec.tq` instead of `faiss.index`.
2. WHEN `scripts/validate_serving_artifacts.py` validates the manifest contract, THE
   Validation_Script SHALL read `"turbovec_index_size"` (with `"faiss_index_size"` as fallback)
   as the expected row count key.
3. WHEN `scripts/backfill_serving_metadata_artifacts.py` runs, THE Backfill_Script SHALL
   record `"turbovec_index"` pointing to `"turbovec.tq"` in the artifact map.
4. WHEN `manage.py` lists required serving artifacts, THE Manage_Script SHALL include
   `"models/turbovec.tq"` and SHALL NOT list `"models/faiss.index"`.
5. WHEN `scripts/rebuild_serving_artifacts.py` runs, THE Rebuild_Script SHALL write the
   vector index to `models/turbovec.tq` using the TurboVec write API.
