# Implementation Plan: Architecture Design Perfection

## Overview

This plan decomposes the five structural improvements into four ordered tracks. Track 1 (documentation) is zero-risk and runs first. Track 2 (ablation script) is self-contained. Track 3 (pipeline decomposition) is the highest-impact work and must be executed in strict dependency order. Track 4 (main.py reduction) follows once the pipeline modules are stable. Property-based tests using Hypothesis validate the universal invariants of the three pipeline stages.

## Tasks

- [x] 1. Track 1 — Documentation Artifacts
  - [x] 1.1 Create `docs/ARCHITECTURE_DECISIONS.md` with all 6 ADRs
    - Create the file with a table of contents linking to each ADR by number and title
    - Write ADR-001: LightGCN as Primary Ensemble Component (weight: 0.65) — Context → Decision → Consequences
    - Write ADR-002: Quantum-Fluid Neural ODE for Temporal Preference Drift (weight: 0.25)
    - Write ADR-003: SASRec for Session-Level Sequential Patterns (weight: 0.10)
    - Write ADR-004: KAN, Hyperbolic, and Diffusion at Zero Weight — Retained for Conditional Activation; include the specific signal each zero-weight model contributes (KAN: B-spline non-linear feature interactions; Hyperbolic: hierarchical genre embeddings; Diffusion: generative diversity)
    - Write ADR-005: 3-Tier Serving Architecture with Hardware Auto-Detection — document the thresholds (GPU + ≥16 GB RAM → Tier1; no GPU + ≥8 GB RAM → Tier2; <8 GB RAM → Tier3)
    - Write ADR-006: Pipeline Decomposition — Monolith → Retrieval/Ranking/Reranking; document tradeoffs of the split
    - Include a "Superseded By" field (empty) in each ADR entry to support future decision evolution
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 1.2 Add Mermaid architecture diagram to `README.md`
    - Insert a new "## Architecture Diagram" section into `README.md` without modifying any existing content or badge links
    - Write a `flowchart TD` diagram with three subgraphs: Serving Path, Data Pipeline, Training Pipeline
    - Serving Path subgraph: `UserRequest → FastAPI → TierDetector → Tier1/Tier2/Tier3 → RetrievalPipeline → RankingPipeline → RerankingPipeline → Response`
    - Label Tier1 (GPU/full ensemble), Tier2 (ONNX CPU), Tier3 (FAISS + TF-IDF only)
    - Show FAISS ANN Index, TF-IDF Sparse Index, and Knowledge Graph as distinct labeled nodes feeding into RetrievalPipeline
    - Show all 6 ensemble models as sub-components of the RankingPipeline node
    - Data Pipeline subgraph: `TMDB API + Kaggle → ETL → Delta Lake Bronze → Silver → Gold → Model Training → Serving Artifacts`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

- [x] 2. Track 2 — Ablation Evidence Script
  - [x] 2.1 Create `backend/pipeline_types.py` with shared dataclasses
    - Define `CandidateItem` dataclass: `movie_id: int`, `retrieval_score: float`, `retrieval_source: Literal["faiss", "tfidf", "knowledge_graph", "hybrid"]`, `metadata: dict`
    - Define `RankedItem` dataclass: all `CandidateItem` fields plus `ensemble_score: float`, `ranker_score: float`, `final_rank: int`, `retrieval_signals: dict`, `metadata: dict`
    - Define `FinalItem` dataclass: all `RankedItem` fields plus `diversity_score: float`, `safety_passed: bool`, `explanation: str | None`
    - This module MUST have zero imports from other `backend/` modules to prevent circular imports
    - Use `from __future__ import annotations` and `from dataclasses import dataclass, field`
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 2.2 Create `scripts/ablation_study.py`
    - Define `ModelAblationResult` dataclass: `model: str`, `ndcg_without: float | None`, `delta: float | None`, `marginal_contribution_pct: float | None`
    - Define `AblationReport` dataclass: `run_timestamp: str` (ISO 8601), `full_ensemble_ndcg: float`, `models: list[ModelAblationResult]`
    - Implement `AblationStudy` class with `__init__(self, recommender, sample_size: int = 1000)`
    - Implement `run_full_ensemble(self) -> float` — evaluate NDCG@10 with all 6 models active
    - Implement `run_leave_one_out(self, model_name: str) -> float | None` — evaluate NDCG@10 with one model removed; return `None` (not `0.0`) if model fails to load, log a warning
    - Implement `run_all(self) -> AblationReport` — run full ensemble + 6 leave-one-out evaluations for LightGCN, Quantum, SASRec, KAN, Hyperbolic, Diffusion
    - Implement `print_table(self, report: AblationReport) -> None` — print formatted table: model name, NDCG@10 without, delta, marginal contribution %
    - Implement `save_report(self, report: AblationReport, output_path: Path) -> None` — serialize to JSON; create parent directory if it does not exist
    - Add CLI entry point with `argparse`: `--sample-size` (default: 1000), `--output` (default: `reports/ablation_report.json`)
    - Script must be runnable as `python scripts/ablation_study.py` from repo root
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [ ]* 2.3 Write property test for ablation report serialization round-trip
    - **Property 11: Ablation Report Serialization Round-Trip**
    - For any `AblationReport` instance, serializing via `save_report()` and deserializing SHALL produce an identical report
    - Use `@given` with `st.builds(AblationReport, ...)` generating arbitrary timestamps, NDCG values, and per-model results including `None` values
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 11: Ablation Report Serialization Round-Trip`
    - **Validates: Requirements 3.4**

- [x] 3. Checkpoint — Track 1 and Track 2 complete
  - Ensure `docs/ARCHITECTURE_DECISIONS.md` exists with all 6 ADRs and a working table of contents
  - Ensure `README.md` Mermaid diagram renders correctly (check syntax)
  - Ensure `scripts/ablation_study.py` runs with `python scripts/ablation_study.py --sample-size 10` without error
  - Ensure `backend/pipeline_types.py` has no imports from other `backend/` modules
  - Ask the user if questions arise before proceeding to Track 3

- [x] 4. Track 3 — Pipeline Decomposition (execute in strict order)
  - [x] 4.1 Create `backend/artifact_validator.py`
    - Implement `ArtifactValidator` class with `__init__(self, manifest_path: Path)`
    - Implement `_load_manifest(self, manifest_path: Path) -> dict` — load JSON manifest mapping artifact names to expected SHA-256 checksums
    - Implement `validate(self, artifact_path: Path) -> bool`:
      - Raise `FileNotFoundError` if file does not exist
      - Compute SHA-256 checksum of file contents
      - Compare against manifest entry; raise `ValueError` with message `"Checksum mismatch for {path}: expected {exp}, got {actual}"` on mismatch
      - Return `True` on success
    - Implement `validate_row_alignment(self, embeddings: np.ndarray, movie_df: pd.DataFrame) -> bool` — assert `embeddings.shape[0] == len(movie_df)`; raise `ValueError` on mismatch
    - Implement `validate_all(self) -> dict[str, bool]` — validate all artifacts in manifest; return `{artifact_name: bool}`
    - _Requirements: 4.4, 4.12_

  - [ ]* 4.2 Write property test for artifact validator idempotence
    - **Property 10: Artifact Validator Idempotence**
    - For any unmodified artifact file, calling `validate()` twice SHALL return the same boolean result
    - Create a temporary file with known content and checksum in the manifest; call `validate()` twice and assert results are equal
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 10: Artifact Validator Idempotence`
    - **Validates: Requirements 6.12**

  - [x] 4.3 Create `backend/retrieval_pipeline.py`
    - Import `CandidateItem` from `backend.pipeline_types` (no other backend imports at module level)
    - Define `RetrievalConfig` dataclass: `faiss_k: int = 100`, `tfidf_k: int = 50`, `kg_k: int = 20`, `low_memory: bool = False`, `enable_kg: bool = True`
    - Implement `RetrievalPipeline.__init__(self, faiss_index, tfidf_index, kg_engine, movie_df, config: RetrievalConfig)`
    - Implement `retrieve(self, query_embedding: np.ndarray, n: int) -> list[CandidateItem]`:
      - Return `[]` immediately when `n == 0`
      - Query FAISS ANN index → top-`faiss_k` candidates tagged `retrieval_source="faiss"`
      - Query TF-IDF sparse index → additional candidates tagged `retrieval_source="tfidf"` (skip if `low_memory=True`)
      - Query KG engine → additional candidates tagged `retrieval_source="knowledge_graph"` (skip if `kg_engine` is None or `enable_kg=False`)
      - Deduplicate by `movie_id` using max-pool on `retrieval_score`; tag merged items `retrieval_source="hybrid"`
      - Sort descending by `retrieval_score`; return top-`n` items
      - Enforce: `len(result) <= n`, all `movie_id` values unique, all `retrieval_source` values valid
      - Fall back to TF-IDF only if FAISS unavailable; return `[]` if all sources unavailable
    - _Requirements: 4.1, 4.5, 6.1, 6.2, 6.8, 6.10_

  - [ ]* 4.4 Write property tests for retrieval pipeline invariants
    - **Property 1: Retrieval Bounds Guarantee** — `1 <= len(result) <= n` for non-empty catalog and `n >= 1`
    - **Property 2: Retrieval Deduplication Invariant** — all `movie_id` values in result are unique
    - **Property 3: Retrieval Source Tagging** — every `CandidateItem.retrieval_source` is in `{"faiss", "tfidf", "knowledge_graph", "hybrid"}`
    - Use `@given` with `st.integers(min_value=1, max_value=200)` for `n` and mock catalog data
    - Use `@settings(max_examples=100)` on each test
    - Tags: `# Feature: architecture-design-perfection, Property 1/2/3`
    - **Validates: Requirements 6.1, 6.2, 6.8, 4.5**

  - [x] 4.5 Create `backend/ranking_pipeline.py`
    - Import `CandidateItem`, `RankedItem` from `backend.pipeline_types`
    - Define `RankingConfig` dataclass: `ensemble_weight: float = 0.7`, `ranker_weight: float = 0.3`, `use_neural_ensemble: bool = True`, `use_learned_ranker: bool = True`
    - Implement `RankingPipeline.__init__(self, ensemble_engine, learned_ranker, config: RankingConfig)`
    - Implement `rank(self, candidates: list[CandidateItem], user_context: dict) -> list[RankedItem]`:
      - Get ensemble scores for all candidates (skip if `use_neural_ensemble=False`; fall back to `retrieval_score` if ensemble engine unavailable)
      - Apply learned ranker (MMoE or LightGBM) if `use_learned_ranker=True`; use ensemble score only if ranker unavailable (`ranker_weight → 0`)
      - Blend: `ensemble_weight * ensemble_score + ranker_weight * ranker_score`
      - Sort descending by blended score
      - Assign `final_rank` (1-indexed, 1 = best)
      - Return `list[RankedItem]` of exactly `len(candidates)` items
      - Enforce: `len(result) == len(candidates)`, `set(movie_ids)` unchanged, deterministic output for identical inputs
    - _Requirements: 4.2, 4.6, 6.3, 6.4, 6.7, 6.11_

  - [ ]* 4.6 Write property tests for ranking pipeline invariants
    - **Property 4: Ranking Count Preservation** — `len(result) == len(candidates)` for any input list
    - **Property 5: Ranking Set-Identity Round-Trip** — `{r.movie_id for r in result} == {c.movie_id for c in candidates}`
    - **Property 6: Ranking Ordering Invariant** — result is sorted descending by blended score
    - **Property 7: Ranking Determinism** — calling `rank()` twice with identical inputs produces identical scores
    - Use `@given(candidates=st.lists(st.builds(CandidateItem, ...), min_size=0, max_size=200, unique_by=lambda c: c.movie_id))`
    - Use `@settings(max_examples=100)` on each test
    - Tags: `# Feature: architecture-design-perfection, Property 4/5/6/7`
    - **Validates: Requirements 6.3, 6.4, 6.7, 6.11**

  - [x] 4.7 Create `backend/reranking_pipeline.py`
    - Import `RankedItem`, `FinalItem` from `backend.pipeline_types`
    - Define `RerankingConfig` dataclass: `mmr_lambda: float = 0.7`, `enable_llm_reranking: bool = False`, `enable_rl_safety: bool = True`, `quality_threshold: float = 0.3`
    - Implement `RerankingPipeline.__init__(self, rl_policy, llm_client, config: RerankingConfig)`
    - Implement `rerank(self, ranked_items: list[RankedItem], constraints: dict) -> list[FinalItem]`:
      - Return `[]` immediately when `ranked_items` is empty
      - Apply RL safety filter (remove items in user's dislike list from `constraints`)
      - Apply quality gate (filter items below `quality_threshold`)
      - Apply MMR diversity (greedy selection with `lambda=mmr_lambda`)
      - Optionally apply LLM reranking if `enable_llm_reranking=True`; skip gracefully if LLM client unavailable
      - Return `list[FinalItem]` where all `movie_id` values are a subset of input `ranked_items`
      - Enforce: no hallucinated items, deterministic output for identical inputs
    - _Requirements: 4.3, 4.7, 6.5, 6.6, 6.9_

  - [ ]* 4.8 Write property tests for reranking pipeline invariants
    - **Property 8: Reranking No-Hallucination** — `{f.movie_id for f in result} ⊆ {r.movie_id for r in ranked_items}`
    - **Property 9: Reranking Determinism** — calling `rerank()` twice with identical inputs produces identical ordered lists
    - Also test empty-input safety: `rerank([], {})` returns `[]` without exception
    - Use `@given` with `st.lists(st.builds(RankedItem, ...), min_size=0, max_size=100, unique_by=lambda r: r.movie_id)`
    - Use `@settings(max_examples=100)` on each test
    - Tags: `# Feature: architecture-design-perfection, Property 8/9`
    - **Validates: Requirements 6.5, 6.6, 6.9**

  - [x] 4.9 Refactor `backend/recommender.py` to thin orchestrator (<600 lines)
    - Extract all retrieval logic (FAISS search, TF-IDF search, KG traversal) into `RetrievalPipeline` — delete from `recommender.py`
    - Extract all ranking logic (ensemble scoring, MMoE/LightGBM ranker application, score blending) into `RankingPipeline` — delete from `recommender.py`
    - Extract all reranking logic (MMR diversity, RL safety filter, LLM reranking, quality gate) into `RerankingPipeline` — delete from `recommender.py`
    - Extract all artifact integrity checks into `ArtifactValidator` — delete from `recommender.py`
    - Rewrite `Recommender.load()` to: detect serving tier, validate artifacts via `ArtifactValidator`, load `movie_df`, initialize `RetrievalPipeline` / `RankingPipeline` / `RerankingPipeline` with tier-aware configs
    - Tier-aware config: Tier3 → `use_neural_ensemble=False`, `use_learned_ranker=False`, `low_memory=True`, `enable_kg=False`; override if `NOVA_SERVING_PROFILE=full`
    - Tier1/Tier2 → `use_neural_ensemble=True`, `use_learned_ranker=True`, `low_memory=False`
    - Preserve ALL existing public method signatures: `recommend_by_id`, `recommend_by_index`, `search_movies`, `semantic_search`, `kg_recommend`, `visual_search`, `get_movie_by_id`, `get_all_titles`, `recommend_for_user_profile`
    - Delegate each recommendation request through `retrieve → rank → rerank` sequence
    - Verify final line count is under 600
    - _Requirements: 4.8, 4.9, 4.10, 4.11_

- [x] 5. Checkpoint — Track 3 complete
  - Verify `backend/recommender.py` is under 600 lines
  - Verify all four new modules exist: `pipeline_types.py`, `artifact_validator.py`, `retrieval_pipeline.py`, `ranking_pipeline.py`, `reranking_pipeline.py`
  - Run existing test suite: `pytest backend/tests/` — all tests must pass
  - Ask the user if questions arise before proceeding to Track 4

- [x] 6. Track 4 — main.py Reduction
  - [x] 6.1 Create `backend/cache.py` — extract `AsyncLRUCache`
    - Move the `AsyncLRUCache` class (currently in `main.py`) into a new `backend/cache.py` module
    - Update `main.py` to import `AsyncLRUCache` from `backend.serving.cache`
    - Verify no other callers are broken
    - _Requirements: 5.1, 5.2, 5.4_

  - [x] 6.2 Create `backend/app_info.py` — extract app metadata helpers
    - Move `app_metadata()` and `public_base_url()` functions from `main.py` into a new `backend/app_info.py` module
    - Update `main.py` to import these functions from `backend.serving.app_info`
    - _Requirements: 5.1, 5.2_

  - [x] 6.3 Consolidate diagnostic and readiness helpers into existing modules
    - Move `_recommendation_diagnostic_report()` and `_readiness_component()` from `main.py` into `backend/recommendation_routes.py` (already exists at ~990 lines — consolidate, do not create a new file)
    - Move `_serving_lineage()` and `_candidate_event_summary()` from `main.py` into `backend/recommendation_events.py` (already exists — extend it)
    - Move `_benchmark_readiness_component()` from `main.py` into `backend/platform_readiness.py` (already exists at ~257 lines — extend it)
    - Update all import sites in `main.py` accordingly
    - _Requirements: 5.1, 5.2, 5.4_

  - [x] 6.4 Verify `main.py` line count and run full test suite
    - Count lines in `backend/main.py` and assert the count is under 800
    - Run `pytest backend/tests/` and confirm all existing tests pass
    - Verify all existing API endpoint paths, HTTP methods, request/response schemas, and authentication middleware are unchanged
    - _Requirements: 5.1, 5.3, 5.5_

- [x] 7. Final Checkpoint — All tracks complete
  - Verify `docs/ARCHITECTURE_DECISIONS.md` exists with 6 ADRs and table of contents
  - Verify `README.md` contains a Mermaid diagram under "Architecture Diagram" section
  - Verify `scripts/ablation_study.py` is executable and writes valid JSON to `reports/ablation_report.json`
  - Verify `backend/pipeline_types.py`, `artifact_validator.py`, `retrieval_pipeline.py`, `ranking_pipeline.py`, `reranking_pipeline.py` all exist
  - Verify `backend/recommender.py` is under 600 lines
  - Verify `backend/main.py` is under 800 lines
  - Run `pytest backend/tests/` — all tests must pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Track 3 tasks (4.1–4.9) MUST be executed in strict order — each module depends on the previous
- `pipeline_types.py` MUST have zero imports from other `backend/` modules (circular import prevention)
- All existing public API contracts of `Recommender` must be preserved — no breaking changes to `main.py` callers
- Property tests use Hypothesis `@settings(max_examples=100)` as specified in the design
- Each property test is tagged with `# Feature: architecture-design-perfection, Property N: <text>`
- The `reports/` directory does not yet exist — `ablation_study.py` must create it
- Tier3 degradation behavior (no neural ensemble, no KG) must be preserved exactly as in the current `recommender.py`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "2.1"] },
    { "id": 1, "tasks": ["2.2", "4.1"] },
    { "id": 2, "tasks": ["2.3", "4.2", "4.3"] },
    { "id": 3, "tasks": ["4.4", "4.5"] },
    { "id": 4, "tasks": ["4.6", "4.7"] },
    { "id": 5, "tasks": ["4.8", "4.9"] },
    { "id": 6, "tasks": ["6.1", "6.2", "6.3"] },
    { "id": 7, "tasks": ["6.4"] }
  ]
}
```
