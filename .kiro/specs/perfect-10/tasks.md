# Implementation Plan: Perfect 10

## Overview

Close the four remaining gaps that prevent a 10/10 rating. All infrastructure, pipeline modules,
frontend pages, CI workflows, and documentation are already complete. The work here is surgical:

- **Track 1** — Slim `recommender.py` from 2118 → <600 lines by deleting the old monolithic
  implementations that duplicate the already-wired pipeline modules.
- **Track 2** — Slim `main.py` from 1009 → <800 lines by extracting the inline readiness-report
  builder into `backend/platform_readiness.py` (which already exists).
- **Track 3** — Add the four missing pipeline property-based tests from the
  `architecture-design-perfection` spec (Tasks 4.4, 4.6, 4.8, 2.3).
- **Track 4** — Verify the full test suite passes after the refactors.

Tracks 1 and 2 are independent and can run in parallel. Track 3 is independent of both.
Track 4 must run last.

---

## Tasks

### Track 1 — Slim `recommender.py` to <600 lines

- [x] 1. Delete monolithic `recommend_by_index` body and delegate to pipelines
  - [x] 1.1 Replace the 379-line `recommend_by_index` implementation with a pipeline delegation
    - Read `backend/recommender.py` lines 1689–2067 (the full `recommend_by_index` body)
    - Replace the entire method body with a delegation to `self._retrieval_pipeline`,
      `self._ranking_pipeline`, and `self._reranking_pipeline`
    - When any pipeline is `None` (tier3 / cold start), fall back to
      `self._metadata_recommend_by_index(movie_idx, n=n)` — preserve existing fallback
    - The delegation pattern:
      ```python
      def recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
          if self._retrieval_pipeline is None or self._ranking_pipeline is None:
              return self._metadata_recommend_by_index(movie_idx, n=n)
          query_vector = self._vectors[movie_idx].reshape(1, -1).astype(np.float32)
          candidates = self._retrieval_pipeline.retrieve(query_vector, n=min(100, len(self._movies)))
          ranked = self._ranking_pipeline.rank(candidates, user_context={})
          final = self._reranking_pipeline.rerank(ranked, constraints={})
          return [self._candidate_to_dict(item) for item in final[:n]]
      ```
    - Add `_candidate_to_dict(item) -> dict` helper that converts a `FinalItem` to the
      existing response dict shape (copy the field-mapping logic from the old implementation)
    - _Requirements: recommender.py < 600 lines, all public method signatures preserved_

  - [x] 1.2 Delete monolithic `recommend_for_user_profile` body and delegate to pipelines
    - The 193-line `recommend_for_user_profile` (lines 1034–1226) builds candidates from
      genre affinity, behavior boost, and FAISS — replace with pipeline delegation
    - When pipelines are `None`, keep the existing metadata-only fallback path
    - Preserve the public signature: `recommend_for_user_profile(self, profile: dict, n: int = 10) -> list[dict]`
    - _Requirements: recommender.py < 600 lines_

  - [x] 1.3 Delete monolithic `search_movies` body and delegate to `RetrievalPipeline`
    - The 136-line `search_movies` (lines 1294–1429) does TF-IDF + FAISS + cross-encoder reranking
    - `RetrievalPipeline.retrieve()` already handles this — delegate to it
    - Preserve the public signature: `search_movies(self, query: str, limit: int = 20) -> list[dict]`
    - When `_retrieval_pipeline` is `None`, keep the existing sparse-only fallback
    - _Requirements: recommender.py < 600 lines_

  - [x] 1.4 Delete monolithic `ai_search` body and delegate to `RetrievalPipeline`
    - The 127-line `ai_search` (lines 2330–2456) does SBERT encoding + FAISS + MMR
    - Delegate to `self._retrieval_pipeline.retrieve()` with the SBERT-encoded query vector
    - Preserve the public signature: `ai_search(self, query: str, n: int = 10, fetch_k: int = 80) -> list[dict]`
    - _Requirements: recommender.py < 600 lines_

  - [x] 1.5 Delete `_rerank_with_llm` and `_apply_mmr` from `recommender.py`
    - These methods (lines 2068–2190, 66 + 57 lines) are now implemented inside
      `RerankingPipeline` — remove the duplicates from `recommender.py`
    - Verify no other method in `recommender.py` calls `self._rerank_with_llm` or
      `self._apply_mmr` directly (they should all go through `_reranking_pipeline.rerank()`)
    - _Requirements: recommender.py < 600 lines_

  - [x] 1.6 Delete `_validate_vector_artifacts` body and delegate to `ArtifactValidator`
    - The 113-line `_validate_vector_artifacts` (lines 593–705) duplicates `ArtifactValidator`
    - Replace with: `ArtifactValidator(manifest_path).validate_row_alignment(self._vectors, self._movies)`
    - Keep the existing `try/except` wrapper and `_disable_vector_artifacts` fallback
    - _Requirements: recommender.py < 600 lines_

  - [x] 1.7 Verify `recommender.py` is under 600 lines
    - Run: `python -c "print(sum(1 for _ in open('backend/recommender.py')))"`
    - If still over 600, identify the next largest extractable block (check `recommend_batch`,
      `recommend_by_title`) and move them to a `backend/recommender_batch.py` helper module
    - All existing public method signatures must remain unchanged
    - _Requirements: recommender.py < 600 lines_

---

### Track 2 — Slim `main.py` to <800 lines

- [x] 2. Extract inline `_platform_readiness_report` from `main.py` into `platform_readiness.py`
  - [x] 2.1 Move `_platform_readiness_report` and `_combine_readiness_status` to `backend/platform_readiness.py`
    - `backend/platform_readiness.py` already exists — append these two functions to it
    - `_platform_readiness_report` (lines 650–831, ~182 lines) and
      `_combine_readiness_status` (lines 636–649, ~14 lines) are currently defined inline in `main.py`
    - Move both functions; update all imports in `main.py` to import them from
      `backend.serving.platform_readiness`
    - The function signatures must not change — callers in `main.py` must work without modification
    - _Requirements: main.py < 800 lines_

  - [x] 2.2 Extract `record_recommendation_events` and `remote_payload_or_raise` to `backend/recommendation_events.py`
    - `backend/recommendation_events.py` already exists — append these two functions to it
    - `record_recommendation_events` (lines 832–900, ~69 lines) and
      `remote_payload_or_raise` (lines 901–920, ~20 lines) are currently inline in `main.py`
    - Move both; update `main.py` to import them from `backend.events.recommendation_events`
    - _Requirements: main.py < 800 lines_

  - [x] 2.3 Verify `main.py` is under 800 lines
    - Run: `python -c "print(sum(1 for _ in open('backend/main.py')))"`
    - If still over 800, identify the next largest inline block and extract it
    - _Requirements: main.py < 800 lines_

---

### Track 3 — Missing pipeline property-based tests

- [x] 3. Create `tests/test_retrieval_pipeline_properties.py`
  - [x] 3.1 Implement Property 1 — Retrieval Bounds Guarantee
    - Use `@given(st.integers(min_value=1, max_value=200))` for `n`
    - Mock a minimal FAISS index and TF-IDF index returning `n` candidates
    - Assert `1 <= len(result) <= n` for any non-empty catalog and `n >= 1`
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 1: Retrieval Bounds Guarantee`
    - _Requirements: retrieval pipeline invariants_

  - [x] 3.2 Implement Property 2 — Retrieval Deduplication Invariant
    - Use `@given(st.integers(min_value=1, max_value=50))` for `n`
    - Construct mock FAISS + TF-IDF sources that return overlapping `movie_id` sets
    - Assert all `movie_id` values in result are unique
    - Tag: `# Feature: architecture-design-perfection, Property 2: Retrieval Deduplication Invariant`
    - _Requirements: retrieval pipeline invariants_

  - [x] 3.3 Implement Property 3 — Retrieval Source Tagging
    - Assert every `CandidateItem.retrieval_source` in result is in
      `{"faiss", "tfidf", "knowledge_graph", "hybrid"}`
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 3: Retrieval Source Tagging`
    - _Requirements: retrieval pipeline invariants_

- [x] 4. Create `tests/test_ranking_pipeline_properties.py`
  - [x] 4.1 Implement Property 4 — Ranking Count Preservation
    - Use `@given(st.lists(st.builds(CandidateItem, ...), min_size=0, max_size=200, unique_by=lambda c: c.movie_id))`
    - Assert `len(result) == len(candidates)` for any input list
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 4: Ranking Count Preservation`
    - _Requirements: ranking pipeline invariants_

  - [x] 4.2 Implement Property 5 — Ranking Set-Identity Round-Trip
    - Assert `{r.movie_id for r in result} == {c.movie_id for c in candidates}`
    - Tag: `# Feature: architecture-design-perfection, Property 5: Ranking Set-Identity Round-Trip`
    - _Requirements: ranking pipeline invariants_

  - [x] 4.3 Implement Property 6 — Ranking Ordering Invariant
    - Assert result is sorted descending by blended score (no two adjacent items out of order)
    - Tag: `# Feature: architecture-design-perfection, Property 6: Ranking Ordering Invariant`
    - _Requirements: ranking pipeline invariants_

  - [x] 4.4 Implement Property 7 — Ranking Determinism
    - Call `rank()` twice with identical inputs; assert scores are identical
    - Tag: `# Feature: architecture-design-perfection, Property 7: Ranking Determinism`
    - _Requirements: ranking pipeline invariants_

- [x] 5. Create `tests/test_reranking_pipeline_properties.py`
  - [x] 5.1 Implement Property 8 — Reranking No-Hallucination
    - Use `@given(st.lists(st.builds(RankedItem, ...), min_size=0, max_size=100, unique_by=lambda r: r.movie_id))`
    - Assert `{f.movie_id for f in result} ⊆ {r.movie_id for r in ranked_items}`
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 8: Reranking No-Hallucination`
    - _Requirements: reranking pipeline invariants_

  - [x] 5.2 Implement Property 9 — Reranking Determinism
    - Call `rerank()` twice with identical inputs; assert identical ordered lists
    - Also test empty-input safety: `rerank([], {})` returns `[]` without exception
    - Tag: `# Feature: architecture-design-perfection, Property 9: Reranking Determinism`
    - _Requirements: reranking pipeline invariants_

- [x] 6. Create `tests/test_ablation_serialization_property.py`
  - [x] 6.1 Implement Property 11 — Ablation Report Serialization Round-Trip
    - Import `AblationReport`, `ModelAblationResult` from `scripts.ablation_study`
    - Use `@given(st.builds(AblationReport, ...))` generating arbitrary timestamps, NDCG values,
      and per-model results including `None` values for `ndcg_without` and `delta`
    - Serialize via `save_report()` to a `tmp_path` file; deserialize with `json.load`
    - Assert all numeric fields round-trip to within `1e-9` tolerance
    - Use `@settings(max_examples=100)`
    - Tag: `# Feature: architecture-design-perfection, Property 11: Ablation Report Serialization Round-Trip`
    - _Requirements: ablation report correctness_

- [x] 7. Register new test files in `ci.yml`
  - [x] 7.1 Add the three new test files to the `unit-tests` job in `.github/workflows/ci.yml`
    - Add `tests/test_retrieval_pipeline_properties.py`,
      `tests/test_ranking_pipeline_properties.py`,
      `tests/test_reranking_pipeline_properties.py`,
      `tests/test_ablation_serialization_property.py`
      to the existing `python -m pytest` command in the `unit-tests` job
    - Preserve the existing `--cov-fail-under=80` and `-x` flags
    - _Requirements: all PBTs run in CI_

---

### Track 4 — Final verification

- [x] 8. Run full test suite and verify all gates pass
  - [x] 8.1 Run backend tests with coverage
    - Run: `pytest tests/ backend/tests/ -v --tb=short -q --cov=backend --cov-fail-under=80`
    - All tests must pass; fix any import errors introduced by the Track 1/2 refactors
    - _Requirements: 80% coverage gate, no regressions_

  - [x] 8.2 Run frontend tests with coverage
    - Run: `cd frontend && npm run test -- --coverage`
    - Coverage threshold (80% lines) must pass per `vite.config.ts`
    - _Requirements: frontend 80% coverage gate_

  - [x] 8.3 Verify final line counts
    - `backend/recommender.py` must be < 600 lines
    - `backend/main.py` must be < 800 lines
    - _Requirements: code organization targets_

---

## Notes

- Track 1 is the highest-risk track — `recommend_by_index` is the core hot path. After each
  sub-task, run `pytest tests/test_api.py -x -q` to catch regressions immediately.
- The pipeline modules (`retrieval_pipeline.py`, `ranking_pipeline.py`, `reranking_pipeline.py`)
  are fully implemented and tested — the delegation in Track 1 is purely mechanical.
- `_metadata_recommend_by_index` must be preserved as the tier3 / no-vector fallback.
- Track 2 is low-risk — it's pure function movement with import updates.
- Track 3 property tests use `CandidateItem`, `RankedItem`, `FinalItem` from
  `backend.pipeline_types` — import them directly, no mocking needed for the dataclasses.
- For Track 3 Property tests, mock the FAISS index with a simple `numpy` array lookup to avoid
  requiring actual model artifacts in CI.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1", "4.1", "5.1", "6.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "3.2", "4.2", "5.2"] },
    { "id": 2, "tasks": ["1.3", "3.3", "4.3"] },
    { "id": 3, "tasks": ["1.4", "4.4"] },
    { "id": 4, "tasks": ["1.5", "1.6", "2.3"] },
    { "id": 5, "tasks": ["1.7", "7.1"] },
    { "id": 6, "tasks": ["8.1", "8.2"] },
    { "id": 7, "tasks": ["8.3"] }
  ]
}
```
