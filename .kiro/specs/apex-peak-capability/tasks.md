# Implementation Plan: APEX Peak Capability

## Overview

Close the five capability gaps in the APEX recommendation system in dependency order:
session sequences first (SASRec needs real data before ensemble weights matter), then
ensemble weight loading (needed before the weight optimizer script is useful), then
online learning (independent background module), then Two-Tower fine-tuning (standalone
script + loader hook), and finally RL + Active Inference wiring (depends on recommender
and main.py being stable). Property-based tests are placed immediately after the code
they validate so regressions are caught early.

---

## Tasks

- [x] 1. Add session-sequence builder to `ensemble_engine.py`
  - [x] 1.1 Extend `ApexEnsembleEngine.__init__` to initialise `_session_cache`
    - Add `self._session_cache: dict[str, tuple[float, list[int]]] = {}` to `__init__`
    - Add `import time` at the top of the file (if not already present)
    - _Requirements: 1.5_

  - [x] 1.2 Implement `_get_session_sequence` private method
    - Priority chain: `override` → Feature Store cache (TTL 60 s) → Event Store query → zero fallback
    - Query `events.get_user_interaction_history(user_id, limit=50)` (or equivalent iterator) sorted by `event_ts` ascending; take the 50 most recent
    - Map each movie ID through `item_id % self.num_items` (consistent with existing `safe_item_ids` pattern)
    - Left-pad with zeros to length 50 when fewer than 50 interactions exist
    - On any `Exception` from the Event Store, log `WARNING` and return zero tensor
    - Return `torch.zeros((1, 50), dtype=torch.long)` for cold-start users (no warning)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.3 Update `predict_ensemble` signature and replace stub sequence
    - Add optional parameter `session_sequence: list[int] | None = None` to `predict_ensemble`
    - Replace `simulated_seq = torch.zeros((1, 50), dtype=torch.long)` with `simulated_seq = self._get_session_sequence(user_id, override=session_sequence)`
    - _Requirements: 1.1, 1.6_

  - [x]* 1.4 Write property tests for session sequence (Properties 1 & 2)
    - **Property 1 & 2: Session Sequence Length Invariant and Padding Correctness**
    - **Validates: Requirements 1.2, 1.3, 1.7**
    - Create `tests/test_session_sequence.py`
    - Use `hypothesis` `@given` with random `user_id` (int) and random lists of 0–200 movie IDs
    - Assert output tensor shape is always `[1, 50]`
    - Assert all values are in `[0, num_items)`
    - Assert leading-zero count equals `max(0, 50 - len(interactions))`
    - Include unit test cases: cache hit, cache miss, I/O error fallback, override parameter bypass
    - Tag: `Feature: apex-peak-capability, Property 1 & 2`

- [x] 2. Add ensemble weight loader to `ensemble_engine.py`
  - [x] 2.1 Implement `_load_weights` private method
    - Read `models/ensemble_weights.json`; parse keys `lightgcn`, `quantum`, `sasrec`, `kan`, `hyperbolic`, `diffusion`
    - If file absent, JSON parse error, or any required key missing: log `WARNING`, return hard-coded defaults (`lightgcn=0.65`, `quantum=0.25`, `sasrec=0.10`, others `0.00`)
    - If loaded weights do not sum to 1.0 (tolerance 1e-6): log `WARNING` and re-normalise before returning
    - Protect weight swap with `threading.Lock` so `reload_weights` is safe under concurrent requests
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 2.2 Call `_load_weights` from `__init__` and wire weights into `predict_ensemble`
    - Add `import threading` and `self._weights_lock = threading.Lock()` in `__init__`
    - Call `self._weights = self._load_weights()` at the end of `__init__`
    - Replace the hard-coded blend coefficients in `predict_ensemble` with `self._weights` values (read inside `self._weights_lock`)
    - _Requirements: 2.3, 2.4_

  - [x] 2.3 Implement `reload_weights` public method
    - Acquire `_weights_lock`, call `_load_weights()`, swap `self._weights`, release lock
    - Return the newly loaded weights dict
    - _Requirements: 2.8_

  - [x]* 2.4 Write property tests for ensemble weights (Property 3)
    - **Property 3: Ensemble Weights Sum to One**
    - **Validates: Requirements 2.5**
    - Create `tests/test_ensemble_weights.py`
    - Use `hypothesis` `@given` with random 6-element non-negative float vectors (Dirichlet-sampled)
    - Assert that after `_load_weights` normalisation, sum == 1.0 ± 1e-6 and all values >= 0
    - Include unit test cases: load from valid file, missing file fallback, malformed JSON fallback, missing key fallback, weight re-normalisation
    - Tag: `Feature: apex-peak-capability, Property 3`

- [ ] 3. Create `scripts/optimize_ensemble_weights.py`
  - [x] 3.1 Implement `run_dirichlet_grid_search` function
    - Sample at least 500 weight vectors from `numpy.random.dirichlet([1]*6)`
    - For each vector, call `ApexEnsembleEngine.predict_ensemble` on the validation split of the Event Store and compute NDCG@10 and Hit_Rate@10
    - Track the best-performing vector; log top-5 candidates and their NDCG@10 scores to stdout
    - Persist the best vector to `models/ensemble_weights.json` with the schema defined in the design (including `evaluated_at`, `ndcg_at_10`, `hit_rate_at_10`, `num_candidates_evaluated`)
    - Enforce all weights >= 0 and sum to 1.0 before writing
    - _Requirements: 2.1, 2.2, 2.5, 2.6, 2.7_

  - [x] 3.2 Add CLI entry point and `__main__` block
    - Accept `--num-candidates`, `--k`, `--output-path` CLI arguments via `argparse`
    - Callable as `python scripts/optimize_ensemble_weights.py`
    - _Requirements: 2.6_

- [x] 4. Add `POST /v1/admin/reload-ensemble-weights` endpoint to `main.py`
  - [x] 4.1 Import `get_apex_engine` and add the admin endpoint
    - Add `from backend.ensemble_engine import get_apex_engine` to `main.py` imports
    - Implement `POST /v1/admin/reload-ensemble-weights` handler that calls `get_apex_engine().reload_weights()`
    - Protect with existing admin-token auth (`resolve_admin_token` dependency)
    - Return `{"status": "ok", "weights": {...}, "source": "file" | "defaults"}`
    - _Requirements: 2.8_

- [x] 5. Create `backend/online_learner.py` with `OnlineLearner` class
  - [x] 5.1 Implement `OnlineLearner.__init__`, `enqueue`, `start`, `stop`
    - Constructor accepts `lightgcn: LightGCN`, `batch_size=32`, `lr=1e-4`, `checkpoint_interval=1000`, `checkpoint_path`
    - `enqueue`: compute `interaction_weight` from event type/rating per the design table; push to internal `queue.Queue(maxsize=10000)`; drop oldest and log `WARNING` if full
    - `start`: launch daemon thread running `_run`; store thread reference
    - `stop`: set stop event; join thread with timeout
    - _Requirements: 3.1, 3.3, 3.4, 3.5, 3.8_

  - [x] 5.2 Implement `_run`, `_apply_gradient_step`, `_checkpoint`
    - `_run`: drain queue in batches of up to `batch_size`; call `_apply_gradient_step`; increment event counter; call `_checkpoint` every `checkpoint_interval` events; on any exception log `ERROR`, clear batch, continue
    - `_apply_gradient_step`: construct BPR (user, pos_item, neg_item) triples from batch; compute BPR loss; clip gradients to L2 norm 1.0; apply Adam step to LightGCN user/item embeddings only
    - `_checkpoint`: save `lightgcn.state_dict()` to `checkpoint_path`; log `ERROR` on write failure but continue
    - _Requirements: 3.2, 3.6, 3.7, 3.9, 3.10_

  - [x]* 5.3 Write property tests for online learner (Property 4)
    - **Property 4: Online Learner Interaction Weight Assignment**
    - **Validates: Requirements 3.3, 3.4, 3.5**
    - Create `tests/test_online_learner.py`
    - Use `hypothesis` `@given` with random rating values in [1.0, 5.0] and random event types (`rating`, `click`)
    - Assert weight is exactly `+1.0` for `rating >= 4.0`, `-0.5` for `rating < 2.5`, `+0.3` for `click`
    - Include unit test cases: batch accumulation, gradient clipping, checkpoint trigger at 1000 events, exception recovery (gradient step raises → loop continues)
    - Tag: `Feature: apex-peak-capability, Property 4`

- [x] 6. Wire `OnlineLearner` into `main.py`
  - [x] 6.1 Instantiate `OnlineLearner` in the `lifespan` context manager
    - Import `OnlineLearner` from `backend.online_learner`
    - After `get_apex_engine()` is initialised in `lifespan`, create `online_learner = OnlineLearner(lightgcn=get_apex_engine().lightgcn)`
    - Call `online_learner.start()`; store as module-level singleton `_online_learner`
    - On `lifespan` shutdown, call `_online_learner.stop()`
    - Add dead-thread watchdog: if thread is not alive after startup, restart once; log `CRITICAL` if restart fails
    - _Requirements: 3.1, 3.8_

  - [x] 6.2 Enqueue events in the event-write endpoint
    - In the existing `POST /v1/events` handler, after the `append_event(...)` call, add `_online_learner.enqueue(event_dict)` for `click` and `rating` event types
    - _Requirements: 3.1_

- [x] 7. Checkpoint — Ensure all tests pass
  - Run `pytest tests/test_session_sequence.py tests/test_ensemble_weights.py tests/test_online_learner.py -v`
  - Ensure all tests pass; ask the user if questions arise.

- [x] 8. Create `scripts/finetune_two_tower.py`
  - [x] 8.1 Implement positive-pair extraction from the Event Store
    - Read events from `events.get_events_path()` (JSONL) or Postgres
    - Include as positive pair: `event_type == "rating"` with `rating >= 3.5`, or `event_type == "click"`
    - Exclude all other event types
    - If fewer than 100 positive pairs found: log `WARNING` and `sys.exit(0)` without writing any model file
    - _Requirements: 4.1, 4.3_

  - [x] 8.2 Implement hard-negative sampling and dataset construction
    - For each positive (user, item) pair, sample exactly 4 items the user has not interacted with as hard negatives
    - Reuse `TwoTowerDataset` from `scripts/train_two_tower.py` (import or duplicate with `num_negatives=4`)
    - Build user/item feature dicts using `build_user_features` / `build_item_features` helpers from `train_two_tower.py`
    - _Requirements: 4.2_

  - [x] 8.3 Implement training loop and output
    - Load base `models/two_tower.pth` weights as starting point
    - Train for `--epochs` (default 5) using `TwoTowerModel.compute_contrastive_loss` (InfoNCE)
    - Hold out 20% of pairs for validation; report Hit_Rate@10 after final epoch
    - On NaN loss: log `ERROR` and `sys.exit(1)` without writing model file
    - On successful completion: save to `models/two_tower_finetuned.pth`; log final training loss and Hit_Rate@10 to stdout
    - Accept `--epochs`, `--lr`, `--negatives` CLI arguments
    - _Requirements: 4.4, 4.5, 4.7, 4.8_

  - [x]* 8.4 Write property tests for fine-tuning pair construction (Properties 5 & 6)
    - **Property 5 & 6: Fine-Tuning Negative Ratio and Positive Pair Filter**
    - **Validates: Requirements 4.1, 4.2**
    - Create `tests/test_finetune_two_tower.py`
    - Use `hypothesis` `@given` with random lists of events (varying types and ratings)
    - Assert only qualifying events produce positive pairs (rating >= 3.5 or click; no other types)
    - Assert negative count == exactly 4 × positive count (before padding for insufficient negatives)
    - Include unit test cases: min-pairs guard (< 100 → exit 0), output file creation, NaN loss guard
    - Tag: `Feature: apex-peak-capability, Property 5 & 6`

- [x] 9. Update `recommender.py` to prefer fine-tuned Two-Tower weights
  - [x] 9.1 Add fine-tuned weight loading in `Recommender.load`
    - After the existing Two-Tower weight loading block, check for `models/two_tower_finetuned.pth`
    - If it exists and loads without error: use it in preference to base weights; log `INFO`
    - If it exists but is corrupt (exception on `load_state_dict`): log `WARNING` and fall back to base `two_tower.pth`
    - _Requirements: 4.6_

- [x] 10. Implement RL state builder and score application in `recommender.py`
  - [x] 10.1 Implement `_build_rl_state` function
    - Add module-level function `_build_rl_state(behavior_profile: dict, als_user_embedding: np.ndarray | None, state_dim: int = 20) -> torch.Tensor`
    - Construct state vector: `[log1p(total_ratings)/log1p(1000), avg_rating/5.0, log1p(click_count)/log1p(500), log1p(view_count)/log1p(500), als_emb[0..15]]`
    - Use zeros for any missing fields or when `als_user_embedding` is None
    - Return `torch.tensor(..., dtype=torch.float32).unsqueeze(0)` — shape `[1, 20]`
    - _Requirements: 5.8, 5.9_

  - [x] 10.2 Load `ActorCriticPolicy` in `Recommender.load`
    - Import `ActorCriticPolicy` from `backend.rl_policy`
    - After existing model loading, attempt to load `models/rl_policy.pth` with `state_dim=20, action_dim=16`
    - If file absent: log `DEBUG`; set `self._rl_policy = None`
    - If `state_dim` mismatch on `load_state_dict`: log `WARNING`; set `self._rl_policy = None`
    - On any other exception: log `WARNING`; set `self._rl_policy = None`
    - _Requirements: 5.3_

  - [x] 10.3 Apply RL score shift in `get_recommendations`
    - After `predict_ensemble` returns normalised scores, if `self._rl_policy` is not None:
      - Build RL state via `_build_rl_state(behavior_profile, als_user_embedding)`
      - Call `self._rl_policy.get_action(rl_state, deterministic=True)` → `action`
      - Project action to item score space: `action_scores = lightgcn_item_embs @ action.squeeze().numpy()`
      - Normalise `action_scores` to [0, 1]; add `0.1 * action_scores_norm` to ensemble scores
      - On any exception: log `WARNING`; serve unmodified ensemble scores
    - _Requirements: 5.1, 5.2_

  - [x] 10.4 Apply `RLSafetyFilter` before response serialisation
    - Import `RLSafetyFilter` from `backend.rl_policy`
    - After final ranking, extract `negative_movie_ids` from `behavior_profile`
    - Call `RLSafetyFilter.apply_hard_constraints(candidate_ids, negative_ids)`
    - If filter removes all candidates: log `WARNING`; revert to pre-filter list (existing fallback behaviour)
    - _Requirements: 5.10, 5.11_

  - [x]* 10.5 Write property tests for RL wiring (Properties 7 & 8)
    - **Property 7: RL State Vector Fixed Length**
    - **Validates: Requirements 5.8, 5.9**
    - **Property 8: RLSafetyFilter Exclusion Invariant**
    - **Validates: Requirements 5.10**
    - Create `tests/test_rl_wiring.py`
    - Property 7: use `hypothesis` `@given` with random behavior profiles (missing fields, zero counts, extreme values); assert output shape is always `[1, 20]` with no NaN/Inf
    - Property 8: use `hypothesis` `@given` with random candidate lists and random dislike subsets; assert output contains no item from dislike set when dislike set does not cover all candidates
    - Include unit test cases: zero-vector fallback when profile unavailable, RL skip when `rl_policy.pth` absent, active inference dispatch, safety filter all-candidates-removed revert
    - Tag: `Feature: apex-peak-capability, Property 7 & 8`

- [ ] 11. Wire Active Inference into `main.py` event-write endpoint
  - [x] 11.1 Add `_trigger_active_inference` helper and `get_active_inference_engine` singleton
    - Import `ActiveInferenceEngine` from `backend.active_inference_engine`
    - Add module-level singleton `_active_inference_engine: ActiveInferenceEngine | None = None` and `get_active_inference_engine()` getter (lazy-init)
    - Implement `async def _trigger_active_inference(movie_id: int, reward: float) -> None` that retrieves the movie embedding from the feature store (or uses a random proxy) and calls `engine.self_heal(movie_emb, reward)`; swallow all exceptions with `WARNING` log
    - _Requirements: 5.4, 5.5, 5.6, 5.7_

  - [x] 11.2 Dispatch `BackgroundTask` from the event-write endpoint
    - In `POST /v1/events`, after persisting the event, add:
      - If `event_type == "rating"` and `rating >= 4.0`: `background_tasks.add_task(_trigger_active_inference, movie_id, +1.0)`
      - If `event_type == "rating"` and `rating <= 2.0`: `background_tasks.add_task(_trigger_active_inference, movie_id, -1.0)`
    - Ensure `BackgroundTasks` is already in the handler signature (it is in the existing code)
    - _Requirements: 5.4, 5.6, 5.7_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/test_session_sequence.py tests/test_ensemble_weights.py tests/test_online_learner.py tests/test_finetune_two_tower.py tests/test_rl_wiring.py -v`
  - Ensure all tests pass; ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 7 and 12 provide incremental validation gates
- Property tests validate universal correctness properties; unit tests cover specific examples and edge cases
- The RL policy is loaded with `state_dim=20` (compact live-feature vector), not the `state_dim=771` used in `train_rl_policy.py` — if a pre-trained 771-dim policy exists it is silently skipped
- `OnlineLearner` applies BPR loss only to LightGCN embeddings; other model weights are not touched at runtime
- `finetune_two_tower.py` is a standalone CLI script; it does not run as part of the serving path

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "5.1"] },
    { "id": 2, "tasks": ["1.3", "2.3", "5.2"] },
    { "id": 3, "tasks": ["1.4", "2.4", "5.3", "3.1"] },
    { "id": 4, "tasks": ["3.2", "4.1", "6.1"] },
    { "id": 5, "tasks": ["6.2", "8.1"] },
    { "id": 6, "tasks": ["8.2"] },
    { "id": 7, "tasks": ["8.3", "10.1"] },
    { "id": 8, "tasks": ["8.4", "9.1", "10.2"] },
    { "id": 9, "tasks": ["10.3"] },
    { "id": 10, "tasks": ["10.4", "11.1"] },
    { "id": 11, "tasks": ["10.5", "11.2"] }
  ]
}
```
