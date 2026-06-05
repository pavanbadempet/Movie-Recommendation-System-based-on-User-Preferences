# Requirements Document

## Introduction

The APEX Peak Capability upgrade closes five critical gaps in the current recommendation system to bring it to full production-grade operation. The system currently runs a 6-model ensemble (LightGCN, Quantum-Fluid, SASRec, KAN, Hyperbolic, Diffusion) but three of those models contribute zero weight, SASRec uses a simulated zero-sequence instead of real user history, the ensemble weights were never tuned against real data, the Two-Tower model has no fine-tuning loop on live interaction data, and the RL/Active Inference components exist as isolated files that are never called from the main serving path.

These five gaps are addressed as a single cohesive feature set so that every inference request benefits from real user history, properly calibrated model weights, continuously improving embeddings, and a live RL policy that optimises for long-term retention.

## Glossary

- **Ensemble_Engine**: The `ApexEnsembleEngine` class in `backend/ensemble_engine.py` that fuses scores from all six models.
- **SASRec**: The Transformer-based sequential recommender in `backend/sasrec.py` that predicts the next item from a user's chronological watch history.
- **Session_Sequence**: An ordered list of up to 50 movie IDs representing a user's most recent interactions, retrieved from the event store or Redis.
- **Event_Store**: The behavior event persistence layer in `backend/events.py` (JSONL or Postgres) that records click, rating, view, and impression events.
- **Feature_Store**: The Redis-backed or in-memory store in `backend/feature_store.py` that caches user and item embeddings for sub-millisecond retrieval.
- **Ensemble_Weights**: The six scalar coefficients (summing to 1.0) that blend LightGCN, Quantum-Fluid, SASRec, KAN, Hyperbolic, and Diffusion scores in `predict_ensemble`.
- **Weight_Optimizer**: The offline grid-search or learned optimizer (script in `scripts/`) that evaluates candidate weight vectors against a held-out validation set and selects the best-performing configuration.
- **Online_Learner**: The background component that consumes live click and rating events and applies incremental gradient updates to model embeddings without a full retraining cycle.
- **Two_Tower_Model**: The dual-encoder neural network in `backend/two_tower.py` that maps users and items into a shared 128-dimensional embedding space.
- **RL_Policy**: The A2C Actor-Critic network in `backend/rl_policy.py` that outputs a genre/cluster weight vector to shift item scores toward long-term retention.
- **Active_Inference_Engine**: The free-energy minimisation component in `backend/active_inference_engine.py` that self-heals the dynamic prior on live thumbs-up / thumbs-down feedback.
- **Recommender**: The main orchestration class in `backend/recommender.py` that coordinates retrieval, reranking, and response assembly.
- **Serving_Path**: The FastAPI request handler chain in `backend/main.py` that processes a recommendation request end-to-end.
- **Validation_Set**: A held-out split of historical interaction events used exclusively for offline metric evaluation and weight selection.
- **NDCG**: Normalised Discounted Cumulative Gain — the primary offline ranking metric used to evaluate ensemble weight candidates.
- **Hit_Rate**: The fraction of validation users for whom at least one ground-truth item appears in the top-K recommendations.

---

## Requirements

### Requirement 1: Real User Session Sequences for SASRec

**User Story:** As a recommendation engineer, I want SASRec to receive the user's actual chronological interaction history instead of a zero-padded dummy sequence, so that the Transformer's sequential intent signal is grounded in real behaviour and contributes meaningfully to ensemble scores.

#### Acceptance Criteria

1. WHEN `predict_ensemble` is called with a `user_id`, THE `Ensemble_Engine` SHALL retrieve the user's `Session_Sequence` from the `Event_Store` before invoking `SASRec.predict`.
2. WHEN a `user_id` has fewer than 50 recorded interactions in the `Event_Store`, THE `Ensemble_Engine` SHALL left-pad the `Session_Sequence` with zeros to produce a tensor of length 50.
3. WHEN a `user_id` has exactly 50 or more recorded interactions, THE `Ensemble_Engine` SHALL use all 50 most recent interactions ordered by `event_ts` ascending without additional padding.
4. WHEN the `Event_Store` is unavailable or returns an error (including I/O failures), THE `Ensemble_Engine` SHALL fall back to a zero-padded sequence, log a warning at WARNING level, and continue serving without retrying or failing the prediction.
5. WHEN the `Feature_Store` contains a cached `Session_Sequence` for the `user_id` that is no older than 60 seconds, THE `Ensemble_Engine` SHALL use the cached sequence instead of querying the `Event_Store`.
6. THE `Ensemble_Engine` SHALL accept an optional `session_sequence` parameter in `predict_ensemble` that, when provided, overrides the `Event_Store` lookup entirely.
7. WHEN a `Session_Sequence` is retrieved from the `Event_Store`, THE `Ensemble_Engine` SHALL map each movie ID to its modulo-bounded index before constructing the input tensor, consistent with the existing `safe_item_ids` pattern.

---

### Requirement 2: Tuned Ensemble Weights via Offline Evaluation

**User Story:** As a recommendation engineer, I want the six ensemble model weights to be determined by offline evaluation against real interaction data rather than hard-coded constants, so that KAN, Hyperbolic, and Diffusion models contribute non-zero weight when they demonstrably improve ranking quality.

#### Acceptance Criteria

1. THE `Weight_Optimizer` SHALL evaluate candidate weight vectors by computing NDCG@10 and Hit_Rate@10 on the `Validation_Set` drawn from the `Event_Store`.
2. WHEN the `Weight_Optimizer` completes a search, THE `Weight_Optimizer` SHALL persist the best-performing weight vector to `models/ensemble_weights.json` as a JSON object with keys `lightgcn`, `quantum`, `sasrec`, `kan`, `hyperbolic`, `diffusion`.
3. WHEN `ensemble_weights.json` exists at startup, THE `Ensemble_Engine` SHALL load the weights from that file instead of using the hard-coded defaults.
4. WHEN `ensemble_weights.json` is absent, malformed, or cannot be read due to I/O or permission errors, THE `Ensemble_Engine` SHALL fall back to the hard-coded defaults (`lightgcn=0.65`, `quantum=0.25`, `sasrec=0.10`, others `0.00`) and log a WARNING.
5. THE `Weight_Optimizer` SHALL enforce that all six weights are non-negative and sum to 1.0 before persisting them.
6. THE `Weight_Optimizer` SHALL support a grid-search mode that evaluates at least 500 candidate weight vectors sampled from a Dirichlet distribution.
7. WHEN the `Weight_Optimizer` is run, THE `Weight_Optimizer` SHALL log the top-5 candidate weight vectors and their NDCG@10 scores to stdout.
8. THE `Ensemble_Engine` SHALL expose a `reload_weights` method that re-reads `ensemble_weights.json` without restarting the process, callable from a FastAPI admin endpoint.

---

### Requirement 3: Online Learning from Live Events

**User Story:** As a recommendation engineer, I want the system to incrementally update model embeddings from live click and rating events so that the recommendation quality improves continuously between full retraining cycles.

#### Acceptance Criteria

1. WHEN a `click` or `rating` event is appended to the `Event_Store`, THE `Online_Learner` SHALL enqueue the event for an incremental embedding update within 5 seconds.
2. THE `Online_Learner` SHALL apply incremental gradient updates to the LightGCN user and item embeddings using the event's implicit or explicit feedback signal.
3. WHEN a `rating` event has a `rating` value of 4.0 or above, THE `Online_Learner` SHALL treat it as a positive interaction with weight 1.0.
4. WHEN a `rating` event has a `rating` value below 2.5, THE `Online_Learner` SHALL treat it as a negative interaction with weight -0.5.
5. WHEN a `click` event is received without an accompanying rating, THE `Online_Learner` SHALL treat it as a weak positive interaction with weight 0.3.
6. THE `Online_Learner` SHALL process events in batches of up to 32 interactions before applying a gradient step, to amortise update cost.
7. WHEN the `Online_Learner` applies a gradient step, THE `Online_Learner` SHALL clip gradients to a maximum L2 norm of 1.0 to prevent embedding collapse.
8. THE `Online_Learner` SHALL run in a background daemon thread and SHALL NOT block the `Serving_Path`. WHEN the background thread fails to start or crashes, THE `Serving_Path` SHALL block new recommendation requests until the `Online_Learner` thread is successfully restored.
9. WHEN the `Online_Learner` has processed 1000 events since the last checkpoint, THE `Online_Learner` SHALL persist updated embeddings to `models/lightgcn_online.pth`.
10. IF the `Online_Learner` raises an unhandled exception during a gradient step, THEN THE `Online_Learner` SHALL log the error at ERROR level and continue processing subsequent events.

---

### Requirement 4: Two-Tower Model Fine-Tuning on Interaction Data

**User Story:** As a recommendation engineer, I want the Two-Tower model to be periodically fine-tuned on actual user interaction data from the event store so that its user and item embeddings reflect real preference signals rather than only the initial ALS priors.

#### Acceptance Criteria

1. THE `Two_Tower_Model` fine-tuning script SHALL read positive interaction pairs (user, item) from the `Event_Store` where the event type is `rating` with `rating >= 3.5` or `click`.
2. THE `Two_Tower_Model` fine-tuning script SHALL construct hard negatives by sampling items that the same user has not interacted with, at a ratio of 4 negatives per positive.
3. WHEN fewer than 100 positive interaction pairs are available in the `Event_Store` (including zero pairs), THE fine-tuning script SHALL log a WARNING and exit without modifying the saved model weights.
4. THE fine-tuning script SHALL train for a configurable number of epochs (default 5) using the InfoNCE contrastive loss already implemented in `TwoTowerModel.compute_contrastive_loss`.
5. WHEN fine-tuning completes, THE fine-tuning script SHALL save the updated model weights to `models/two_tower_finetuned.pth`.
6. WHEN `models/two_tower_finetuned.pth` exists, THE `Recommender` SHALL load it in preference to any base Two-Tower weights at startup.
7. THE fine-tuning script SHALL report final training loss and validation Hit_Rate@10 to stdout upon completion.
8. THE fine-tuning script SHALL be runnable as a standalone CLI command: `python scripts/finetune_two_tower.py`.

---

### Requirement 5: RL Policy and Active Inference Wired into the Live Serving Path

**User Story:** As a recommendation engineer, I want the RL policy and Active Inference engine to be connected to the main recommendation loop so that every serving request benefits from long-term retention optimisation and every piece of user feedback immediately updates the system's prior.

#### Acceptance Criteria

1. WHEN the `Recommender` generates a ranked candidate list, THE `Serving_Path` SHALL invoke `RL_Policy.get_action` with a state vector derived from the user's behavior profile to produce a genre/cluster weight shift vector.
2. WHEN `RL_Policy.get_action` successfully returns a weight shift vector, THE `Serving_Path` SHALL apply it to the ensemble scores before final ranking, by adding the action vector (projected to item score space) to the normalised ensemble scores. WHEN `RL_Policy.get_action` fails or raises an exception, THE `Serving_Path` SHALL serve the unmodified ensemble scores without applying any adjustment.
3. WHEN `RL_Policy` weights are absent from `models/rl_policy.pth`, THE `Serving_Path` SHALL skip the RL adjustment and serve the unmodified ensemble scores, logging a DEBUG message.
4. WHEN a `rating` event with `rating >= 4.0` is received, THE `Serving_Path` SHALL call `Active_Inference_Engine.self_heal` with the corresponding movie embedding and reward `+1.0`.
5. WHEN a positive feedback signal is received without an accompanying rating event, THE `Serving_Path` SHALL call `Active_Inference_Engine.self_heal` with the corresponding movie embedding and reward `+1.0`.
6. WHEN a `rating` event with `rating <= 2.0` or a negative feedback signal is received, THE `Serving_Path` SHALL call `Active_Inference_Engine.self_heal` with the corresponding movie embedding and reward `-1.0`.
7. THE `Active_Inference_Engine.self_heal` call SHALL be dispatched as a FastAPI `BackgroundTask` so that it does not add latency to the event-write response.
8. THE `RL_Policy` state vector SHALL be constructed from the user's behavior profile fields: `total_ratings`, `avg_rating`, `click_count`, `view_count`, and a 16-dimensional ALS user embedding, concatenated into a fixed-length vector.
9. WHEN the user behavior profile is unavailable for a given `user_id`, THE `Serving_Path` SHALL use a zero vector as the RL state and SHALL NOT raise an exception.
10. THE `RLSafetyFilter.apply_hard_constraints` SHALL be applied to the final candidate list before the response is serialised, removing any items present in the user's `negative_movie_ids` from the behavior profile.
11. WHEN `RLSafetyFilter` removes all candidates, THE `Serving_Path` SHALL revert to the pre-filter candidate list and log a WARNING, consistent with the existing fallback in `RLSafetyFilter`.
