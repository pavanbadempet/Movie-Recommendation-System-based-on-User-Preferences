# Requirements Document

## Introduction

APEX currently scores 9.5/10 on architecture and design. This feature — **apex-architecture-10** — closes four specific gaps to reach a perfect score:

1. **No model selection rationale**: The 6-model ensemble and its blend weights (LightGCN=65%, Quantum=25%, SASRec=10%) have no documented justification.
2. **God Class**: `backend/recommender.py` is 2,528 lines with 50+ methods spanning unrelated concerns.
3. **No architecture diagram**: The README has no visual showing the 4 layers, 3 serving tiers, and data flow.
4. **Implicit pipeline**: The retrieval → ensemble → ranking → diversity → LLM reranking stages are buried inside `recommend_by_index` as sequential method calls with no explicit abstraction.

The deliverables are: an ADR-style architecture decisions document, a controlled ablation study script, a decomposition of `recommender.py` into focused modules, and a Mermaid architecture diagram in the README.

---

## Glossary

- **APEX**: The Advanced AI Recommendation Engine — the production system being improved.
- **ADR**: Architecture Decision Record — a document capturing the context, decision, and rationale for a significant architectural choice.
- **Ablation Study**: A controlled experiment that measures the marginal contribution of each model by removing it from the ensemble and measuring the change in NDCG@10.
- **ArtifactLoader**: The new module (`backend/artifact_loader.py`) responsible for loading, validating, and SHA-256 checking all model artifacts.
- **RetrievalEngine**: The new module (`backend/retrieval.py`) responsible for FAISS dense retrieval, sparse TF-IDF retrieval, and index building.
- **RecommendationPipeline**: The new class (`backend/ranking_pipeline.py`) that makes the five recommendation stages explicit, composable, and independently testable.
- **UserProfiler**: The new module (`backend/user_profiling.py`) responsible for behavior profiling, genre affinity computation, behavior boost, and recency decay.
- **Orchestrator**: The refactored `backend/recommender.py` (~300 lines) that wires ArtifactLoader, RetrievalEngine, RecommendationPipeline, and UserProfiler together and preserves the existing public API.
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10 — the primary offline evaluation metric.
- **LightGCN**: Light Graph Convolutional Network — a graph-based collaborative filtering model.
- **SASRec**: Self-Attentive Sequential Recommendation — a transformer-based sequential model.
- **Quantum-Fluid Neural ODE**: A continuous-time neural ODE model using quantum-inspired wave interference for user-item interaction modelling.
- **Hyperbolic**: A Poincaré manifold-based model that embeds items in hyperbolic space to capture hierarchical genre/franchise structure.
- **KAN**: Kolmogorov-Arnold Network — a model using B-spline activation functions as a learnable ranker.
- **Diffusion**: Latent Diffusion Recommender — a score-based generative model that produces item embeddings via denoising.
- **FAISS HNSW**: Facebook AI Similarity Search with Hierarchical Navigable Small World graph index — the approximate nearest-neighbour library used for dense retrieval.
- **Blend Weights**: The scalar coefficients (summing to 1.0) that combine per-model scores into a single ensemble score.
- **Leave-One-Out Ablation**: An ablation variant where one model is removed and the remaining weights are renormalised to sum to 1.0.
- **Single-Model Ablation**: An ablation variant where only one model is active (weight = 1.0) and all others are zeroed.
- **Mermaid**: A Markdown-native diagramming language rendered natively by GitHub.
- **Backward Compatibility**: All existing public methods on the `Recommender` class remain callable and return the same types after the decomposition.

---

## Requirements

### Requirement 1: Architecture Decisions Document

**User Story:** As a reviewer or new engineer, I want a single document that explains why each architectural choice was made, so that I can evaluate whether the design is principled rather than a collection of fashionable models.

#### Acceptance Criteria

1. THE Architecture_Decisions_Document SHALL be created at `docs/ARCHITECTURE_DECISIONS.md`.
2. THE Architecture_Decisions_Document SHALL contain one ADR section for each of the 6 ensemble models (LightGCN, SASRec, Quantum-Fluid Neural ODE, Hyperbolic, KAN, Diffusion), documenting the unique contribution each model provides that the others cannot replicate.
3. THE Architecture_Decisions_Document SHALL contain an ADR section for the blend weights (LightGCN=65%, Quantum=25%, SASRec=10%, KAN=0%, Hyperbolic=0%, Diffusion=0%) that explains the evidence or principled reasoning behind each weight value.
4. THE Architecture_Decisions_Document SHALL contain an ADR section for the 3-tier serving architecture that documents at least two alternatives considered and the reasons they were rejected.
5. THE Architecture_Decisions_Document SHALL contain an ADR section for the choice of FAISS HNSW that documents at least two alternative ANN libraries considered and the reasons they were rejected.
6. WHEN a reader opens `docs/ARCHITECTURE_DECISIONS.md`, THE Architecture_Decisions_Document SHALL present each ADR with the sections: Status, Context, Decision, Rationale, Alternatives Considered, and Consequences.

---

### Requirement 2: Ablation Study Script

**User Story:** As a machine learning engineer, I want a reproducible ablation study script, so that I can generate empirical evidence for the ensemble blend weights and detect regressions when models are retrained.

#### Acceptance Criteria

1. THE Ablation_Script SHALL be created at `scripts/ablation_study.py`.
2. WHEN the Ablation_Script is executed, THE Ablation_Script SHALL evaluate NDCG@10 for each leave-one-out configuration (one model removed, remaining weights renormalised to sum to 1.0).
3. WHEN the Ablation_Script is executed, THE Ablation_Script SHALL evaluate NDCG@10 for each single-model configuration (one model active with weight 1.0, all others zeroed).
4. WHEN the Ablation_Script completes, THE Ablation_Script SHALL write results to `reports/ablation_results.json` containing: configuration name, active models, weights used, and NDCG@10 score for each configuration.
5. WHEN the Ablation_Script completes, THE Ablation_Script SHALL print a summary table to stdout showing each model's marginal NDCG@10 contribution (full ensemble score minus leave-one-out score).
6. IF trained model weights are absent from the `models/` directory, THEN THE Ablation_Script SHALL fall back to random scores and log a warning indicating that results are not meaningful without trained weights.
7. THE Ablation_Script SHALL complete execution without raising an unhandled exception regardless of whether trained weights are present.

---

### Requirement 3: ArtifactLoader Module

**User Story:** As a backend engineer, I want artifact loading and validation logic isolated in its own module, so that I can test, audit, and replace it without touching recommendation logic.

#### Acceptance Criteria

1. THE ArtifactLoader SHALL be implemented in `backend/artifact_loader.py`.
2. THE ArtifactLoader SHALL expose a `load()` method that loads all model artifacts (FAISS index, embedding vectors, movie metadata DataFrame, TF-IDF matrices, learned ranker, RL policy) from the paths currently used by `Recommender.__init__`.
3. THE ArtifactLoader SHALL expose a `validate()` method that verifies SHA-256 checksums for all artifacts listed in the artifact manifest file.
4. WHEN `validate()` is called on a set of artifacts, THE ArtifactLoader SHALL return the same validation result on a second call with the same artifacts (idempotent).
5. IF an artifact file is missing or its SHA-256 checksum does not match the manifest, THEN THE ArtifactLoader SHALL raise a descriptive exception identifying the specific artifact that failed validation.
6. THE ArtifactLoader SHALL expose the artifact manifest contract as a typed dataclass or TypedDict so that callers can inspect expected artifact paths without reading file system paths from source code.

---

### Requirement 4: RetrievalEngine Module

**User Story:** As a backend engineer, I want FAISS dense retrieval and sparse TF-IDF retrieval isolated in their own module, so that I can benchmark, swap, or extend retrieval strategies independently of ranking logic.

#### Acceptance Criteria

1. THE RetrievalEngine SHALL be implemented in `backend/retrieval.py`.
2. THE RetrievalEngine SHALL expose a `dense_retrieve(query_vector, k)` method that returns at most `k` candidate movie indices using the FAISS HNSW index.
3. WHEN `dense_retrieve(query_vector, k)` is called, THE RetrievalEngine SHALL return a list whose length is less than or equal to `k`.
4. WHEN `dense_retrieve(query_vector, k)` is called, THE RetrievalEngine SHALL return only movie IDs that exist in the loaded movie catalog.
5. THE RetrievalEngine SHALL expose a `sparse_retrieve(query_text, k)` method that returns at most `k` candidate movie indices using TF-IDF cosine similarity.
6. THE RetrievalEngine SHALL expose a `build_dense_index(vectors)` method that constructs and stores the FAISS HNSW index from a matrix of embedding vectors.
7. THE RetrievalEngine SHALL expose a `build_sparse_index(content_text)` method that constructs and stores the TF-IDF matrix from a Series of document strings.

---

### Requirement 5: RecommendationPipeline Module

**User Story:** As a backend engineer, I want the five recommendation stages to be an explicit, named pipeline class, so that each stage is independently testable, the pipeline degrades gracefully when optional stages are removed, and both sync and async execution paths are supported.

#### Acceptance Criteria

1. THE RecommendationPipeline SHALL be implemented in `backend/ranking_pipeline.py`.
2. THE RecommendationPipeline SHALL define the following named stages in order: `retrieve`, `ensemble_score`, `learned_rank`, `diversity_rerank`, `llm_rerank`.
3. WHEN any single optional stage (any stage except `retrieve`) is removed from the RecommendationPipeline, THE RecommendationPipeline SHALL execute the remaining stages without raising an exception.
4. THE RecommendationPipeline SHALL expose a synchronous `run(context)` method that executes all configured stages in order and returns the final ranked list.
5. THE RecommendationPipeline SHALL expose an asynchronous `arun(context)` method that executes all configured stages in order and returns the final ranked list.
6. WHEN `run(context)` and `arun(context)` are called with the same context and the same deterministic stage implementations, THE RecommendationPipeline SHALL return equivalent results from both execution paths.
7. THE RecommendationPipeline SHALL accept stage implementations as constructor arguments so that individual stages can be replaced with test doubles without modifying the pipeline class.

---

### Requirement 6: UserProfiler Module

**User Story:** As a backend engineer, I want user behavior profiling logic isolated in its own module, so that I can test genre affinity computation and behavior boost independently of the retrieval and ranking logic.

#### Acceptance Criteria

1. THE UserProfiler SHALL be implemented in `backend/user_profiling.py`.
2. THE UserProfiler SHALL expose a `compute_genre_affinity(user_events)` method that returns a dictionary mapping genre names to affinity scores.
3. WHEN `compute_genre_affinity(user_events)` is called, THE UserProfiler SHALL return affinity scores where every value is in the closed interval [0.0, 1.0].
4. WHEN `compute_genre_affinity(user_events)` is called, THE UserProfiler SHALL return affinity scores whose sum is less than or equal to 1.0.
5. THE UserProfiler SHALL expose a `compute_behavior_boost(movie_id, behavior_profile)` method that returns a scalar boost value based on the user's interaction history with the candidate movie's genres and attributes.
6. THE UserProfiler SHALL expose a `apply_recency_decay(events, half_life_days)` method that weights older events lower than recent events using exponential decay with the specified half-life.

---

### Requirement 7: Orchestrator (Refactored recommender.py)

**User Story:** As a backend engineer, I want `backend/recommender.py` to be a thin orchestrator that wires the new modules together, so that the file is comprehensible and the public API remains unchanged for all callers.

#### Acceptance Criteria

1. THE Orchestrator SHALL be implemented in `backend/recommender.py` and SHALL be no longer than 350 lines after decomposition.
2. THE Orchestrator SHALL delegate artifact loading to ArtifactLoader, retrieval to RetrievalEngine, pipeline execution to RecommendationPipeline, and user profiling to UserProfiler.
3. THE Orchestrator SHALL preserve all existing public methods: `recommend_by_index`, `recommend_by_id`, `recommend_batch`, `recommend_by_title`, `visual_search`, `kg_recommend`, `semantic_search`, and `refresh_behavior_features`.
4. WHEN any existing public method is called on the Orchestrator, THE Orchestrator SHALL return a result of the same type and structure as the pre-decomposition implementation.
5. THE Orchestrator SHALL not require changes to `backend/main.py` beyond import path updates for symbols that have moved to new modules.
6. WHEN the existing test suite in `backend/tests/` is executed after decomposition, THE Orchestrator SHALL not cause any previously passing test to fail.

---

### Requirement 8: Architecture Diagram in README

**User Story:** As a reviewer or new engineer, I want a visual architecture diagram in the README, so that I can understand the system's structure in under two minutes without reading source code.

#### Acceptance Criteria

1. THE README SHALL contain a Mermaid flowchart diagram in the Architecture Overview section.
2. THE Mermaid_Diagram SHALL show all 4 intelligence layers (Data Platform & Streaming, Machine Learning Engine, Advanced Aesthetics & Multi-Modal Understanding, Cognitive Intelligence & Compliance) and their primary components.
3. THE Mermaid_Diagram SHALL show the 3 serving tiers (Tier 1: GPU full ensemble, Tier 2: ONNX CPU, Tier 3: FAISS + TF-IDF) and the hardware conditions that activate each tier.
4. THE Mermaid_Diagram SHALL show the 5 recommendation pipeline stages (retrieve → ensemble_score → learned_rank → diversity_rerank → llm_rerank) and the data flow between them.
5. THE Mermaid_Diagram SHALL show the data flow from event ingestion through the feature store to the recommendation response.
6. WHEN the README is rendered on GitHub, THE Mermaid_Diagram SHALL render without syntax errors.

---

### Requirement 9: Backward Compatibility and Test Integrity

**User Story:** As a backend engineer, I want all existing callers and tests to continue working after the decomposition, so that the refactoring introduces zero regressions.

#### Acceptance Criteria

1. THE Orchestrator SHALL expose the `Recommender` class at `backend.pipeline.recommender.Recommender` so that all existing import statements remain valid.
2. WHEN `backend/main.py` is updated with import path corrections, THE System SHALL start without import errors.
3. WHEN the full test suite (`backend/tests/`) is executed after decomposition, THE System SHALL pass all tests that passed before decomposition.
4. IF a test imports a symbol that has moved to a new module, THEN THE Orchestrator module SHALL re-export that symbol so that the import continues to resolve without modification to the test file.
5. THE ArtifactLoader, RetrievalEngine, RecommendationPipeline, and UserProfiler modules SHALL each have their own unit test file in `backend/tests/`.
