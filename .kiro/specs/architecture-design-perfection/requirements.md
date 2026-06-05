# Requirements Document

## Introduction

The APEX Movie Recommendation System currently scores 9.5/10 on architecture and design. This feature closes the remaining gap by addressing five structural deficiencies: missing Architecture Decision Records (ADRs) that justify the 6-model ensemble and its blend weights; the absence of a visual architecture diagram in the README; no empirical ablation evidence for the ensemble composition; a monolithic `recommender.py` (2,528 lines) that conflates retrieval, ranking, reranking, and artifact validation into a single class; and a `main.py` (1,328 lines) that exceeds the 800-line maintainability threshold.

The deliverables are: `docs/ARCHITECTURE_DECISIONS.md`, a Mermaid diagram in `README.md`, `scripts/ablation_study.py` with output to `reports/ablation_report.json`, four new backend modules (`retrieval_pipeline.py`, `ranking_pipeline.py`, `reranking_pipeline.py`, `artifact_validator.py`), a refactored `recommender.py` reduced to under 600 lines acting as a thin orchestrator, and a `main.py` reduced to under 800 lines.

---

## Glossary

- **ADR (Architecture Decision Record)**: A short document that captures a significant architectural decision, its context, the decision made, and the consequences. Format: Context → Decision → Consequences.
- **Ablation Study**: An experiment that systematically removes one component at a time from a system and measures the degradation in a target metric (here, NDCG@10) to quantify each component's marginal contribution.
- **Retrieval Pipeline**: The first stage of the recommendation pipeline. Responsible for generating a broad set of candidate items from the full catalog using approximate nearest-neighbor search (FAISS), sparse TF-IDF matching, and knowledge-graph traversal. Exposes `retrieve(query_embedding, n) -> list[CandidateItem]`.
- **Ranking Pipeline**: The second stage of the recommendation pipeline. Responsible for scoring and ordering the candidate set produced by the Retrieval Pipeline using the 6-model ensemble engine, the MMoE ranker, and the LightGBM ranker. Exposes `rank(candidates, user_context) -> list[RankedItem]`.
- **Reranking Pipeline**: The third and final stage of the recommendation pipeline. Responsible for post-processing the ranked list to enforce diversity (MMR/submodular), apply the RL safety filter, optionally invoke LLM reranking, and enforce the quality gate. Exposes `rerank(ranked_items, constraints) -> list[FinalItem]`.
- **Candidate Item**: A data structure produced by the Retrieval Pipeline representing a movie that has passed the initial retrieval stage. Contains at minimum: `movie_id`, `retrieval_score`, and `retrieval_source` (one of `faiss`, `tfidf`, `knowledge_graph`).
- **Ranked Item**: A data structure produced by the Ranking Pipeline. Extends `CandidateItem` with `ensemble_score`, `ranker_score`, and `final_rank`.
- **Final Item**: A data structure produced by the Reranking Pipeline. Extends `RankedItem` with `diversity_score`, `safety_passed` (bool), and `explanation` (optional string).
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10. The primary offline evaluation metric used to measure recommendation quality.
- **MMR (Maximal Marginal Relevance)**: A greedy diversity algorithm that iteratively selects the item that maximizes a trade-off between relevance and dissimilarity to already-selected items.
- **LightGCN**: Light Graph Convolutional Network. A graph-based collaborative filtering model that propagates user-item interaction signals through a bipartite graph.
- **Quantum-Fluid Neural ODE**: A continuous-time neural network model (`QuantumFluidRecommender`) that models user preference dynamics as a wave-interference differential equation.
- **SASRec**: Self-Attentive Sequential Recommendation. A transformer-based model that captures sequential user behavior patterns.
- **KAN (Kolmogorov-Arnold Network)**: A neural network architecture using learnable B-spline activation functions instead of fixed activations, used here as a ranker.
- **Hyperbolic Recommender**: A model that embeds users and items in a Poincaré hyperbolic manifold to capture hierarchical content relationships.
- **Diffusion Recommender**: A score-based generative model (`LatentDiffusionRecommender`) that recommends items by reversing a diffusion process in the latent space.
- **Thin Orchestrator**: A class whose sole responsibility is to coordinate the three pipeline stages (Retrieval → Ranking → Reranking) and manage artifact loading, without containing any stage-specific logic itself.
- **Artifact Validator**: A module responsible for verifying the integrity of loaded model artifacts using SHA-256 checksums, row-count alignment, and manifest validation.

---

## Requirements

### Requirement 1: Architecture Decision Records Document

**User Story:** As a system architect reviewing APEX, I want a dedicated ADR document that explains why each of the 6 ensemble models was chosen and how the blend weights were determined, so that I can evaluate the design rationale without reading source code.

#### Acceptance Criteria

1. THE System SHALL create the file `docs/ARCHITECTURE_DECISIONS.md` containing at least one ADR entry per ensemble model (LightGCN, Quantum-Fluid Neural ODE, SASRec, KAN, Hyperbolic, Diffusion).
2. WHEN an ADR entry is rendered, THE Document SHALL follow the structure: **Context** (problem being solved), **Decision** (what was chosen), **Consequences** (tradeoffs accepted and alternatives rejected).
3. THE Document SHALL include an ADR entry explaining the default blend weights (LightGCN 0.65, Quantum 0.25, SASRec 0.10, KAN 0.00, Hyperbolic 0.00, Diffusion 0.00) with justification for why the three zero-weight models are retained in the codebase.
4. THE Document SHALL include an ADR entry for the 3-tier serving architecture (Tier1/GPU, Tier2/ONNX, Tier3/FAISS) explaining the hardware thresholds used for auto-detection (GPU + ≥16 GB RAM → Tier1; no GPU + ≥8 GB RAM → Tier2; <8 GB RAM → Tier3).
5. THE Document SHALL include an ADR entry for the pipeline decomposition decision (monolith → Retrieval/Ranking/Reranking stages), documenting the tradeoffs of the split.
6. WHERE a model's zero blend weight is documented in the ADR, THE Document SHALL explain the specific signal that model contributes that justifies its presence (e.g., KAN's B-spline activations for non-linear feature interactions, Hyperbolic's hierarchical genre embeddings, Diffusion's generative diversity); zero-weight models that are not yet documented SHALL remain functional in the codebase.
7. THE Document SHALL include a table of contents linking to each ADR by number and title.
8. IF a future decision supersedes an existing ADR, THEN THE Document SHALL support a "Superseded By" field in the affected ADR entry.

---

### Requirement 2: Architecture Diagram in README

**User Story:** As a developer onboarding to APEX, I want a visual architecture diagram in the README that shows the full data flow and component relationships, so that I can understand the system structure in under five minutes without reading multiple documentation files.

#### Acceptance Criteria

1. THE System SHALL add a Mermaid `flowchart TD` or `graph LR` diagram to `README.md` under a dedicated "Architecture Diagram" section.
2. WHEN the diagram is rendered on GitHub, THE Diagram SHALL show the complete request path: `UserRequest → RetrievalPipeline → RankingPipeline → RerankingPipeline → Response`.
3. THE Diagram SHALL show the 3-tier serving system with labeled nodes for Tier1 (GPU/full ensemble), Tier2 (ONNX CPU), and Tier3 (FAISS + TF-IDF only).
4. THE Diagram SHALL show the data pipeline flow: `Kafka Events → Feature Store → ETL → Delta Lake → Model Training → Serving Artifacts`.
5. THE Diagram SHALL show the actual ensemble models present in the system as sub-components of the Ranking Pipeline node, reflecting the true count rather than a fixed number.
6. THE Diagram SHALL use distinct visual groupings (subgraphs) to separate the serving path, the data pipeline, and the training pipeline.
7. WHEN the diagram is added, THE README SHALL retain all existing content and badge links without modification.
8. THE Diagram SHALL label the FAISS ANN index, the TF-IDF sparse index, and the Knowledge Graph as distinct retrieval sources feeding into the Retrieval Pipeline node.

---

### Requirement 3: Ensemble Ablation Evidence Script

**User Story:** As a machine learning engineer evaluating the APEX ensemble, I want a reproducible ablation study script that quantifies each model's marginal NDCG@10 contribution, so that the 6-model composition is defensible with empirical evidence rather than assertion.

#### Acceptance Criteria

1. THE System SHALL create the file `scripts/ablation_study.py` that implements a leave-one-out ablation over the 6 ensemble models (LightGCN, Quantum, SASRec, KAN, Hyperbolic, Diffusion).
2. WHEN the script is executed, THE Ablation_Script SHALL evaluate NDCG@10 for the full ensemble and for each of the 6 leave-one-out configurations (one model removed at a time).
3. THE Ablation_Script SHALL print a formatted table to stdout showing: model name, NDCG@10 with model removed, NDCG@10 delta vs. full ensemble, and marginal contribution percentage.
4. WHEN the script completes, THE Ablation_Script SHALL write results to `reports/ablation_report.json` containing: `run_timestamp`, `full_ensemble_ndcg`, and an array of per-model objects with fields `model`, `ndcg_without`, `delta`, `marginal_contribution_pct`.
5. IF the `reports/` directory does not exist, THEN THE Ablation_Script SHALL create it before writing the report.
6. THE Ablation_Script SHALL accept a `--sample-size` CLI argument (default: 1000) to control the number of evaluation queries, enabling fast runs during development.
7. THE Ablation_Script SHALL accept a `--output` CLI argument to override the default output path of `reports/ablation_report.json`.
8. WHEN a model fails to load (e.g., missing weights file), THE Ablation_Script SHALL log a warning, record `ndcg_without: null` (not `0.0`, to avoid confusion with a model that genuinely contributes nothing) for that model, and continue evaluating the remaining models.
9. THE Ablation_Script SHALL be executable as `python scripts/ablation_study.py` from the repository root without additional setup beyond the standard `requirements.txt`.

---

### Requirement 4: Recommender Pipeline Decomposition

**User Story:** As a backend engineer maintaining APEX, I want `recommender.py` decomposed into focused pipeline modules, so that each stage (retrieval, ranking, reranking, artifact validation) can be understood, tested, and modified independently without navigating a 2,528-line monolith.

#### Acceptance Criteria

1. THE System SHALL create `backend/retrieval_pipeline.py` exposing a `RetrievalPipeline` class with a public method `retrieve(query_embedding: np.ndarray, n: int) -> list[CandidateItem]`.
2. THE System SHALL create `backend/ranking_pipeline.py` exposing a `RankingPipeline` class with a public method `rank(candidates: list[CandidateItem], user_context: dict) -> list[RankedItem]`.
3. THE System SHALL create `backend/reranking_pipeline.py` exposing a `RerankingPipeline` class with a public method `rerank(ranked_items: list[RankedItem], constraints: dict) -> list[FinalItem]`.
4. THE System SHALL create `backend/artifact_validator.py` exposing an `ArtifactValidator` class with a public method `validate(artifact_path: Path) -> bool` that performs SHA-256 checksum verification, row-count alignment, and manifest validation.
5. WHEN `RetrievalPipeline.retrieve()` is called, THE RetrievalPipeline SHALL query at least one of: FAISS ANN index, sparse TF-IDF index, or Knowledge Graph, and return results tagged with their `retrieval_source`.
6. WHEN `RankingPipeline.rank()` is called, THE RankingPipeline SHALL apply the ApexEnsembleEngine scores and at least one learned ranker (MMoE or LightGBM) to produce a sorted `list[RankedItem]`.
7. WHEN `RerankingPipeline.rerank()` is called, THE RerankingPipeline SHALL apply MMR/submodular diversity optimization and the RL safety filter before returning the final list.
8. THE refactored `recommender.py` SHALL be reduced to under 600 lines and SHALL function as a thin orchestrator that: loads artifacts via `ArtifactValidator`, initializes the three pipeline instances, and delegates each recommendation request through the `retrieve → rank → rerank` sequence.
9. WHEN the serving tier is Tier3, THE Recommender orchestrator SHALL default to configuring `RankingPipeline` to use only the FAISS + TF-IDF path (skipping neural ensemble scoring) to preserve the low-memory Tier3 behavior; WHERE `NOVA_SERVING_PROFILE=full` is explicitly set, THE Recommender orchestrator SHALL allow the full ensemble to be applied even on Tier3.
10. WHEN the serving tier is Tier1 or Tier2, THE Recommender orchestrator SHALL configure `RankingPipeline` to use the full ensemble including all 6 neural models.
11. THE System SHALL preserve all existing public API contracts of `Recommender` (method signatures, return types, exception behavior) so that `main.py` and all callers require no changes.
12. WHEN `ArtifactValidator.validate()` detects a SHA-256 checksum mismatch, THE ArtifactValidator SHALL raise a `ValueError` with a message identifying the artifact path and the expected vs. actual checksum.

---

### Requirement 5: main.py Line Count Reduction

**User Story:** As a backend engineer reviewing APEX, I want `main.py` to be under 800 lines, so that the API entry point remains readable and focused on routing and request handling rather than business logic.

#### Acceptance Criteria

1. THE System SHALL reduce `backend/main.py` from its current 1,328 lines to under 800 lines.
2. WHEN large logical blocks are extracted from `main.py`, THE System SHALL move them into appropriately named modules under `backend/` (e.g., `backend/recommendation_routes.py` if it does not already exist, or extend existing route modules).
3. THE refactored `main.py` SHALL retain all existing API endpoint paths, HTTP methods, request/response schemas, and authentication middleware without behavioral change.
4. IF a block of code extracted from `main.py` is already partially present in an existing module, THEN THE System SHALL consolidate it into that module rather than creating a new file.
5. THE System SHALL verify that all existing tests in `backend/tests/` continue to pass after the refactor.

---

### Requirement 6: Pipeline Correctness Properties

**User Story:** As a quality engineer validating the APEX pipeline decomposition, I want formally specified correctness properties for the three pipeline stages, so that property-based tests can verify the pipeline behaves correctly across arbitrary inputs.

#### Acceptance Criteria

1. WHEN `RetrievalPipeline.retrieve(query_embedding, n)` is called with a non-empty catalog, THE RetrievalPipeline SHALL return a list containing at least 1 `CandidateItem` (non-empty retrieval guarantee).
2. WHEN `RetrievalPipeline.retrieve(query_embedding, n)` is called, THE RetrievalPipeline SHALL return at most `n` items (upper-bound guarantee).
3. WHEN `RankingPipeline.rank(candidates, user_context)` is called, THE RankingPipeline SHALL return a list whose length equals `len(candidates)` (rank preserves candidate count — no items added or dropped).
4. WHEN `RankingPipeline.rank(candidates, user_context)` is called, THE RankingPipeline SHALL return items sorted in descending order by `final_rank` score (ordering invariant).
5. WHEN `RerankingPipeline.rerank(ranked_items, constraints)` is called, THE RerankingPipeline SHALL return only items whose `movie_id` values are a subset of the `movie_id` values in `ranked_items` (no hallucinated items — reranking cannot introduce items not in the input).
6. WHEN `RerankingPipeline.rerank(ranked_items, constraints)` is called with the same `ranked_items` and `constraints`, THE RerankingPipeline SHALL return the same ordered list on every call (determinism property).
7. WHEN `RankingPipeline.rank(candidates, user_context)` is called twice with identical inputs, THE RankingPipeline SHALL return identical scores for each item (determinism property).
8. WHEN `RetrievalPipeline.retrieve()` is called, THE RetrievalPipeline SHALL return no duplicate `movie_id` values in the result list (deduplication invariant).
9. WHEN `RerankingPipeline.rerank(ranked_items, constraints)` is called with an empty `ranked_items` list, THE RerankingPipeline SHALL return an empty list without raising an exception (empty-input safety).
10. WHEN `RetrievalPipeline.retrieve(query_embedding, n)` is called with `n = 0`, THE RetrievalPipeline SHALL return an empty list without raising an exception (zero-n safety).
11. FOR ALL valid `CandidateItem` lists `C`, `RankingPipeline.rank(C, ctx)` followed by extracting `movie_id` values SHALL produce the same set of `movie_id` values as the input `C` (set-identity round-trip property).
12. WHEN `ArtifactValidator.validate()` is called on the same artifact file twice in succession without modification, THE ArtifactValidator SHALL return the same boolean result both times (idempotence property).
