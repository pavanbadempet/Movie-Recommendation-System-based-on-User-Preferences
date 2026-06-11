# Architecture Decision Records

This document captures the significant architectural decisions made in the APEX Movie Recommendation System. Each record follows the standard **Context → Decision → Consequences** format and is intended to give reviewers and future maintainers a clear rationale for why the system is built the way it is, without requiring them to read source code.

---

## Table of Contents

- [ADR-001: LightGCN as Primary Ensemble Component (weight: 0.65)](#adr-001-lightgcn-as-primary-ensemble-component-weight-065)
- [ADR-002: Quantum-Fluid Neural ODE for Temporal Preference Drift (weight: 0.25)](#adr-002-quantum-fluid-neural-ode-for-temporal-preference-drift-weight-025)
- [ADR-003: SASRec for Session-Level Sequential Patterns (weight: 0.10)](#adr-003-sasrec-for-session-level-sequential-patterns-weight-010)
- [ADR-004: KAN, Hyperbolic, and Diffusion at Zero Weight — Retained for Conditional Activation](#adr-004-kan-hyperbolic-and-diffusion-at-zero-weight--retained-for-conditional-activation)
- [ADR-005: 3-Tier Serving Architecture with Hardware Auto-Detection](#adr-005-3-tier-serving-architecture-with-hardware-auto-detection)
- [ADR-006: Pipeline Decomposition — Monolith → Retrieval / Ranking / Reranking](#adr-006-pipeline-decomposition--monolith--retrieval--ranking--reranking)
- [ADR-007: Doubly Robust IPS for Ensemble Weight Selection](#adr-007-doubly-robust-ips-for-ensemble-weight-selection)
- [ADR-008: Unified Online Learning Coordinator — Closing the Feedback Loop](#adr-008-unified-online-learning-coordinator--closing-the-feedback-loop)
- [ADR-009: Differential Privacy at Inference Time](#adr-009-differential-privacy-at-inference-time)
- [ADR-010: Uncertainty-Gated Ensemble Blending](#adr-010-uncertainty-gated-ensemble-blending)

---

## ADR-001: LightGCN as Primary Ensemble Component

| Field | Value |
|---|---|
| **Status** | Superseded by ADR-007 |
| **Date** | 2024-01-15 |
| **Superseded By** | ADR-007 (DR-Optimized Ensemble Weights) |

### Context

The APEX system must produce high-quality personalized recommendations for a catalog of tens of thousands of movies across a user base that spans a wide range of interaction densities — from power users with hundreds of ratings to cold-start users with fewer than five. The core challenge is learning a compact, generalizable representation of user-item affinity from sparse, implicit feedback (clicks, views, ratings). Classical matrix factorization approaches (ALS, SVD) treat each user-item pair independently and fail to propagate collaborative signals through the interaction graph. Neural approaches that rely on deep MLP towers (e.g., NCF) tend to overfit on sparse users and are computationally expensive at inference time. The system needs a model that is both expressive enough to capture higher-order collaborative signals and efficient enough to serve recommendations at low latency on commodity hardware.

### Decision

LightGCN (Light Graph Convolutional Network) is designated as the primary ensemble component with a blend weight of **0.65** — the largest weight in the ensemble. LightGCN propagates user and item embeddings through the bipartite user-item interaction graph using a simplified graph convolution that removes the non-linear transformation and feature transformation matrices present in standard GCN layers. This simplification makes LightGCN both more interpretable and empirically stronger than its heavier predecessors on recommendation benchmarks. The model is initialized with PySpark ALS embeddings derived from the Delta Lake Gold layer, which anchors the random initialization in a meaningful prior and dramatically accelerates convergence. At inference time, LightGCN scores are computed as the dot product of the propagated user and item embeddings, which is a single matrix multiply — fast enough to run on CPU without GPU acceleration.

### Consequences

**Positive:** LightGCN consistently achieves the highest NDCG@10 among all six ensemble components in leave-one-out ablation experiments, justifying its dominant weight. The graph propagation mechanism naturally handles the long-tail distribution of user interactions: even users with few direct interactions benefit from multi-hop neighborhood signals. The PySpark ALS initialization means the model produces meaningful scores even before fine-tuning on the full interaction graph. The dot-product scoring function is compatible with FAISS approximate nearest-neighbor search, enabling sub-millisecond candidate retrieval at scale.

**Negative:** LightGCN is a static collaborative filtering model — it has no mechanism for modeling temporal preference drift or session-level sequential patterns. A user whose tastes have shifted recently will receive recommendations anchored to their historical interaction distribution rather than their current intent. This limitation is explicitly addressed by the complementary ensemble components (ADR-002 for temporal drift, ADR-003 for session patterns). Additionally, LightGCN requires the full user-item interaction graph to be loaded into memory at startup, which contributes to the system's memory footprint and is one reason Tier3 (low-memory) mode bypasses the neural ensemble entirely.

**Alternatives Rejected:** Neural Collaborative Filtering (NCF) was evaluated but rejected due to higher inference latency and weaker empirical performance on sparse users. PinSage (a production-scale GCN variant) was considered but requires a distributed graph engine not available in the current infrastructure. Standard ALS was retained as the initialization prior but not as a standalone serving model, since it lacks the higher-order propagation that gives LightGCN its edge.

---

## ADR-002: Quantum-Fluid Neural ODE for Temporal Preference Drift

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-01-15 |
| **Superseded By** | *(none)* |

### Context

User preferences in movie recommendation are not static. A user who recently watched a series of action films is more likely to want another action film today than their long-term historical average would suggest. Conversely, a user who has been on a documentary binge may temporarily suppress their usual preference for comedies. Standard collaborative filtering models, including LightGCN, learn a single static embedding per user that represents their average preference over all historical interactions. This averaging effect smooths out the temporal dynamics that are often the most actionable signal for a recommendation system. The system needs a component that explicitly models how user preferences evolve over time — not just what a user has liked historically, but how their taste is currently trending.

### Decision

The `QuantumFluidRecommender` (Quantum-Fluid Neural ODE) is included in the ensemble with a blend weight of **0.25**. This model treats user preference dynamics as a continuous-time differential equation, inspired by the wave-interference formalism of quantum mechanics. User and item embeddings are represented as complex-valued amplitude vectors; the temporal evolution of a user's preference state is governed by a learned neural ODE that integrates the embedding trajectory from the user's last interaction timestamp to the current request time. The "quantum-fluid" framing captures the intuition that preferences can exist in superposition (a user can simultaneously prefer action and drama) and that interference effects (watching too many films of one genre can temporarily suppress appetite for that genre) are real phenomena in user behavior. The model is initialized with the same PySpark ALS priors as LightGCN, ensuring that the temporal dynamics are learned on top of a meaningful static baseline rather than from random noise.

### Consequences

**Positive:** The Neural ODE formulation allows the model to generalize to arbitrary time deltas between interactions without requiring discretization into fixed time buckets. This is particularly valuable for users who interact with the system infrequently — the model can extrapolate their preference trajectory over weeks or months rather than treating each session as independent. The wave-interference mechanism provides a natural way to model genre fatigue and novelty-seeking behavior, which are well-documented phenomena in recommendation system research. In ensemble ablation experiments, removing the Quantum-Fluid component produces the second-largest NDCG@10 drop after LightGCN, confirming that it captures a signal orthogonal to the graph-based collaborative filtering.

**Negative:** The Neural ODE integration step adds latency compared to a simple embedding lookup. The model requires a `time_delta` parameter at inference time, which means the serving layer must track the timestamp of each user's last interaction — an additional dependency on the event store. The complex-valued embedding arithmetic is not natively supported by standard deep learning hardware optimizations, which limits the benefit of GPU acceleration for this component. The "quantum-fluid" framing, while mathematically grounded, can be confusing to engineers unfamiliar with the formalism; the implementation in `backend/neural_ode_recommender.py` includes inline comments explaining the physical analogy.

**Alternatives Rejected:** Recurrent Neural Networks (RNNs/LSTMs) were considered for temporal modeling but rejected because they require fixed-length discrete time sequences and struggle with irregular interaction intervals. Time-aware matrix factorization (TimeSVD++) was evaluated but lacks the expressiveness of a continuous-time neural model. Attention-based temporal models (e.g., JODIE) were considered but require joint training of user and item trajectories, which is computationally prohibitive at the scale of the APEX catalog.

---

## ADR-003: SASRec for Session-Level Sequential Patterns

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-01-15 |
| **Superseded By** | *(none)* |

### Context

While LightGCN captures long-term collaborative signals and the Quantum-Fluid Neural ODE captures temporal preference drift, neither model is optimized for the within-session sequential patterns that drive a significant fraction of recommendation clicks. When a user watches three films in a single session, the third film they choose is heavily influenced by the specific sequence of the first two — not just by their overall historical preferences. A user who just watched *The Dark Knight* followed by *Inception* is in a very different mental state than a user who watched *The Notebook* followed by *La La Land*, even if both users have similar long-term preference profiles. The system needs a component that reads the current session's interaction sequence and predicts what the user wants next based on that specific context.

### Decision

SASRec (Self-Attentive Sequential Recommendation) is included in the ensemble with a blend weight of **0.10**. SASRec uses a unidirectional transformer (causal self-attention) to model the sequential dependencies in a user's interaction history. The model takes as input a fixed-length sequence of the user's 50 most recent item interactions (padded with zeros for cold-start users) and produces a query embedding that is used to score candidate items. The session sequence is retrieved from a three-tier priority chain: (1) the real-time in-memory feature updater for sub-millisecond latency, (2) a 60-second LRU session cache for recently active users, and (3) the event store index for users not in cache. The model is initialized with the same PySpark ALS item embeddings as LightGCN, ensuring that the sequential attention operates on semantically meaningful item representations from the start.

### Consequences

**Positive:** SASRec's causal self-attention mechanism is highly effective at capturing short-range sequential dependencies (e.g., genre momentum within a session) that neither LightGCN nor the Quantum-Fluid ODE model explicitly. The transformer architecture scales well with sequence length and can be efficiently batched across multiple users. The three-tier session retrieval chain ensures that the model has access to the most recent interactions with minimal latency overhead. In ablation experiments, SASRec's 0.10 weight reflects a genuine but modest marginal contribution — it is most valuable for active users with rich session histories and least valuable for cold-start users.

**Negative:** SASRec's contribution is inherently bounded by session data availability. For cold-start users or users who interact infrequently, the input sequence is mostly padding zeros, and the model degrades to a popularity-based prior. The 50-item sequence window means that very long interaction histories are truncated, potentially losing relevant signals from older interactions. The transformer's quadratic attention complexity over the sequence length is not a practical concern at 50 items, but would become one if the sequence window were extended significantly. The model also requires the event store to be queryable at inference time, adding an I/O dependency that must be managed carefully under high load.

**Alternatives Rejected:** GRU4Rec (GRU-based sequential recommendation) was evaluated but produces lower NDCG@10 than SASRec on the APEX evaluation set, consistent with the broader literature showing transformer-based models outperforming RNN-based models on sequential recommendation tasks. BERT4Rec (bidirectional transformer) was considered but requires masked training and is not naturally suited to the causal next-item prediction task. Markov chain-based sequential models were rejected as too simplistic to capture the multi-step dependencies present in movie-watching sessions.

---

## ADR-004: KAN, Hyperbolic, and Diffusion at Zero Weight — Retained for Conditional Activation

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-01-15 |
| **Superseded By** | *(none)* |

### Context

The APEX ensemble was designed to incorporate the most advanced recommendation paradigms available in the research literature. During development, four additional model architectures were implemented and evaluated: a Kolmogorov-Arnold Network (KAN) ranker, a Hyperbolic Poincaré manifold recommender, and a Latent Diffusion generative recommender. Each of these models captures a distinct and theoretically well-motivated signal that is not captured by LightGCN, the Quantum-Fluid ODE, or SASRec. However, empirical ablation experiments on the APEX evaluation set showed that the marginal NDCG@10 contribution of each of these three models, when added to the existing LightGCN + Quantum + SASRec ensemble, was below the noise floor of the evaluation. The question arose: should these models be removed from the codebase entirely, or retained at zero weight?

### Decision

KAN, Hyperbolic, and Diffusion are retained in the codebase with blend weights of **0.00** each. They are not removed. The rationale is that each model captures a signal that is genuinely valuable in specific contexts that are not well-represented in the current evaluation set, and the cost of retaining them (memory, initialization time) is acceptable given the hardware requirements already imposed by the active ensemble components. The three models are available for **conditional activation** — their weights can be raised above zero via the `models/ensemble_weights.json` configuration file without any code changes, enabling rapid experimentation when the evaluation context changes.

The specific signal each zero-weight model contributes is:

- **KAN (Kolmogorov-Arnold Network):** B-spline non-linear feature interactions. KAN replaces fixed activation functions with learnable B-spline curves, allowing the model to learn arbitrary non-linear mappings between user and item feature combinations. This is most valuable when rich side-feature vectors (genre, director, cast, mood tags) are available and the relationship between features and preference is highly non-linear. In the current APEX deployment, the feature vectors are relatively sparse, which limits KAN's advantage over simpler dot-product scoring. As the feature engineering pipeline matures and richer content features become available, KAN's contribution is expected to increase.

- **Hyperbolic Recommender:** Hierarchical genre embeddings. The Hyperbolic model embeds users and items in a Poincaré ball (a hyperbolic manifold) rather than Euclidean space. Hyperbolic geometry is the natural space for representing tree-structured hierarchies — a property that maps directly onto the genre taxonomy of movies (e.g., Action → Superhero → Marvel Cinematic Universe). The model is most valuable for users whose preferences are strongly organized around a genre hierarchy (e.g., a user who watches exclusively arthouse drama) and least valuable for users with eclectic, cross-genre tastes. In the current evaluation set, the user population is diverse enough that the hierarchical signal is diluted in aggregate metrics, but the model is expected to provide significant lift for the hierarchically-concentrated user segment.

- **Diffusion Recommender:** Generative diversity. The Latent Diffusion model (`LatentDiffusionRecommender`) generates item recommendations by reversing a diffusion process in the latent embedding space, starting from Gaussian noise and denoising toward a user-conditioned target. Unlike discriminative models that score existing items, the diffusion model's denoising trajectory naturally explores the latent space in a way that surfaces unexpected but coherent recommendations — items that a user would not have discovered through standard collaborative filtering. This generative diversity signal is most valuable in serendipity-focused recommendation scenarios and for users who have expressed explicit interest in discovering new content. The current ensemble already includes an MMR diversity reranking step, which partially subsumes the diffusion model's contribution at the aggregate level.

### Consequences

**Positive:** Retaining the three models at zero weight preserves the ability to activate them instantly via configuration change, without requiring a code deployment. This is particularly valuable for A/B testing: a new experiment can raise KAN's weight to 0.05 and lower LightGCN's weight to 0.60 in a single JSON file edit, with the change taking effect on the next weight reload cycle (no restart required, thanks to the `reload_weights()` hot-reload mechanism). The models also serve as a research platform: their implementations in `backend/kan_ranker.py`, `backend/hyperbolic_recommender.py`, and `backend/diffusion_recommender.py` are production-quality and can be used as baselines for future model development.

**Negative:** The three zero-weight models still consume memory and initialization time at startup. On Tier3 hardware (< 8 GB RAM), this overhead is non-trivial. The current implementation initializes all six models unconditionally in `ApexEnsembleEngine.__init__()`, which means even Tier3 deployments pay the full initialization cost. A future optimization would be to lazily initialize zero-weight models only when their weight is raised above zero. Additionally, maintaining six model implementations increases the surface area for bugs and the burden of keeping all models compatible with future changes to the embedding infrastructure.

**Alternatives Rejected:** Removing the three models entirely was considered and rejected on the grounds that re-implementing them later would require significant engineering effort and would lose the institutional knowledge embedded in the current implementations. Keeping them as separate optional modules (loaded only when explicitly configured) was considered but rejected as premature optimization — the current memory overhead is acceptable on Tier1 and Tier2 hardware, and Tier3 already bypasses the neural ensemble entirely.

---

## ADR-005: 3-Tier Serving Architecture with Hardware Auto-Detection

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-01-15 |
| **Superseded By** | *(none)* |

### Context

The APEX system is deployed across a heterogeneous set of environments: a GPU-equipped production server, a CPU-only staging environment, developer laptops with varying amounts of RAM, and a free-tier cloud hosting service (Render) with constrained memory. A single serving configuration cannot be optimal across all of these environments. Running the full 6-model neural ensemble on a machine with 4 GB of RAM will cause out-of-memory errors; conversely, falling back to FAISS + TF-IDF on a GPU machine wastes the available compute and degrades recommendation quality. The system needs a mechanism to automatically select the appropriate serving configuration based on the hardware available at startup, without requiring manual configuration changes per deployment environment.

### Decision

The system implements a **3-tier serving architecture** with automatic hardware detection at startup. The tier is determined by the `TierDetector` module, which inspects the available hardware and selects the appropriate tier according to the following thresholds:

| Tier | Hardware Condition | Active Components |
|---|---|---|
| **Tier 1** | GPU present **AND** system RAM ≥ 16 GB | Full 6-model neural ensemble (LightGCN + Quantum + SASRec + KAN + Hyperbolic + Diffusion), `torch.compile` optimization, full FAISS + TF-IDF + KG retrieval |
| **Tier 2** | No GPU **AND** system RAM ≥ 8 GB | ONNX Runtime quantized inference (2–5× faster than PyTorch on CPU), full retrieval stack, dynamic INT8 quantization on KAN and Diffusion |
| **Tier 3** | System RAM < 8 GB | FAISS ANN + TF-IDF only (no neural ensemble), TF-IDF vocabulary capped at 12,000 features, KG traversal disabled, low-memory mode |

The RAM thresholds (16 GB for Tier1, 8 GB for Tier2) were chosen empirically based on the memory footprint of the full ensemble at the current embedding dimension (16) and catalog size (~10,000 items). The 16 GB threshold for Tier1 ensures that the GPU VRAM is not the bottleneck — the full ensemble fits comfortably in 16 GB with room for the FAISS index and the serving process overhead. The 8 GB threshold for Tier2 ensures that the ONNX-quantized models fit in RAM with the TF-IDF index loaded. Below 8 GB, only the FAISS index and a lightweight TF-IDF model are loaded.

The tier can be overridden by setting the `NOVA_SERVING_PROFILE=full` environment variable, which forces Tier1 behavior regardless of detected hardware. This escape hatch is intended for development and testing scenarios where the engineer wants to exercise the full ensemble on a machine that would otherwise be classified as Tier3.

### Consequences

**Positive:** The auto-detection mechanism eliminates the need for environment-specific configuration files. A single Docker image can be deployed to any environment and will automatically select the appropriate serving mode. This dramatically simplifies the deployment pipeline and reduces the risk of misconfiguration. The three-tier design also provides a natural degradation path: if the GPU fails or memory pressure increases, the system can be restarted in a lower tier without any code changes. The ONNX Runtime path in Tier2 provides a significant performance improvement over PyTorch CPU inference (2–5× speedup), making the system viable on CPU-only cloud instances without sacrificing recommendation quality.

**Negative:** The auto-detection logic adds complexity to the startup sequence. The three-tier design means that the system's behavior is not fully deterministic from the user's perspective: the same request may produce different results on Tier1 vs. Tier3, which complicates debugging and A/B testing across environments.

**Resolution (2026-06-10):** `TierDetector.detect()` now queries `torch.cuda.get_device_properties(0).total_memory` and records `gpu_vram_gb` on `HardwareProfile`. The `_auto_select()` method requires `gpu_vram_gb >= 8.0` (the empirically measured headroom threshold for the 6-model ensemble at `emb_dim=16`, 10k items) before selecting Tier1. If VRAM cannot be measured (`gpu_vram_gb == 0.0`), the method falls back to Tier2 with a warning and requires an explicit `NOVA_SERVING_TIER=tier1` override. This eliminates the OOM risk identified above.

**Alternatives Rejected:** A single universal serving configuration was rejected because it would either be too resource-intensive for constrained environments or too conservative for well-resourced ones. A manual configuration file per environment was considered but rejected because it requires operational discipline to keep in sync with the actual hardware, and experience shows that such files inevitably drift. A two-tier design (GPU vs. CPU) was considered but rejected because the memory constraint is orthogonal to the GPU constraint — a CPU machine with 32 GB of RAM should use the full ONNX ensemble, while a CPU machine with 4 GB should not.

---

## ADR-006: Pipeline Decomposition — Monolith → Retrieval / Ranking / Reranking

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-01-15 |
| **Superseded By** | *(none)* |

### Context

The original `recommender.py` grew organically to 2,528 lines as new features were added: FAISS retrieval, TF-IDF search, knowledge graph traversal, the 6-model neural ensemble, MMoE and LightGBM learned rankers, MMR diversity reranking, RL safety filtering, LLM reranking, artifact validation, and tier-aware configuration. All of these concerns were implemented as methods on a single `Recommender` class, making the file extremely difficult to navigate, test, and modify. A change to the diversity reranking logic required understanding the entire file to avoid unintended side effects. Unit tests for the retrieval logic had to instantiate the full `Recommender` object, including loading all neural models, which made tests slow and fragile. The file had become a maintenance liability that slowed down feature development and increased the risk of regressions.

### Decision

`recommender.py` is decomposed into four focused modules plus a thin orchestrator, following the single-responsibility principle:

| Module | Responsibility | Public Interface |
|---|---|---|
| `backend/pipeline_types.py` | Shared dataclass definitions (`CandidateItem`, `RankedItem`, `FinalItem`) | Data types only — no logic |
| `backend/retrieval_pipeline.py` | Stage 1: FAISS + TF-IDF + KG → candidate set | `retrieve(query_embedding, n) -> list[CandidateItem]` |
| `backend/ranking_pipeline.py` | Stage 2: Ensemble + Learned Ranker → scored, sorted list | `rank(candidates, user_context) -> list[RankedItem]` |
| `backend/reranking_pipeline.py` | Stage 3: MMR + RL Safety + LLM → final list | `rerank(ranked_items, constraints) -> list[FinalItem]` |
| `backend/artifact_validator.py` | SHA-256 checksum + row-alignment validation | `validate(artifact_path) -> bool` |
| `backend/recommender.py` | Thin orchestrator: load artifacts, init pipelines, delegate requests | All existing public methods preserved |

The `pipeline_types.py` module is the architectural keystone of this decomposition. By defining all shared dataclasses in a single module with no imports from other `backend/` modules, it breaks the circular import chain that would otherwise arise if `retrieval_pipeline.py` imported from `ranking_pipeline.py` or vice versa. The import graph is strictly acyclic: `pipeline_types` ← `retrieval_pipeline`, `ranking_pipeline`, `reranking_pipeline` ← `recommender` ← `main`.

The refactored `recommender.py` is reduced to under 600 lines. It contains no retrieval, ranking, or reranking logic — only artifact loading, tier detection, pipeline initialization, and delegation. All existing public API contracts (`recommend_by_id`, `recommend_by_index`, `search_movies`, `semantic_search`, `kg_recommend`, `visual_search`, `get_movie_by_id`, `get_all_titles`, `recommend_for_user_profile`) are preserved without behavioral change, ensuring that `main.py` and all callers require no modifications.

### Consequences

**Positive:** Each pipeline stage can now be unit-tested in isolation without loading the full model stack. A test for `RetrievalPipeline.retrieve()` only needs a mock FAISS index and a small movie DataFrame — it does not need to instantiate `QuantumFluidRecommender` or `LightGCN`. This reduces test setup time from tens of seconds to milliseconds and makes tests far more reliable. The well-defined interfaces between stages (`CandidateItem`, `RankedItem`, `FinalItem`) also enable property-based testing: universal invariants such as "ranking preserves candidate count" and "reranking cannot introduce items not in the input" can be verified across thousands of randomly generated inputs using Hypothesis. The decomposition also makes the system's data flow explicit and auditable — a reviewer can understand the full recommendation pipeline by reading four short files rather than one enormous one.

**Negative:** The decomposition introduces more files and more indirection. A developer who wants to understand how a recommendation is produced must now trace the call chain across `recommender.py` → `retrieval_pipeline.py` → `ranking_pipeline.py` → `reranking_pipeline.py`, rather than reading a single file. This is a standard tradeoff in software architecture: modularity improves maintainability at the cost of increased navigation overhead. The `pipeline_types.py` module also introduces a shared dependency that all three pipeline modules must import — any change to the dataclass definitions (e.g., adding a field to `CandidateItem`) requires updating all three pipeline modules. This coupling is intentional and manageable, but it means that the dataclass definitions must be treated as a stable public API.

**Alternatives Rejected:** Keeping the monolith but adding better internal organization (e.g., splitting into private methods with clear naming conventions) was considered but rejected because it does not solve the core problems of slow test setup and inability to test stages in isolation. Splitting into separate Python packages (e.g., `backend.retrieval`, `backend.ranking`) was considered but rejected as over-engineering for the current scale — the module-level decomposition achieves the same separation of concerns without the additional packaging complexity. An event-driven pipeline (where each stage publishes to a message queue and the next stage consumes from it) was considered for future scalability but rejected for the current implementation because it would introduce significant operational complexity (message broker, consumer management, distributed tracing) that is not justified by the current traffic volume.

---

## ADR-007: Doubly Robust IPS for Ensemble Weight Selection

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-26 |
| **Supersedes** | ADR-001 (LightGCN weight 0.65), ADR-002 (Quantum weight 0.25), ADR-003 (SASRec weight 0.10), ADR-004 (KAN/Hyperbolic/Diffusion at 0.00) |

### Context

The original ensemble weights (LightGCN=0.65, Quantum=0.25, SASRec=0.10, KAN/Hyperbolic/Diffusion=0.00) were set by hand based on intuition about each model's expected contribution. This approach has two problems. First, hand-tuned weights are not grounded in empirical evidence — they reflect prior beliefs rather than measured performance. Second, standard offline evaluation metrics (NDCG@10, HR@10) are biased toward popular items because popular items appear more frequently in test sets. A model that always recommends popular items scores well on standard metrics but provides no personalization value. Hand-tuning weights against biased metrics compounds this problem.

### Decision

Ensemble blend weights are determined by **Doubly Robust (DR) Inverse Propensity Scoring** grid search, implemented in `scripts/causal_debias_training.py`. The DR estimator combines:

1. **Direct reward imputation**: the model's predicted click probability for each candidate
2. **IPS correction**: reweighting each ground-truth interaction by `1 / propensity(item)`, where propensity is estimated from the empirical impression frequency in the event store

This produces an unbiased estimate of each weight vector's true recommendation quality, correcting for the popularity bias in the logging policy. The grid search evaluates 200 Dirichlet-sampled weight vectors and selects the one with the highest DR score.

The DR-optimized weights as of 2026-05-26:

| Model | DR Weight | Individual HR@10 | Individual NDCG@10 |
|---|---|---|---|
| SASRec | 0.659 | 0.761 | 0.520 |
| KAN | 0.298 | 0.694 | 0.439 |
| LightGCN | 0.005 | 0.672 | 0.411 |
| Diffusion | 0.024 | 0.521 | 0.309 |
| Quantum | 0.010 | 0.583 | 0.354 |
| Hyperbolic | 0.004 | 0.498 | 0.287 |

**Ensemble HR@10: 0.785 | NDCG@10: 0.542 | Lift over best individual (SASRec): +4.3%**

The shift from LightGCN-dominant (0.65) to SASRec-dominant (0.659) reflects the fact that real session sequences are now wired into the serving path via `backend/realtime_feature_updater.py`. When session data is available, SASRec's sequential attention provides stronger signal than LightGCN's static graph embeddings. KAN's high weight (0.298) validates that its Fourier basis functions capture non-linear feature interactions that the other models miss.

LightGCN's near-zero weight (0.005) in this run reflects sparse live event data — the online learner has not yet accumulated enough interactions to fully train the graph. The weight is expected to increase as the event store grows.

### Consequences

**Positive:** Weights are now grounded in empirical evidence rather than intuition. The DR estimator corrects for popularity bias, so the weights reflect true personalization quality rather than popularity-matching ability. The weight file (`models/ensemble_weights.json`) is hot-reloadable via `ApexEnsembleEngine.reload_weights()` — no restart required to update weights after a new DR optimization run. The DR optimization script can be run on a schedule (e.g., weekly) to keep weights current as the event store grows.

**Negative:** The DR estimator requires a non-trivial amount of interaction data to produce reliable estimates. With fewer than ~1,000 unique users in the event store, the propensity estimates are noisy and the DR scores may not be meaningfully different across weight vectors. In this regime, the DR-optimized weights may not be significantly better than hand-tuned weights. The optimization also requires loading the full ensemble engine, which takes 30–60 seconds on CPU hardware.

**Alternatives Rejected:** Standard NDCG@10 grid search was rejected because it is biased toward popular items. Bayesian optimization over the weight simplex was considered but rejected as over-engineering for a 6-dimensional search space — random Dirichlet sampling with 200 candidates is sufficient to find a good solution. Online bandit-based weight adaptation was considered but rejected because it requires a production traffic stream to generate feedback, which is not available in the current deployment.

### Mathematical Formulation

The Doubly Robust (DR) estimator combines a direct reward model r̂ with Inverse Propensity Scoring (IPS) to produce an unbiased estimate of a target policy's value:

```
            1   n
V_DR(π) =  — · Σ  [ r̂(xᵢ, aᵢ)  +  (rᵢ − r̂(xᵢ, aᵢ)) · π(aᵢ|xᵢ) / p(aᵢ|xᵢ) ]
            n  i=1
```

where:
- `rᵢ` is the observed reward (1 if user interacted, 0 otherwise)
- `r̂(xᵢ, aᵢ)` is the direct reward model's predicted click probability for user `xᵢ` and item `aᵢ`
- `p(aᵢ|xᵢ)` is the logging policy's propensity — estimated from the empirical impression frequency in the event store
- `π(aᵢ|xᵢ)` is the target policy's probability of recommending item `aᵢ` to user `xᵢ`

The DR estimator is **doubly robust** because it is unbiased if *either* the reward model r̂ is correct *or* the propensity model p is correct. This is strictly better than pure IPS (unbiased only if p is correct) or pure direct method (unbiased only if r̂ is correct).

#### Worked Example: 3-Item Popularity Correction

Consider a user who has interacted with three items: a blockbuster, a mid-tier film, and a niche indie.

| Item | Observed Reward rᵢ | Propensity p(aᵢ) | IPS Weight 1/p | Direct Model r̂ |
|---|---|---|---|---|
| Blockbuster (id=1) | 1 (clicked) | 0.40 (shown to 40% of users) | 2.5 | 0.85 |
| Mid-tier (id=2) | 1 (clicked) | 0.10 (shown to 10% of users) | 10.0 | 0.60 |
| Niche Indie (id=3) | 0 (not clicked) | 0.02 (shown to 2% of users) | 50.0 (clipped to 10) | 0.30 |

**Without IPS correction (standard NDCG):** The blockbuster and mid-tier film contribute equally (both clicked), so a model that ranks the blockbuster higher scores the same as one that ranks the mid-tier higher. This rewards popularity matching.

**With IPS correction:** The mid-tier click is reweighted by 10.0× (inverse of its low propensity), while the blockbuster click is only reweighted by 2.5×. The DR estimator gives ~4× more credit for correctly predicting the mid-tier interaction, because discovering that preference is more informative than confirming a popular item's popularity.

The IPS weight for the niche indie (50.0) is **clipped** to `clip_val=10.0` to prevent variance explosion from extremely rare items.

#### Dirichlet Sampling on the 6-Simplex

To search over the weight space, 200 candidate weight vectors are sampled from a symmetric Dirichlet distribution:

```
w₁, w₂, ..., w₂₀₀  ~  Dirichlet(α = [1, 1, 1, 1, 1, 1])
```

Each wᵢ is a 6-dimensional vector that sums to 1 (one weight per ensemble component: LightGCN, Quantum, SASRec, KAN, Hyperbolic, Diffusion). The symmetric Dirichlet with α=1 produces a uniform distribution over the 6-simplex, ensuring that all weight combinations are explored without bias toward any particular model. Each candidate is scored by V_DR, and the weight vector with the highest DR score is selected and written to `models/ensemble_weights.json`.

---

## ADR-008: Unified Online Learning Coordinator — Closing the Feedback Loop

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-05 |
| **Implements** | Real-time feedback loop for SASRec (DR weight 0.659) and KAN (DR weight 0.298) |

### Context

The original `OnlineLearner` (`backend/online_learner.py`) applied incremental BPR gradient updates only to **LightGCN** embeddings from live click and rating events. LightGCN's DR-optimized weight is 0.005 — the lowest in the ensemble. The two highest-weighted models, SASRec (0.659) and KAN (0.298), had no feedback loop: their weights were frozen at the values learned during offline training and never updated from production interactions. This meant that 95.7% of the ensemble's effective weight (SASRec + KAN combined) was learning nothing from real user behavior. The system was online in name only.

### Decision

A `OnlineLearningCoordinator` (`backend/online_learning_coordinator.py`) is introduced as a unified fan-out layer that routes every live event to three independent learners:

| Learner | File | Models Updated | Batch Size | LR | Checkpoint |
|---|---|---|---|---|---|
| `OnlineLearner` | `online_learner.py` | LightGCN user+item embeddings | 32 | 1e-4 | Every 1000 events |
| `SASRecOnlineLearner` | `sasrec_online_learner.py` | SASRec item embeddings + last attention block | 16 | 5e-5 | Every 500 events |
| `KANOnlineLearner` | `kan_online_learner.py` | KAN Fourier sin/cos coefficients | 32 | 1e-4 | Every 750 events |

Each learner runs in an independent daemon thread with a bounded queue (5,000–10,000 events). A single `coordinator.enqueue(event)` call fans out to all three queues. The coordinator exposes a `status()` method used by the SLO endpoint.

**SASRec learner design:** Fine-tunes only the item embedding table and the last attention block. Full backprop through all attention blocks would be too computationally expensive for an online update. The user's current session sequence is fetched from the real-time feature updater cache for accurate context.

**KAN learner design:** Updates only KAN's Fourier coefficients. LightGCN embeddings are passed as `detached()` tensors — no gradient flows back into LightGCN. This decoupling prevents two learners from issuing conflicting gradient updates to the same embeddings.

The coordinator is started in `main.py` lifespan only for **Tier 1** (GPU or high-RAM CPU). Tier 2 and Tier 3 continue to use the ONNX inference path which does not support online gradient updates.

### Consequences

**Positive:** 95.7% of the ensemble's effective weight now benefits from live feedback. SASRec's sequential attention adapts to real session patterns within minutes of deployment. KAN's edge functions adapt to actual click/rating distributions rather than offline training distributions. The three learners are fully independent — a crash in one learner does not affect the others or the serving path. The coordinator's `status()` method provides real-time visibility into queue depths and events-processed counters.

**Negative:** Three concurrent background threads add memory and CPU overhead on Tier 1 hardware. The per-model checkpoint files (`lightgcn_online.pth`, `sasrec_online.pth`, `kan_online.pth`) can diverge from the offline-trained weights if the online learner receives a systematically biased event stream (e.g., a burst of ratings from a single user). A future improvement would add a staleness detector that resets online weights to the offline baseline if the distribution shift exceeds a configurable threshold.

**Alternatives Rejected:** A single shared learner updating all three models was rejected because it would require serializing gradient updates across models, creating a bottleneck. Asynchronous gradient aggregation (federated-style) was considered but rejected as over-engineering for a single-server deployment. ONNX models cannot receive gradient updates, so Tier 2 was correctly excluded.

---

## ADR-009: Differential Privacy at Inference Time

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-05 |
| **Regulatory Context** | GDPR Article 25 (Privacy by Design), EU AI Act Article 10 |

### Context

APEX's architecture documentation and README have always described differential privacy as a compliance feature. The `DifferentialPrivacyEngine` and `privatize_user_embedding` were implemented in `backend/privacy.py` and `backend/privacy_preserving_ml.py` and tested in `backend/tests/test_fairness.py`. However, auditing the serving path revealed that **neither function was called during recommendation serving**. The privacy guarantee existed on paper but not in practice. A GDPR audit would correctly flag this as non-compliant.

### Decision

`privatize_user_embedding` is called in `apply_learned_ranker` (`backend/recommender_core.py`) at every recommendation request, before the user embedding is passed to the ensemble engine. The mechanism is **Gaussian (ε, δ)-DP**:

- **ε (epsilon):** Privacy budget. Default 1.0. Configurable via `APEX_DP_EPSILON` environment variable. Lower = more private, slightly less accurate.
- **δ (delta):** Failure probability. Fixed at 1e-5.
- **Sensitivity (Δf):** 2.0 (maximum L2 norm of a normalized embedding).
- **σ (noise scale):** σ = c × Δf / ε, where c = √(2 × log(1.25/δ)).
- **Re-normalization:** The noisy embedding is L2-normalized post-noise injection to prevent cosine similarity explosion.

The privatized embedding is injected back into the LightGCN embedding table **for this request only** — it is not persisted and does not affect the online learner's gradient updates (the online learner fetches the raw embedding directly from the table before the DP noise is applied).

### Consequences

**Positive:** The privacy guarantee is now mathematically enforced at every serving request. A single user's actual preference vector cannot be reconstructed from the ensemble's output, even with white-box access to all model weights. The ε=1.0 default is the standard recommendation from the differential privacy literature for high-utility, medium-privacy applications. The configurable `APEX_DP_EPSILON` allows operators to tune the privacy-utility tradeoff without code changes.

**Negative:** Injecting noise into the LightGCN embedding table entry is a hack — the correct approach would be to create a per-request noisy copy of the embedding without touching the shared table. The current implementation uses `torch.no_grad()` and writes directly to `.data`, which avoids creating a gradient but is not thread-safe if two concurrent requests for the same user_id are processed simultaneously. A future improvement would create a per-request embedding buffer rather than modifying the shared table.

**Resolution (2026-06-10):** The thread-safety issue is resolved. `apply_learned_ranker` in `recommender_core.py` now computes the DP-noised embedding into a local `privatized_user_emb_tensor` and passes it to `predict_ensemble()` via the new `user_emb_override` parameter. `_predict_ensemble_pytorch` in `ensemble_engine.py` uses this tensor directly instead of reading from the shared embedding table when it is provided. The shared table is never mutated. Concurrent requests for the same `user_id` each receive independently noised embeddings, which is both thread-safe and the correct DP behavior.

**Alternatives Rejected:** Applying DP noise at training time only (i.e., DP-SGD) was rejected because it does not protect against model inversion attacks at inference time. Applying DP noise only to the API response (output perturbation) was rejected because it would not protect the intermediate embedding representations. Local DP (applying noise on the client before sending events) was rejected because it requires client-side SDK changes that are outside the current scope.

---

## ADR-010: Uncertainty-Gated Ensemble Blending

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-06-05 |

### Context

The standard ensemble blending formula (weighted sum of normalized model scores) treats all candidates equally regardless of how much the 6 models agree on each candidate's relevance. A candidate that receives high scores from all 6 models is different from a candidate that receives a very high score from SASRec (weight 0.659) but very low scores from the other 5 models. In the second case, the ensemble sum is high (dominated by SASRec's weight) but the recommendation is fragile — it depends entirely on a single model's judgment, and that model may be wrong for this specific user-item pair.

### Decision

A per-item uncertainty gate is computed in `_predict_ensemble_pytorch` and applied as a multiplicative penalty to the blended score:

```
per_item_uncertainty[i] = Σ_m w_m × (score_m[i] - weighted_mean[i])²
confidence_gate[i] = 1 - 0.5 × (per_item_uncertainty[i] / max_uncertainty)
blended_score[i] = (Σ_m w_m × score_m[i]) × confidence_gate[i]
```

The gate is bounded to [0.5, 1.0]: even maximally uncertain items receive at least 50% of their raw blended score (rather than being suppressed entirely). Items where all models agree receive a gate of 1.0 (no penalty).

This is mathematically equivalent to a soft Bayesian model averaging step: the ensemble down-weights predictions that are not supported by consensus across architectures.

### Consequences

**Positive:** Fragile recommendations (high score from one model, low from others) are gently penalized rather than surfaced with false confidence. This is especially beneficial for cold-start users and rare items where individual models may have sparse training signal. The gate is computed in the same forward pass as the ensemble blend — no additional model calls are required.

**Negative:** The uncertainty gate introduces a non-linear interaction between model scores that makes it harder to attribute recommendation quality improvements to individual models in ablation experiments. The 0.5 lower bound is a heuristic; a future improvement would learn this bound from held-out validation data rather than fixing it.

**Alternatives Rejected:** Full Bayesian model averaging (computing the posterior over model weights given the data) was rejected as computationally prohibitive for real-time serving. Monte Carlo dropout for uncertainty estimation was considered but rejected because not all 6 models use dropout in their inference paths. Simple score variance (without weighting by DR weights) was considered but rejected because it treats all models equally regardless of their empirically validated contribution.
