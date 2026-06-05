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

**Negative:** The auto-detection logic adds complexity to the startup sequence. If the `TierDetector` misclassifies the hardware (e.g., a GPU is present but has insufficient VRAM for the full ensemble), the system may attempt to load models that exceed available GPU memory and crash. The current implementation does not check GPU VRAM — it only checks for GPU presence and system RAM. A future improvement would add a VRAM check to the Tier1 detection logic. Additionally, the three-tier design means that the system's behavior is not fully deterministic from the user's perspective: the same request may produce different results on Tier1 vs. Tier3, which complicates debugging and A/B testing across environments.

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
