# APEX Model Cards

This document provides model cards for each of the 6 ensemble models in the APEX recommendation engine. Each card documents training data, architecture, evaluation metrics, known limitations, and intended use — following the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) standard (Mitchell et al., 2019).

---

## Ensemble Summary

The APEX ensemble combines 6 complementary architectures. Weights are determined by **Doubly Robust IPS grid search** (200 Dirichlet-sampled candidates) to correct for popularity bias in the training signal.

| Model | HR@10 | NDCG@10 | DR-Optimized Weight | Paradigm |
|---|---|---|---|---|
| **Ensemble** | **0.785** | **0.542** | — | Weighted blend |
| SASRec | 0.761 | 0.520 | 0.659 | Sequential Transformer |
| KAN | 0.694 | 0.439 | 0.298 | Kolmogorov-Arnold Network |
| LightGCN | 0.672 | 0.411 | 0.005 | Graph Collaborative Filtering |
| Quantum-Fluid | 0.583 | 0.354 | 0.010 | Neural ODE + Complex Embeddings |
| Diffusion | 0.521 | 0.309 | 0.024 | Generative Latent Diffusion |
| Hyperbolic | 0.498 | 0.287 | 0.004 | Poincaré Ball Manifold |

**Ensemble lift over best individual model (SASRec): +4.3% NDCG@10**

Evaluation protocol: leave-one-out, 200 users, 100 candidates per user.  
Semantic benchmark (17 curated intent cases): HR@10 = 1.0, bad-hit rate = 0.0.

---

## 1. LightGCN — Graph Collaborative Filtering

**File:** `backend/lightgcn.py` | **Weights:** `models/lightgcn.pth`, `models/lightgcn_ips.pth`

### Architecture
- Bipartite user-item graph with 3-layer message passing
- No non-linear activations (pure linear propagation)
- Mean pooling of multi-hop embeddings (E⁰ through E³)
- BPR (Bayesian Personalized Ranking) loss with L2 regularization
- Embedding dim: 64 | Layers: 3

### Training
- **Dataset:** Synthetic interactions bootstrapped from TMDB metadata + real user events via online learner
- **Loss:** IPS-weighted BPR (`scripts/causal_debias_training.py`) — corrects for popularity bias
- **Optimizer:** Adam, lr=5e-4, weight_decay=1e-5, cosine LR schedule
- **IPS debiasing:** Laplace-smoothed propensity estimation from impression events; weights clipped at 10.0
- **Federated simulation:** `backend/privacy_preserving_ml.py` — gradients aggregated with DP noise (ε=5.0)

### Evaluation
- HR@10: 0.672 | NDCG@10: 0.411 (individual)
- DR-optimized ensemble weight: 0.005 (low in current run; expected to increase as live events accumulate via online learner)

### Intended Use
Multi-hop collaborative filtering. Captures "users who liked A also liked B" patterns that content-based models miss. Most valuable for users with rich interaction history.

### Known Limitations
- Cold-start: no signal for new users or items with zero interactions
- Requires adjacency matrix construction at training time — not updated in real-time
- Current weight near-zero reflects sparse live event data; weight will shift as the online learner accumulates interactions

---

## 2. SASRec — Self-Attentive Sequential Recommendation

**File:** `backend/sasrec.py` | **Weights:** `models/sasrec.pth`

### Architecture
- 2-block causal Transformer (same architecture as GPT, applied to item sequences)
- Multi-head self-attention (2 heads, hidden_dim=64)
- Causal mask prevents attending to future items
- Positional embeddings encode chronological order
- Point-wise feed-forward with residual connections
- Sequence length: 50 items

### Training
- **Dataset:** User interaction sequences from event store (click, rating, view events)
- **Loss:** Binary cross-entropy on next-item prediction
- **Session sequences:** Wired to real-time feature updater (`backend/realtime_feature_updater.py`) for live session data
- **Attention enrichment:** Bidirectional BERT4Rec-style user encoder (`backend/attention_user_model.py`) blended 70/30 with base embedding

### Evaluation
- HR@10: 0.761 | NDCG@10: 0.520 (individual, highest single model)
- DR-optimized ensemble weight: 0.659 (dominant model — real session sequences provide strong signal)

### Intended Use
Sequential intent modeling. Predicts the next item based on the user's exact chronological watch history. Most valuable for active users with recent session data.

### Known Limitations
- Degrades to cold-start behavior (zero sequence) for new users
- Sequence truncated to 50 items — very long histories lose early context
- Causal attention means it cannot leverage future context (by design)

---

## 3. KAN — Kolmogorov-Arnold Network Ranker

**File:** `backend/kan_ranker.py` | **Weights:** `models/kan_ranker.pth`

### Architecture
- 3-layer Kolmogorov-Arnold Network using Fourier basis functions
- Learnable edge functions (Fourier series: sin/cos coefficients per edge) instead of fixed node activations
- Grid size: 5 frequency components per edge
- SiLU base activation + Fourier superposition
- Input: concatenated user + item embeddings (32d → 64d → 32d → 1)

### Training
- **Dataset:** (user_embedding, item_embedding, label) triples derived from LightGCN embeddings
- **Loss:** Binary cross-entropy (CTR prediction)
- **Key property:** Interpretable — each edge function can be visualized as a 1D curve

### Evaluation
- HR@10: 0.694 | NDCG@10: 0.439 (individual, second-best)
- DR-optimized ensemble weight: 0.298 (second-highest — validates meaningful contribution)

### Intended Use
High-precision ranking of pre-retrieved candidates. The Fourier basis functions capture non-linear interaction patterns between user and item features that MLPs approximate less efficiently.

### Known Limitations
- Fourier KAN is a simplified approximation of the full B-spline KAN (Liu et al., 2024)
- Grid size of 5 limits expressiveness for very complex preference patterns
- Requires pre-computed embeddings as input — not an end-to-end retrieval model

---

## 4. Quantum-Fluid Neural ODE Recommender

**File:** `backend/neural_ode_recommender.py` | **Weights:** `models/quantum_fluid.pth`

### Architecture
- Complex-valued embeddings: z = amplitude × e^(i×phase) (Euler's formula)
  - Real part (amplitude): explicit historical preference
  - Imaginary part (phase): latent potential / trajectory
- Neural ODE dynamics: Euler approximation of dz/dt = W×z (4 steps)
- Wave interference scoring: |user_state + item_state|² (constructive = high score)
- Inspired by: Chen et al. "Neural ODEs" (NeurIPS 2018)

### Training
- **Loss:** Max-margin quantum loss: clamp(1 - pos_interference + neg_interference, min=0)
- **Initialization:** PySpark ALS embeddings injected as amplitude priors

### Evaluation
- HR@10: 0.583 | NDCG@10: 0.354 (individual)
- DR-optimized ensemble weight: 0.010

### Intended Use
Captures temporal drift in user preferences via continuous-time dynamics. The complex embedding space allows modeling of latent preference trajectories that static embeddings cannot represent.

### Known Limitations
- Euler ODE approximation introduces discretization error vs. true continuous-time integration
- Complex arithmetic is not natively GPU-optimized in all PyTorch versions
- Wave interference scoring is theoretically motivated but lacks empirical validation on large-scale datasets

---

## 5. Hyperbolic Recommender — Poincaré Ball Model

**File:** `backend/hyperbolic_recommender.py` | **Weights:** `models/hyperbolic.pth`

### Architecture
- Poincaré ball manifold with curvature c=1.0
- Möbius addition for vector arithmetic on the manifold
- Exponential map (exp_map0) to project Euclidean embeddings onto the ball
- Poincaré distance: (2/√c) × arctanh(√c × ||−x ⊕ y||)
- Fermi-Dirac margin loss: softplus(d_pos − d_neg + margin)

### Training
- **Loss:** Fermi-Dirac ranking loss (margin=1.0)
- **Initialization:** Uniform(-1e-3, 1e-3) — small values keep embeddings near the origin (root of the hierarchy)
- **Riemannian optimization:** Gradients projected back onto the manifold after each step

### Evaluation
- HR@10: 0.498 | NDCG@10: 0.287 (individual, lowest)
- DR-optimized ensemble weight: 0.004

### Intended Use
Hierarchical preference modeling. Hyperbolic space embeds tree-structured taxonomies (Genre → Sub-genre → Franchise → Film) with exponentially less distortion than Euclidean space. Most valuable for catalog navigation and franchise-aware recommendations.

### Known Limitations
- Numerical instability near the boundary of the Poincaré ball (clamped at 1−1e-5)
- Riemannian optimization is more complex than standard SGD — requires careful learning rate tuning
- Current implementation uses a simplified distance formula; full Riemannian SGD with retraction is not implemented
- Low individual performance suggests the movie catalog hierarchy is not strongly tree-structured

---

## 6. Latent Diffusion Recommender

**File:** `backend/diffusion_recommender.py` | **Weights:** `models/diffusion_recommender.pth`

### Architecture
- Conditioned Denoising Diffusion Probabilistic Model (DDPM)
- Forward process: q(x_t | x_0) = √ᾱ_t × x_0 + √(1−ᾱ_t) × ε
- Reverse process: conditioned on user embedding (user history as guidance signal)
- Denoiser: MLP with GELU activations, LayerNorm, time embedding (SiLU MLP)
- Linear beta schedule: 1e-4 → 0.02 over 100 timesteps
- Generative retrieval: denoised embedding → FAISS nearest-neighbor search

### Training
- **Loss:** MSE between true noise ε and predicted noise ε̂ (standard DDPM objective)
- **Conditioning:** User embedding concatenated with noisy item embedding and time embedding
- **Inspired by:** Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020), applied to recommendation latent space

### Evaluation
- HR@10: 0.521 | NDCG@10: 0.309 (individual)
- DR-optimized ensemble weight: 0.024

### Intended Use
Generative retrieval — bypasses the softmax bottleneck by hallucinating the ideal item embedding and finding the nearest real item via FAISS. Most valuable for serendipitous discovery and cold-start scenarios where no interaction history exists.

### Known Limitations
- 100-step reverse diffusion is computationally expensive at inference (mitigated by using t=0.5 proxy in ensemble scoring)
- Generated embeddings may not align perfectly with the FAISS index space if embeddings were updated after training
- DDPM is slower than DDIM (Denoising Diffusion Implicit Models) — DDIM sampling would reduce inference steps from 100 to ~10

---

## Causal Debiasing

All models are trained with **Inverse Propensity Scoring (IPS)** to correct for popularity bias:

- **Propensity estimation:** Empirical impression frequency with Laplace smoothing
- **IPS-weighted BPR:** Each training sample weighted by 1/propensity (clipped at 10.0)
- **Doubly Robust (DR) weight selection:** Combines direct reward imputation with IPS correction for unbiased ensemble weight optimization
- **Script:** `scripts/causal_debias_training.py`

This follows the methodology of Schnabel et al. "Recommendations as Treatments" (ICML 2016) and Saito et al. "Unbiased Recommender Learning" (WSDM 2020).

---

## Fairness & Compliance

| Metric | Threshold | Status |
|---|---|---|
| Gini Coefficient (popularity bias) | < 0.70 | ✅ Enforced |
| KL Divergence (genre calibration) | Minimized | ✅ Monitored |
| Differential Privacy (ε-DP) | ε=1.0 (Laplace) | ✅ Applied to user embeddings |
| Federated gradient aggregation | DP noise ε=5.0 | ✅ Simulated |
| Safety filter (disliked content) | Hard boundary | ✅ RLSafetyFilter |

See `scripts/fairness_audit.py` and `backend/privacy_preserving_ml.py` for implementation details.

---

## Cross-Architecture Design Rationale

Each model addresses a distinct failure mode of the others:

| Failure Mode | Model That Addresses It |
|---|---|
| Ignores interaction order | SASRec (causal Transformer) |
| Misses multi-hop graph patterns | LightGCN (graph propagation) |
| Linear scoring bottleneck | KAN (learnable edge functions) |
| Static preference assumption | Quantum-Fluid (continuous-time ODE) |
| Euclidean hierarchy distortion | Hyperbolic (Poincaré manifold) |
| Candidate ranking bottleneck | Diffusion (generative retrieval) |
| Popularity bias in training | IPS-weighted BPR + DR weight selection |

No single model addresses all failure modes. The ensemble, with DR-optimized weights, achieves +4.3% NDCG@10 lift over the best individual model.
