# Online Learning in APEX

APEX closes the feedback loop between production user behavior and the recommendation models
in real time, without requiring a full retraining cycle. This document describes the
architecture, design decisions, and operational characteristics of the online learning system.

---

## Overview

When a user clicks on a recommendation or submits a rating, three independent background
threads receive the event simultaneously and apply incremental gradient updates to their
respective models:

```
Live Event (click / rating)
         │
         ▼
 OnlineLearningCoordinator.enqueue(event)
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
LightGCN  SASRec         KAN
OnlineLearner  SASRecOnlineLearner  KANOnlineLearner
 (BPR)      (sequential BPR)      (Fourier BPR)
    │         │                    │
    ▼         ▼                    ▼
lightgcn_online.pth  sasrec_online.pth  kan_online.pth
```

Each learner operates completely independently — a crash or queue overflow in one
does not affect the others or the serving path.

---

## Why All Three Models Matter

| Model | DR Weight | Benefit from Online Learning |
|---|---|---|
| SASRec | **0.659** | Session sequences adapt to current intent within minutes. The dominant model getting live signal has the largest impact on recommendation quality. |
| KAN | **0.298** | Fourier edge functions adapt to production click/rating distributions that differ from offline training data. |
| LightGCN | 0.005 | Graph embeddings accumulate user-item interaction signal over time. Weight expected to grow as event store matures. |

Before online learning, 95.7% of the ensemble's effective weight (SASRec + KAN) was
frozen at offline training values and never updated from production behavior.

---

## Architecture

### OnlineLearningCoordinator

**File:** `backend/learning/online_learning_coordinator.py`

The coordinator is the single entry point for all live events. It:
- Wraps all three learners under one lifecycle (`start()` / `stop()`)
- Fans out each event to all three queues in a single non-blocking call
- Exposes a `status()` method with per-learner thread health and queue depths
- Is instantiated in `main.py` lifespan for **Tier 1 only** (GPU or high-RAM CPU)

```python
coordinator = OnlineLearningCoordinator(engine=apex_engine)
coordinator.start()

# From serving path (recommendation_routes.py):
coordinator.enqueue({
    "event_type": "rating",
    "user_id": "123",
    "movie_id": 550,
    "rating": 5.0
})
```

### SASRecOnlineLearner

**File:** `backend/learning/sasrec_online_learner.py`

Updates SASRec from live events using BPR-style contrastive loss on
`(session_sequence → positive_item, negative_item)` triples.

**What gets updated:**
- `SASRec.item_emb` (item embedding table)
- `SASRec.attention_layers[-1]` (last attention block)
- `SASRec.forward_layers[-1]` (last feed-forward block)

Full backprop through all Transformer blocks is too slow for online updates.
Fine-tuning only the last block is empirically effective and keeps per-step
latency under 5 ms.

**Session sequence:** The learner fetches the user's current session from
`backend/realtime_feature_updater.py` (sub-millisecond), then falls back to
the background event index, then falls back to an empty sequence (cold-start).

### KANOnlineLearner

**File:** `backend/learning/kan_online_learner.py`

Updates KAN's Fourier edge functions from live events.

**What gets updated:**
- `NaiveFourierKANLayer.fourier_coeffs_sin` (all three layers)
- `NaiveFourierKANLayer.fourier_coeffs_cos` (all three layers)
- `NaiveFourierKANLayer.base_weight` (all three layers)

**Critical design choice:** User and item embeddings are sourced from LightGCN
and passed as **detached tensors** — no gradient flows back into LightGCN. This
decouples KAN's updates from LightGCN's updates, preventing conflicting gradient
signals on shared embedding parameters.

### OnlineLearner (LightGCN)

**File:** `backend/learning/online_learner.py`

The original learner, retained for backward compatibility. Updates LightGCN's
user and item embedding tables via IPS-weighted BPR loss. Uses a persistent
Adam optimizer so momentum state accumulates correctly across gradient steps
(creating a new optimizer per batch degrades to plain SGD).

---

## Event Classification

All three learners apply the same event weighting scheme:

| Event Type | Condition | Interaction Weight | Action |
|---|---|---|---|
| `rating` | rating ≥ 4.0 | +1.0 | Strong positive — BPR positive item |
| `rating` | rating < 2.5 | -0.5 | Negative — BPR roles reversed |
| `rating` | 2.5 ≤ rating < 4.0 | — | Neutral — silently dropped |
| `click` | any | +0.3 | Weak positive — BPR positive item |
| `view`, `search`, other | any | — | Dropped — not actionable |

For negative interactions (weight < 0), BPR roles are swapped: the disliked item
becomes the negative sample, and a random item becomes the positive. This teaches
the model to rank disliked content below random items.

---

## Queue and Thread Design

Each learner maintains a bounded in-memory queue:

| Learner | Queue Size | Batch Size | Drop Policy |
|---|---|---|---|
| `OnlineLearner` | 10,000 | 32 | Oldest event dropped on overflow |
| `SASRecOnlineLearner` | 5,000 | 16 | Oldest event dropped on overflow |
| `KANOnlineLearner` | 5,000 | 32 | Oldest event dropped on overflow |

All threads are daemon threads — they terminate when the serving process exits.
The coordinator's `stop()` method signals all threads and waits up to 5 seconds
for graceful shutdown.

---

## Checkpointing

Each learner saves model weights periodically to prevent loss on process restart:

| Learner | Checkpoint Path | Interval |
|---|---|---|
| `OnlineLearner` | `models/lightgcn_online.pth` | Every 1,000 events |
| `SASRecOnlineLearner` | `models/sasrec_online.pth` | Every 500 events |
| `KANOnlineLearner` | `models/kan_online.pth` | Every 750 events |

Checkpoints are loaded automatically at startup if they exist, allowing the
online learner to resume from where it left off after a process restart.

---

## Monitoring

The coordinator's `status()` method is included in the `/v1/platform/slo` response:

```json
{
  "online_learning": {
    "started": true,
    "learners": {
      "lightgcn": {
        "thread_alive": true,
        "events_processed": 4821,
        "queue_depth": 3
      },
      "sasrec": {
        "thread_alive": true,
        "events_processed": 4821,
        "queue_depth": 2
      },
      "kan": {
        "thread_alive": true,
        "events_processed": 4821,
        "queue_depth": 2
      }
    }
  }
}
```

A `thread_alive: false` for any learner indicates a background thread crash.
The coordinator attempts one automatic restart at startup; if threads fail to
start after retry, online learning is disabled and the serving path falls back
to the last checkpointed weights.

---

## Tier Availability

| Tier | Online Learning |
|---|---|
| **Tier 1** (GPU or RAM ≥ 16 GB) | ✅ All three learners active |
| **Tier 2** (ONNX CPU) | ❌ ONNX models do not support gradient updates |
| **Tier 3** (FAISS only) | ❌ No neural ensemble loaded |

---

## Property-Based Tests

`tests/test_online_learning_coordinator.py` contains 30 property-based and
unit tests covering:

- Lifecycle (start/stop/idempotency)
- Event routing fan-out invariant: `click → all 3 queues receive exactly 1 event`
- Neutral rating (3.0) dropped by all learners
- Gradient step mutates correct parameters (SASRec item emb, KAN Fourier coeffs)
- KAN learner does **not** mutate LightGCN embeddings (decoupling invariant)
- End-to-end gradient flow: after 6 positive ratings, all 3 models show measurable weight changes

Run:
```bash
python -m pytest tests/test_online_learning_coordinator.py -v
```

---

## References

- Schnabel et al. "Recommendations as Treatments" (ICML 2016) — IPS-weighted BPR
- Koren "Collaborative Filtering with Temporal Dynamics" (KDD 2009) — temporal online learning
- He et al. "LightGCN: Simplifying GCN for Recommendation" (SIGIR 2020)
- Kang & McAuley "Self-Attentive Sequential Recommendation" (ICDM 2018)
- Liu et al. "KAN: Kolmogorov-Arnold Networks" (arXiv 2024)
