import time

import numpy as np

try:
    import rust_core
except Exception:
    rust_core = None


def synthetic_candidate_data(m=1000, dim=256, k=20):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(m, dim)).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)
    candidate_indices = list(range(k)) + [None] * (m - k)
    candidate_relevance = [float(x) for x in rng.random(k).tolist()] + [0.0] * (m - k)
    return candidate_indices[:k], candidate_relevance[:k], vectors


def bench_mmr(iterations=10):
    idxs, rels, vectors = synthetic_candidate_data(m=1000, dim=256, k=100)
    if rust_core is None:
        print("rust_core not available; bench skipped")
        return 1
    t0 = time.perf_counter()
    for _ in range(iterations):
        order = rust_core.mmr_diversify_rust(idxs, rels, vectors, 10, 0.72)
    t1 = time.perf_counter()
    total = t1 - t0
    avg_ms = (total / iterations) * 1000
    result = {
        "iterations": iterations,
        "total_time_s": total,
        "avg_ms": avg_ms,
    }
    # write result to file for CI consumption
    import json

    with open("bench_result.json", "w") as fh:
        json.dump(result, fh)
    print("BENCH_RESULT_JSON_WRITTEN")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(bench_mmr())
