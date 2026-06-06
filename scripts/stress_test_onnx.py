"""
Benchmarking script to prove ONNX Runtime latency supremacy over native PyTorch.
"""

import logging
from pathlib import Path
import time

import numpy as np
import torch

from backend.models.mmoe_ranker import MMoERanker
from backend.serving.onnx_engine import get_onnx_engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("============================================================")
    logger.info("PHASE 8: ONNX vs PyTorch Latency Stress Test (CPU)")
    logger.info("============================================================")

    num_users_mmoe = 611
    num_items_mmoe = 193610

    # 1. Load PyTorch Model
    mmoe_pt = MMoERanker(user_vocab_size=num_users_mmoe, item_vocab_size=num_items_mmoe)
    mmoe_path = Path("models/mmoe_ranker.pth")
    if mmoe_path.exists():
        mmoe_pt.load_state_dict(torch.load(mmoe_path, map_location="cpu", weights_only=True))
    mmoe_pt.eval()

    # 2. Load ONNX Model
    onnx = get_onnx_engine()

    # Generate Dummy Batch
    batch_size = 200  # Ranking top 200 candidates
    u_pt = torch.randint(0, num_users_mmoe, (batch_size,))
    i_pt = torch.randint(0, num_items_mmoe, (batch_size,))

    u_np = u_pt.numpy()
    i_np = i_pt.numpy()

    iterations = 100

    # WARMUP
    with torch.no_grad():
        for _ in range(10):
            mmoe_pt(u_pt, i_pt)
            onnx.predict_mmoe(u_np, i_np)

    # BENCHMARK PYTORCH
    pt_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            mmoe_pt(u_pt, i_pt)
            pt_times.append((time.perf_counter() - start) * 1000)

    # BENCHMARK ONNX
    onnx_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        onnx.predict_mmoe(u_np, i_np)
        onnx_times.append((time.perf_counter() - start) * 1000)

    logger.info(f"Target Workload: Re-Ranking {batch_size} candidates.")
    logger.info(f"Native PyTorch P99 Latency: {np.percentile(pt_times, 99):.2f} ms")
    logger.info(f"ONNX Runtime P99 Latency:  {np.percentile(onnx_times, 99):.2f} ms")
    logger.info(f"Speedup Factor: {np.mean(pt_times) / np.mean(onnx_times):.2f}x")

    if np.percentile(onnx_times, 99) < 50:
        logger.info("✅ SUCCESS: ONNX P99 Latency is Sub-50ms!")
    else:
        logger.warning("⚠️ WARNING: ONNX P99 Latency exceeds 50ms constraint.")


if __name__ == "__main__":
    main()
