**Title:** Show HN: APEX – Open-Source Enterprise Causal Recommendation Engine

**URL:** https://github.com/pavanbadempet/Movie-Recommendation-System

**First Comment (OP):**

Hi HN,

Most recommendation system examples online stop at a Jupyter notebook. I wanted to open-source a complete, production-ready system that tackles the real-world engineering challenges: serving latency, causal popularity debiasing, and big data ETL.

**APEX** is a recommendation engine built on a 6-model PyTorch ensemble (SASRec, KAN B-Splines, LightGCN, Neural ODE, Poincaré Hyperbolic, and Latent Diffusion). 

To make it actually deployable without a massive cloud bill, the FastAPI backend implements an **Adaptive 3-Tier Serving Engine**. It profiles your hardware at container boot: if it finds GPUs, it runs the full ensemble (~12.5ms). If it's on a CPU, it routes to an INT8 Quantized ONNX runtime (~24.8ms). If resources are fully constrained, it degrades gracefully to a FAISS SIMD vector index (<4.2ms).

The data pipeline uses PySpark 4.2 and a Delta Lake Medallion architecture (Bronze -> Silver -> Gold), mimicking Databricks-grade workloads but completely on open-source standards with zero vendor lock-in.

I'd love feedback from backend engineers and ML practitioners on the architecture, particularly the async feedback loop and ONNX quantization strategies. Happy to answer any questions!
