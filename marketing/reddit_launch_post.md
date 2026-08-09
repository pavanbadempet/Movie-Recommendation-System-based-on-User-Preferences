# Title: I got tired of toy RecSys tutorials, so I open-sourced an Enterprise Causal Recommendation Engine (SASRec + PySpark 4.2 + Delta Lake + FastAPI)

Hey r/MachineLearning,

I see a lot of "Recommendation System" tutorials that just train a collaborative filtering matrix in a Jupyter Notebook and call it a day. That leaves out the hardest parts of production ML: serving latency, causal debiasing, offline-to-online feature drift, and async feedback loops.

I decided to open-source **APEX**, a full production-grade recommendation platform that I’ve been building. It’s engineered to run entirely on open standards (no vendor lock-in).

**GitHub:** [https://github.com/pavanbadempet/Movie-Recommendation-System](https://github.com/pavanbadempet/Movie-Recommendation-System)

### The Architecture:
1. **6-Model PyTorch Ensemble**: We don’t just use one model. It dynamically ensembles **SASRec** (Sequential Transformers), **KAN** (Kolmogorov-Arnold B-Splines), **LightGCN** (Graph), **Neural ODE** (Temporal dynamics), **Poincaré Hyperbolic**, and **Latent Diffusion**.
2. **Adaptive 3-Tier Serving**: Since GPU instances are expensive, the FastAPI backend profiles hardware at boot. 
   - Tier 1: Full PyTorch GPU Ensemble (~12.5ms latency).
   - Tier 2: INT8 Quantized ONNX on CPU (~24.8ms).
   - Tier 3: FAISS / SIMD Vector Indexing (<4.2ms). 
3. **Unified Data Intelligence**: The ETL is built on **PySpark 4.2** using a **Delta Lake Medallion Architecture** (Bronze/Silver/Gold) with Lakeflow declarative pipelines. 
4. **Causal Debiasing**: We use Inverse Propensity Scoring (IPS) and Doubly Robust estimators to counter popularity bias, ensuring the long-tail movies get surfaced instead of just recommending the Avengers to everyone.

I also recently added a full **Agentic Multi-Agent AI architecture** that routes the inference pipelines dynamically using LLM reasoning.

If you are a Data Engineer or ML Ops practitioner, I’d love for you to rip this apart, check out the PySpark Delta pipelines, and tell me where I can improve the latency or architecture.

Code and interactive docs are all in the repo! ⭐️ Star it if you find the reference architecture useful for your own deployments!
