# Welcome to the APEX AI Recommendation Engine & Data Intelligence Wiki

<p align="center">
  <img src="https://img.shields.io/badge/CI%2FCD-Passing-brightgreen?style=flat&logo=githubactions" alt="CI/CD" />
  <img src="https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen?style=flat" alt="Tests Passed" />
  <img src="https://img.shields.io/badge/Spark-4.2-E25A1C?style=flat&logo=apachespark" alt="Spark 4.2" />
  <img src="https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Bun-1.2-F9F1E1?style=flat&logo=bun" alt="Bun 1.2" />
</p>

This Wiki serves as the definitive engineering manual for the **APEX AI Recommendation System and Unified Data Intelligence Platform**. It provides deep architectural guides, API contracts, PySpark 4.2 declarative pipeline specifications, multi-agent AI flowcharts, and production deployment procedures.

---

## 📂 Documentation Navigation Directory

### 🎬 User & Developer Onboarding
* **[Quick Start & Installation Guide](Quick-Start-Guide)**: Comprehensive setup for Docker Compose and local Bun 1.2 + Python environments.
* **[Developer Onboarding Guide](Developer-Onboarding-Guide)**: Coding standards, pre-commit linter checks, and GitHub pull request procedures.

### 🌊 Unified Data Intelligence Platform
* **[Unified Data Intelligence Architecture](Unified-Data-Intelligence-Platform)**: Databricks 1-to-1 open platform mapping matrix, Delta Lake Medallion architecture (Bronze/Silver/Gold), and catalog lineage.
* **[Spark Declarative Pipelines](Spark-Declarative-Pipelines)**: Declarative pipeline YAML specifications, Lakeflow ingestion, and DAG execution engine.
* **[Spark 4.2 Features Guide](Spark-42-Features)**: Technical breakdown of Spark 4.2 Variant data type, Python Data Source API v2, and vector acceleration.

### 🤖 Agentic AI & Multi-Agent Systems
* **[Agentic AI Architecture](Agentic-AI-Architecture)**: Design specifications for `RetrievalAgent`, `RecommenderAgent`, `RankingAgent`, and `ExplanationAgent`.
* **[Adaptive 3-Tier Serving](Adaptive-3-Tier-Serving)**: Hardware auto-profiling across Tier 1 GPU Ensembling (`~12.5ms`), Tier 2 ONNX INT8 CPU (`~24.8ms`), and Tier 3 SIMD Vector Indexing (`<4.2ms`).

### 🧠 Deep Learning ML Ensemble & API Specs
* **[6-Model PyTorch Ensemble](6-Model-PyTorch-Ensemble)**: SASRec, KAN B-Splines, LightGCN, Neural ODE, Poincaré Hyperbolic, and Latent Diffusion.
* **[API Specifications](API-Reference)**: Complete REST endpoint documentation and gRPC Protobuf definitions.
