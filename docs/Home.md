# Welcome to the APEX AI Recommendation Engine & Data Intelligence Wiki

<p align="center">
  <img src="https://img.shields.io/badge/CI%2FCD-Passing-brightgreen?style=flat&logo=githubactions" alt="CI/CD" />
  <img src="https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen?style=flat" alt="Tests Passed" />
  <img src="https://img.shields.io/badge/Spark-4.2-E25A1C?style=flat&logo=apachespark" alt="Spark 4.2" />
  <img src="https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Bun-1.2-F9F1E1?style=flat&logo=bun" alt="Bun 1.2" />
</p>

This Wiki serves as the definitive engineering manual for the **APEX AI Recommendation System and Unified Data Intelligence Platform**. It provides deep architectural guides, API contracts, PySpark 4.2 declarative pipeline specifications, multi-agent AI flowcharts, and production deployment procedures.

> [!NOTE]
> **APEX Recommendation Engine** is an enterprise-grade platform combining a **6-Model PyTorch Deep Learning Ensemble**, a Databricks-compatible **PySpark 4.2 Delta Lake Data Intelligence Engine**, an **Agentic Multi-Agent AI System**, and an **Adaptive 3-Tier Hardware Serving Architecture**.

---

## 📂 Documentation Navigation Directory

Select a guide from the categorized directories below or use the sidebar menu:

### 🎬 User & Developer Onboarding
* **[Quick Start & Installation Guide](INSTALLATION.md)**: Comprehensive step-by-step setup for Docker Compose and local Bun 1.2 + Python environments.
* **[Developer Onboarding & Contribution Guide](CONTRIBUTING.md)**: Coding standards, pre-commit linter checks, and GitHub pull request procedures.
* **[Beginner Tutorial](BEGINNER_TUTORIAL.md)**: Step-by-step walkthrough for making your first API call and training a custom recommendation model.

### 🌊 Unified Data Intelligence Platform
* **[Unified Data Intelligence Architecture](UNIFIED_DATA_INTELLIGENCE_PLATFORM.md)**: Databricks 1-to-1 open platform mapping matrix, Delta Lake Medallion architecture (Bronze/Silver/Gold), and catalog lineage.
* **[Spark Declarative Pipelines (SDP)](SPARK_DECLARATIVE_PIPELINES.md)**: Declarative pipeline YAML specifications, Lakeflow ingestion, and DAG execution engine.
* **[Spark 4.2 Features Guide](SPARK_42_FEATURES.md)**: Technical breakdown of Spark 4.2 Variant data type, Python Data Source API v2, and vector acceleration.

### 🤖 Agentic AI & Multi-Agent Systems
* **[Agentic AI Architecture & Multi-Agent Orchestrator](AGENTIC_AI.md)**: Design specifications for `RetrievalAgent`, `RecommenderAgent`, `RankingAgent`, and `ExplanationAgent`.
* **[AI Safety & Causal Debiasing](MODERN_SYSTEM_ARCHITECTURE.md)**: Inverse Propensity Score (IPS) weighting, Doubly Robust (DR) estimators, and MMR matrix diversification.

### 🧠 Deep Learning ML Ensemble & Model Registry
* **[6-Model Deep Learning Ensemble](MODEL_CARDS.md)**: Mathematical foundations of SASRec, KAN B-Splines, LightGCN, Neural ODE, Poincaré Hyperbolic, and Latent Continuous Diffusion.
* **[Online Learning & Feedback Loop](ONLINE_LEARNING.md)**: Real-time clickstream ingestion, Redis session queues, and asynchronous mini-batch SGD state updates.

### ⚡ Adaptive Serving & Production Operations
* **[Adaptive 3-Tier Serving Engine](MODERN_SYSTEM_ARCHITECTURE.md)**: Hardware auto-profiling across Tier 1 GPU Ensembling (`~12.5ms`), Tier 2 ONNX INT8 CPU (`~24.8ms`), and Tier 3 SIMD Vector Indexing (`<4.2ms`).
* **[REST API Reference](API_REFERENCE.md)**: Complete endpoint documentation, parameter schemas, and response JSON formats.
* **[gRPC Protobuf Protocol Specification](API_REFERENCE.md)**: High-throughput RPC definitions for real-time recommendation streaming.

---

## 🏛️ High-Level System Architecture

```mermaid
graph TB
    subgraph Client["Client Application Layer"]
        UI[Bun 1.2 + React 19 UI]
        Mobile[Mobile & External Services]
    end

    subgraph Serving["API & Serving Gateway"]
        API[FastAPI Gateway / gRPC Server]
        TD[ServingTierDetector]
        T1["Tier 1: PyTorch GPU Ensemble (~12.5ms)"]
        T2["Tier 2: Quantized ONNX CPU (~24.8ms)"]
        T3["Tier 3: SIMD Vector Index (<4.2ms)"]
    end

    subgraph Agents["Agentic AI Orchestrator"]
        Retrieval["RetrievalAgent"]
        Recommender["RecommenderAgent"]
        Ranking["RankingAgent (KAN B-Splines)"]
        Explainer["ExplanationAgent"]
    end

    subgraph Platform["Unified Data Intelligence Platform"]
        Lakeflow[Lakeflow Ingestion Engine]
        SDP[Spark Declarative Pipelines]
        Bronze[Bronze Delta Lake Raw Store]
        Silver[Silver Delta Lake Cleaned Store]
        Gold[Gold Delta Lake Aggregated Feature Store]
    end

    UI --> API
    Mobile --> API
    API --> TD
    TD --> T1
    TD --> T2
    TD --> T3
    T1 --> Agents
    T2 --> Agents
    T3 --> Agents
    Agents --> Retrieval
    Agents --> Recommender
    Agents --> Ranking
    Agents --> Explainer
    Lakeflow --> Bronze
    SDP --> Silver
    SDP --> Gold
```
