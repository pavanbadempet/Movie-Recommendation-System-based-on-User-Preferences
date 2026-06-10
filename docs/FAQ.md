# Frequently Asked Questions - Nova

## General

**Q: What is Nova?**
A: Nova is a B2B recommendation and semantic discovery platform for content businesses. It's not a streaming app clone—it's the infrastructure layer for media companies, OTT platforms, and education providers.

**Q: Who should use Nova?**
A: Regional OTT platforms, education libraries, creator marketplaces, digital publishers, and anyone needing advanced recommendation infrastructure.

**Q: Is this production-ready?**
A: Yes! Nova is designed for production use with proper deployment configuration.

## Technical

**Q: What are the system requirements?**
A: Python 3.10+, 2GB RAM minimum. Optional: PySpark for ETL, Kafka for events. See [INSTALLATION.md](INSTALLATION.md).

**Q: Can I use without Spark?**
A: Yes, but ETL operations will be slower. Spark is recommended for large catalogs.

**Q: Does it support real-time recommendations?**
A: Yes, via hybrid search. Behavior-based personalization updates with streaming events.

## Data & Privacy

**Q: How is customer data protected?**
A: Optional API-key tenant mode, audit logging, and SCD Type 2 history tracking.

**Q: What data formats are supported?**
A: CSV (batch upload), JSON (API), and streaming events.

**Q: Can I export my data?**
A: Yes, all data is in open formats (Parquet, Delta Lake tables).

## Deployment

**Q: What's the easiest way to deploy?**
A: Docker Compose for local, or Render/HuggingFace for free hosting.

**Q: Can I deploy to AWS?**
A: Yes, but infrastructure configuration is your responsibility.

**Q: Do you provide hosting?**
A: No, Nova is self-hosted. Use free tiers (Render, HuggingFace) or your infrastructure.

## Features

**Q: What ML models are included?**
A: Hybrid search (sparse + dense), optional learned ranker, and behavior personalization.

**Q: Can I use custom embeddings?**
A: Yes, implement the embedding interface in `backend/recommender.py`.

**Q: How accurate are recommendations?**
A: Depends on your data quality. Nova includes benchmarking tools to measure performance.

---

**More questions?** Visit [Discussions](https://github.com/pavanbadempet/Movie-Recommendation-System/discussions)!
