# Installation Guide - Nova

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Local Development
```bash
# Install dependencies
python manage.py setup

# Run API + Streamlit
python manage.py run
```

Access at http://localhost:8000 (API) and http://localhost:8501 (Streamlit)

### 3. (Optional) Kaggle ETL
```bash
# Requires Kaggle account
python notebooks/kaggle_etl_pipeline.py
```

## Docker Setup

### Local with Docker
```bash
docker compose up --build
```

Access at http://localhost:8000

## System Requirements

### Minimum
- Python 3.10+
- 2GB RAM
- 1GB storage

### Recommended
- Python 3.12
- 4GB+ RAM
- PySpark (for ETL)
- PostgreSQL (for events)
- Kafka (for streaming)

## Environment Setup

### 1. Install Python
```bash
python3.12 --version  # Must be 3.10+
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Serving (required)
pip install -r requirements.txt

# ETL (optional but recommended)
pip install -r requirements-etl.txt

# Development
pip install -r requirements-dev.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env as needed
```

## Configuration

### Environment Variables

```bash
# Optional: API key for tenant mode
NOVA_API_KEYS=secret-key:demo-media-co:tmdb-movies:free

# Optional: Event storage (default: local JSONL)
NOVA_EVENT_STORE=postgres
NOVA_EVENT_DATABASE_URL=postgresql://user:password@localhost:5432/nova

# Optional: Distributed cache for failover
UPSTASH_REDIS_REST_URL=https://your-cache.upstash.io
UPSTASH_REDIS_REST_TOKEN=your-token
```

## First Run

### 1. Download Movie Data
```bash
# Kaggle (automatic)
python notebooks/kaggle_etl_pipeline.py

# Or download manually and place in data/
```

### 2. Run ETL Pipeline
```bash
python etl/pyspark_etl.py --sink delta --tenant-id demo-media-co
```

### 3. Build Embeddings & Artifacts
```bash
Embeddings build automatically on first API call
```

### 4. Access UI
- **React UI**: http://localhost:8000/ui/
- **Streamlit Console**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## Production Deployment

### Render (Free Tier)
```bash
# render.yaml included
git push origin main  # Auto-deploys
```

### HuggingFace Spaces
```bash
# Docker deployment
hf_hub_upload_to_spaces nova/
```

### AWS
- ECS for containerized deployment
- S3 for artifact storage
- RDS for PostgreSQL
- ElastiCache for Redis

## Troubleshooting

### Port Already in Use
```bash
uvicorn backend.main:app --port 8001
```

### PySpark Issues
```bash
pip install pyspark --upgrade
```

### Memory Issues
```bash
# Reduce batch size in config
# Or increase available RAM
```

---

See [USAGE.md](USAGE.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.