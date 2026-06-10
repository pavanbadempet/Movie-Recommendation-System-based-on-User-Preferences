# Troubleshooting - Nova

## Common Issues

### Installation

**PySpark won't install**
```bash
pip install pyspark --upgrade
# Or use conda
conda install pyspark
```

**Java not found (PySpark)**
```bash
# Install Java
# Ubuntu: sudo apt-get install openjdk-11-jdk
# macOS: brew install openjdk@11
export JAVA_HOME=/path/to/java
```

### Runtime

**Port 8000 already in use**
```bash
uvicorn backend.main:app --port 8001
```

**Embeddings won't build**
```bash
# Clear cache and rebuild
rm -rf data/artifacts/
python manage.py setup
```

### ETL

**Spark DataFrame operations slow**
```bash
# Enable adaptive query execution
export SPARK_LOCAL_IP=127.0.0.1
# Increase executor memory
export SPARK_EXECUTOR_MEMORY=4g
```

**Out of memory during ETL**
```bash
# Reduce batch size
# Or increase system RAM
# Use PySpark partitioning
```

### API

**Search returns no results**
- Check embeddings built successfully
- Verify catalog data loaded
- Try /v1/diagnostics endpoint

**Recommendations seem random**
- More behavior events needed
- Check ranker loaded correctly
- Review /v1/diagnostics output

### Deployment

**Cold starts on Render**
```bash
# Use keep-alive workflow
# See .github/workflows/
```

**CORS errors**
```bash
# Update ALLOWED_ORIGINS in config
```

---

See [FAQ.md](FAQ.md) or [Discussions](https://github.com/pavanbadempet/Movie-Recommendation-System/discussions)!
