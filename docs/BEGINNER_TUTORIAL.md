# Beginner Tutorial: Building Your First Recommendation

## Prerequisites

- Basic Python knowledge
- Understanding of functions and classes
- 4GB RAM available
- Python 3.11+ installed

## Lesson 1: Understanding the Basics (30 minutes)

### What is a Recommendation System?

A recommendation system suggests items to users based on their preferences. For movies, it might suggest "The Dark Knight" because you liked "Inception".

### Key Concepts

**Collaborative Filtering**: "Users who liked X also liked Y"
**Content-Based Filtering**: "Because you liked action movies, here's more action"
**Hybrid Approach**: Combines both methods

## Lesson 2: Setting Up Your Environment (15 minutes)

### Step 1: Clone and Install

```bash
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure for Learning

```bash
# Copy environment template
cp .env.example .env

# Edit .env file to use minimal configuration
# Add these lines:
NOVA_SERVING_TIER=tier3
NOVA_DISABLE_MODEL_DOWNLOADS=1
NOVA_DISABLE_ONLINE_LEARNING=1
```

### Step 3: Build Simple Artifacts

```bash
# This creates basic search indices (takes 2-3 minutes)
python scripts/rebuild_serving_artifacts.py
```

## Lesson 3: Making Your First Recommendation (20 minutes)

### Start the Server

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Test the API

Open a new terminal and try:

```bash
# Get recommendations for a movie
curl "http://127.0.0.1:8000/v1/recommendations/id/1"

# Search for movies
curl "http://127.0.0.1:8000/v1/search?q=action"

# Check system health
curl "http://127.0.0.1:8000/v1/platform/status"
```

### Understanding the Response

```json
{
  "recommendations": [
    {
      "id": 234,
      "title": "The Matrix",
      "genres": ["Action", "Sci-Fi"],
      "score": 0.85,
      "explanation": "Recommended based on your preferences"
    }
  ]
}
```

## Lesson 4: Modifying the Code (45 minutes)

### Exercise 1: Change the Number of Recommendations

Open `backend/api/recommendation_routes.py` and find the recommendation endpoint:

```python
@router.get("/id/{movie_id}")
async def get_recommendations(movie_id: int, limit: int = 10):
    # Change the default limit from 10 to 5
    recommendations = recommender.get_recommendations(movie_id, limit=limit)
    return {"recommendations": recommendations}
```

**Your Task**: Change the default from 10 to 5 and test it.

### Exercise 2: Add a Simple Filter

Add a genre filter to the search endpoint:

```python
@router.get("/search")
async def search_movies(q: str, genre: str = None):
    results = search_service.search(q)

    # Add this filter
    if genre:
        results = [m for m in results if genre in m.get("genres", [])]

    return {"results": results}
```

**Your Task**: Test this with `curl "http://127.0.0.1:8000/v1/search?q=action&genre=Sci-Fi"`

### Exercise 3: Understand the Retrieval Pipeline

Open `backend/pipeline/retrieval_pipeline.py`:

```python
class RetrievalPipeline:
    def retrieve(self, user_id: str, movie_id: int, limit: int = 100):
        # Step 1: Get similar movies from FAISS index
        faiss_results = self.faiss_index.search(movie_id, limit)

        # Step 2: Get TF-IDF matches
        tfidf_results = self.tfidf_index.search(movie_id, limit)

        # Step 3: Combine and deduplicate
        combined = self._combine_results(faiss_results, tfidf_results)

        return combined[:limit]
```

**Your Task**: Add a print statement to see what's being returned:

```python
print(f"Retrieved {len(combined)} candidates for movie {movie_id}")
```

## Lesson 5: Understanding the Model (30 minutes)

### The SASRec Model

Open `backend/models/sasrec.py` - this is the main recommendation model:

```python
class SASRecModel(nn.Module):
    def __init__(self, num_items, embedding_dim=128):
        super().__init__()
        # Item embeddings: convert movie IDs to vectors
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Transformer: learns patterns in user sequences
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4),
            num_layers=2
        )

    def forward(self, user_sequence):
        # Convert sequence to embeddings
        embeddings = self.item_embeddings(user_sequence)

        # Apply transformer to learn patterns
        sequence_representation = self.transformer(embeddings)

        # Predict next item
        scores = torch.matmul(sequence_representation, self.item_embeddings.weight.t())

        return scores
```

### Key Concepts

**Embeddings**: Numbers that represent movies (similar movies have similar numbers)
**Transformer**: Learns patterns like "after watching action movies, users want more action"
**Sequence**: The order of movies a user watched matters

## Lesson 6: Adding Your Own Feature (60 minutes)

### Exercise: Add a Year Filter

**Step 1**: Add year to the movie data structure

Open `backend/response_models.py`:

```python
class Movie(BaseModel):
    id: int
    title: str
    genres: list[str]
    year: int  # Add this field
    poster: str | None = None
```

**Step 2**: Modify the retrieval to filter by year

Open `backend/pipeline/retrieval_pipeline.py`:

```python
def retrieve(self, user_id: str, movie_id: int, limit: int = 100, min_year: int = None):
    results = self._get_all_candidates(movie_id, limit)

    # Add year filter
    if min_year:
        results = [m for m in results if m.get("year", 0) >= min_year]

    return results
```

**Step 3**: Update the API endpoint

Open `backend/api/recommendation_routes.py`:

```python
@router.get("/id/{movie_id}")
async def get_recommendations(
    movie_id: int,
    limit: int = 10,
    min_year: int = None  # Add this parameter
):
    recommendations = retrieval_pipeline.retrieve(
        user_id="test",
        movie_id=movie_id,
        limit=limit,
        min_year=min_year
    )
    return {"recommendations": recommendations}
```

**Step 4**: Test it

```bash
curl "http://127.0.0.1:8000/v1/recommendations/id/1?min_year=2010"
```

## Lesson 7: Testing Your Changes (30 minutes)

### Write a Simple Test

Create `tests/test_my_changes.py`:

```python
import pytest
from backend.pipeline.retrieval_pipeline import RetrievalPipeline

def test_year_filter():
    pipeline = RetrievalPipeline()

    # Mock data
    movies = [
        {"id": 1, "title": "Old Movie", "year": 1990},
        {"id": 2, "title": "New Movie", "year": 2020},
    ]

    # Test filtering
    filtered = [m for m in movies if m["year"] >= 2000]

    assert len(filtered) == 1
    assert filtered[0]["title"] == "New Movie"
    print("✓ Year filter works correctly")

if __name__ == "__main__":
    test_year_filter()
```

Run your test:

```bash
python tests/test_my_changes.py
```

## Lesson 8: Understanding the Full Pipeline (45 minutes)

### The Three-Stage Pipeline

```
User Request
    ↓
Stage 1: Retrieval (Get 100 candidates)
    ├─ FAISS: Vector similarity search
    ├─ TF-IDF: Text-based search
    └─ Knowledge Graph: Genre/actor relationships
    ↓
Stage 2: Ranking (Score candidates)
    ├─ SASRec: Sequential patterns
    ├─ LightGCN: Graph collaborative filtering
    └─ KAN: Tabular feature learning
    ↓
Stage 3: Reranking (Diversity & fairness)
    ├─ MMR: Maximal Marginal Relevance
    ├─ Genre diversity: Ensure variety
    └─ Fairness: Prevent bias
    ↓
Final Recommendations (Top 10)
```

### Visualizing the Flow

Open `backend/main.py` and trace a request:

1. **API Layer** (`recommendation_routes.py`): Receives HTTP request
2. **Pipeline Layer** (`retrieval_pipeline.py`): Gets candidates
3. **Model Layer** (`sasrec.py`): Scores candidates
4. **Response Layer** (`response_models.py`): Formats JSON

## Lesson 9: Common Issues and Solutions (20 minutes)

### Issue 1: Out of Memory

**Problem**: Server crashes with "CUDA out of memory"

**Solution**: Use Tier 3 mode
```bash
export NOVA_SERVING_TIER=tier3
```

### Issue 2: Slow Recommendations

**Problem**: Recommendations take >5 seconds

**Solution**: Reduce candidate count
```python
# In retrieval_pipeline.py
def retrieve(self, ..., limit: int = 50):  # Was 100
```

### Issue 3: Poor Recommendations

**Problem**: Recommendations don't make sense

**Solution**: Check your data
```python
# Verify movie data is loaded correctly
print(f"Total movies: {len(feature_store.movies)}")
print(f"Sample movie: {feature_store.movies[0]}")
```

## Lesson 10: Next Steps (15 minutes)

### What You've Learned

- ✅ How to set up the development environment
- ✅ How to make API requests
- ✅ How to modify the code
- ✅ How to add new features
- ✅ How to write tests
- ✅ How the pipeline works

### Continue Learning

1. **Read the Architecture Decisions**: `docs/ARCHITECTURE_DECISIONS.md`
2. **Study the Models**: `backend/models/` directory
3. **Explore Advanced Features**: Real-time learning, causal debiasing
4. **Contribute**: See `CONTRIBUTING.md`

### Advanced Topics

- **Ensemble Methods**: How to combine multiple models
- **Causal Inference**: Why popularity bias matters
- **Online Learning**: Updating models in real-time
- **Differential Privacy**: Protecting user data

## Quick Reference

### Essential Files

| File | Purpose |
|------|---------|
| `backend/main.py` | Server entry point |
| `backend/api/recommendation_routes.py` | API endpoints |
| `backend/pipeline/retrieval_pipeline.py` | Candidate generation |
| `backend/models/sasrec.py` | Main recommendation model |
| `backend/response_models.py` | Data structures |

### Essential Commands

```bash
# Start server
uvicorn backend.main:app --reload

# Run tests
pytest tests/ -v

# Rebuild artifacts
python scripts/rebuild_serving_artifacts.py

# Check code quality
ruff check backend/
ruff format backend/
```

### Environment Variables

```bash
NOVA_SERVING_TIER=tier3          # Use minimal resources
NOVA_DISABLE_MODEL_DOWNLOADS=1    # Don't download models
NOVA_DISABLE_ONLINE_LEARNING=1   # Disable real-time updates
NOVA_LOG_LEVEL=DEBUG             # Verbose logging
```

## Congratulations!

You've completed the beginner tutorial. You now understand:
- How recommendation systems work
- How to modify the APEX codebase
- How to add your own features
- How to test your changes

Ready for more? Check out the advanced documentation in the `docs/` folder!
