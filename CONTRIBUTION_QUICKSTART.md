# Contribution Quick-Start Guide

## 5-Minute Setup for Contributors

### Prerequisites
- Git installed
- Python 3.12+ installed
- 4GB RAM available
- Basic Python knowledge

### Step 1: Fork and Clone (2 minutes)

```bash
# Fork the repository on GitHub first
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# Add upstream remote
git remote add upstream https://github.com/pavanbadempet/Movie-Recommendation-System.git
```

### Step 2: Create Virtual Environment (1 minute)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 3: Configure for Development (1 minute)

```bash
cp .env.example .env

# Add to .env:
NOVA_SERVING_TIER=tier3
NOVA_DISABLE_MODEL_DOWNLOADS=1
NOVA_LOG_LEVEL=DEBUG
```

### Step 4: Verify Setup (1 minute)

```bash
# Run linter
python -m ruff check backend/ tests/

# Run quick tests
python -m pytest tests/test_api.py -v

# Start server
uvicorn backend.main:app --reload
```

✅ You're ready to contribute!

## Finding Good First Issues

### Easy Issues (Beginner)
- Add missing docstrings
- Fix typos in documentation
- Add unit tests for simple functions
- Improve error messages

### Medium Issues (Intermediate)
- Add new API endpoints
- Implement new features
- Refactor code for clarity
- Add integration tests

### Hard Issues (Advanced)
- Implement new models
- Optimize performance
- Add advanced ML features
- Architectural improvements

## Making Your First Contribution

### Example: Adding a Missing Docstring

**1. Find code without documentation**

```bash
# Search for functions without docstrings
grep -r "def.*:" backend/ | grep -v "def _" | head -20
```

**2. Add docstring**

```python
# Before
def get_recommendations(movie_id: int):
    return recommender.recommend(movie_id)

# After
def get_recommendations(movie_id: int):
    """
    Get movie recommendations for a given movie.

    Args:
        movie_id: The ID of the movie to get recommendations for.

    Returns:
        List of recommended movie dictionaries.
    """
    return recommender.recommend(movie_id)
```

**3. Test your change**

```bash
# Run tests
python -m pytest tests/ -v

# Run linter
python -m ruff check backend/
```

**4. Commit and push**

```bash
git add backend/api/recommendation_routes.py
git commit -m "Add docstring to get_recommendations function"
git push origin your-branch-name
```

**5. Create Pull Request**

- Go to GitHub
- Click "New Pull Request"
- Describe your change
- Link to any related issues

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

```bash
# Edit files
# Run tests frequently
python -m pytest tests/ -v

# Check code quality
python -m ruff check backend/
python -m ruff format backend/
```

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=backend --cov-report=term-missing
```

### 4. Commit Changes

```bash
git add .
git commit -m "Clear description of your changes"
```

**Commit message format:**
```
type(scope): subject

body

footer
```

**Types:** feat, fix, docs, style, refactor, test, chore

**Example:**
```
feat(api): add genre filter to search endpoint

Users can now filter search results by genre using the
?genre= parameter.

Closes #123
```

### 5. Sync with Upstream

```bash
git fetch upstream
git rebase upstream/main
```

### 6. Push and Create PR

```bash
git push origin your-branch-name
# Create PR on GitHub
```

## Code Style Guidelines

### Python Code Style

We use **Ruff** for linting and formatting:

```bash
# Check code
python -m ruff check backend/

# Format code
python -m ruff format backend/

# Auto-fix issues
python -m ruff check --fix backend/
```

### Key Style Rules

- Line length: 120 characters
- Use double quotes for strings
- Use type hints for function signatures
- Add docstrings to all public functions
- Order imports: stdlib → third-party → local

### Example Good Code

```python
"""
Recommendation service module.

Provides movie recommendation functionality using ensemble models.
"""

from typing import List, Optional
import logging

from backend.models.sasrec import SASRecModel
from backend.pipeline.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating movie recommendations."""

    def __init__(
        self,
        model: SASRecModel,
        retrieval_pipeline: RetrievalPipeline
    ):
        """
        Initialize the recommendation service.

        Args:
            model: The SASRec model for scoring.
            retrieval_pipeline: Pipeline for candidate retrieval.
        """
        self.model = model
        self.retrieval_pipeline = retrieval_pipeline

    def get_recommendations(
        self,
        movie_id: int,
        limit: int = 10
    ) -> List[dict]:
        """
        Get recommendations for a movie.

        Args:
            movie_id: The movie ID to get recommendations for.
            limit: Maximum number of recommendations to return.

        Returns:
            List of recommended movie dictionaries.
        """
        try:
            candidates = self.retrieval_pipeline.retrieve(movie_id, limit * 2)
            scored = self.model.score(candidates)
            return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
```

## Testing Guidelines

### Writing Tests

```python
import pytest
from backend.pipeline.retrieval_pipeline import RetrievalPipeline

class TestRetrievalPipeline:
    """Test suite for RetrievalPipeline."""

    def test_retrieve_returns_candidates(self):
        """Test that retrieve returns candidate movies."""
        pipeline = RetrievalPipeline()
        candidates = pipeline.retrieve(movie_id=1, limit=10)

        assert len(candidates) > 0
        assert all("id" in c for c in candidates)

    def test_retrieve_with_limit(self):
        """Test that limit parameter works correctly."""
        pipeline = RetrievalPipeline()
        candidates = pipeline.retrieve(movie_id=1, limit=5)

        assert len(candidates) <= 5

    def test_retrieve_empty_movie_id(self):
        """Test behavior with invalid movie ID."""
        pipeline = RetrievalPipeline()
        candidates = pipeline.retrieve(movie_id=999999, limit=10)

        # Should return empty list or handle gracefully
        assert isinstance(candidates, list)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::test_get_recommendations -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

## Common Contribution Tasks

### Adding a New API Endpoint

1. **Add route in `backend/api/recommendation_routes.py`:**

```python
@router.get("/popular")
async def get_popular_movies(limit: int = 10):
    """
    Get most popular movies.

    Args:
        limit: Number of movies to return.

    Returns:
        List of popular movies.
    """
    popular = feature_store.get_popular_movies(limit)
    return {"movies": popular}
```

2. **Add test in `tests/test_api.py`:**

```python
def test_get_popular_movies():
    """Test popular movies endpoint."""
    response = client.get("/v1/recommendations/popular?limit=5")
    assert response.status_code == 200
    assert len(response.json()["movies"]) <= 5
```

3. **Update documentation in `README.md`:**

```markdown
### Popular Movies
- `GET /v1/recommendations/popular`: Get most popular movies
```

### Adding a New Feature

1. **Create feature branch**
2. **Implement feature in appropriate module**
3. **Add comprehensive tests**
4. **Update documentation**
5. **Submit PR with description**

### Fixing a Bug

1. **Create issue describing the bug**
2. **Create branch to fix it**
3. **Write test that reproduces bug**
4. **Fix the bug**
5. **Verify test passes**
6. **Submit PR**

## Getting Help

### Ask Questions

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Code Comments**: Add `# TODO:` or `# FIXME:` for unclear code

### Useful Resources

- **Architecture Decisions**: `docs/ARCHITECTURE_DECISIONS.md`
- **Complexity Guide**: `docs/COMPLEXITY_MANAGEMENT.md`
- **Beginner Tutorial**: `docs/BEGINNER_TUTORIAL.md`
- **API Documentation**: `http://localhost:8000/docs` (when server running)

## Pre-Commit Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (`ruff check` passes)
- [ ] Code is formatted (`ruff format` applied)
- [ ] Tests pass (`pytest` passes)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Documentation
- [ ] README updated
- [ ] API docs updated
- [ ] Code comments added

## Checklist
- [ ] Code follows style guidelines
- [ ] No merge conflicts
- [ ] Ready for review
```

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors graph

Thank you for contributing to APEX! 🚀
