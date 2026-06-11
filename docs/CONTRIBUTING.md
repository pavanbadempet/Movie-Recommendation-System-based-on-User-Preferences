# Contributing Guide - Nova

## Getting Started

1. Fork the repository
2. Clone: `git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git`
3. Create branch: `git checkout -b feature/your-feature`
4. Set up environment (see [INSTALLATION.md](INSTALLATION.md))

## Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install  # Optional: git hooks for formatting
```

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Format with `black`
- Lint with `flake8`

```bash
black .
flake8 .
pylint backend/
```

## Testing

```bash
pytest tests/ --cov=backend --cov-report=html
```

## Commit Guidelines

Use conventional commits:
```
feat: add personalized recommendations
fix: resolve embedding cache issue
docs: update API reference
test: add recommendation quality tests
refactor: simplify ranker logic
```

## Pull Request Process

1. Tests pass: `pytest`
2. Code quality: `black`, `flake8`, `pylint`
3. Documentation updated
4. CHANGELOG.md entry added
5. PR description includes what/why/how

## Areas for Contribution

- [ ] Additional embedding models
- [ ] New reranking strategies
- [ ] Performance optimization
- [ ] Documentation improvements
- [ ] UI enhancements
- [ ] Deployment guides

## License

By contributing, you agree work is under MIT License.

---

Questions? Visit [Discussions](https://github.com/pavanbadempet/Movie-Recommendation-System/discussions)!
