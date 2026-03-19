# Contributing to HDIM

## Development Setup

```bash
git clone https://github.com/your-repo/hdim.git
cd hdim
pip install -r requirements.txt
pip install pytest ruff mypy
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_hdim.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit PR with description

## Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Type hints added
