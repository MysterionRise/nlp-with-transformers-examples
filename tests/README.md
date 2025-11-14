# Tests

This directory contains the test suite for the NLP with Transformers Examples project.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests
│   ├── test_launch_ui.py
│   └── test_sentiment_analysis.py
└── integration/             # Integration tests
    ├── test_ui_imports.py
    └── test_model_loading.py
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Only Fast Tests (skip slow model loading tests)

```bash
pytest -m "not slow"
```

### Run Only Slow/Integration Tests

```bash
pytest -m "slow"
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

### Run Specific Test File

```bash
pytest tests/unit/test_launch_ui.py
```

### Run Specific Test Class or Function

```bash
pytest tests/unit/test_launch_ui.py::TestUIConfiguration
pytest tests/unit/test_launch_ui.py::TestUIConfiguration::test_uis_dict_structure
```

### Run Tests in Parallel

```bash
pytest -n auto  # Uses all CPU cores
pytest -n 4     # Uses 4 workers
```

## Test Markers

- `@pytest.mark.slow` - Tests that take a long time (model downloads, etc.)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.ui` - UI-related tests

## CI/CD

Tests are automatically run on every push and pull request via GitHub Actions. See `.github/workflows/ci.yml` for the CI configuration.

### CI Jobs

1. **Lint and Format Check** - Checks code style with black, isort, and flake8
2. **Fast Tests** - Runs unit tests on Python 3.9, 3.10, and 3.11
3. **Integration Tests** - Runs slow tests with model loading on Python 3.10
4. **All Tests** - Runs complete test suite and generates coverage report

## Writing Tests

### Adding a New Test

1. Create a new test file in `tests/unit/` or `tests/integration/`
2. Name it `test_*.py`
3. Create test classes with names starting with `Test`
4. Create test functions with names starting with `test_`
5. Use fixtures from `conftest.py` or create new ones

Example:

```python
import pytest

class TestMyFeature:
    def test_something(self, sample_text):
        # Your test code here
        assert True

    @pytest.mark.slow
    def test_slow_operation(self):
        # Slow test that will be skipped in fast runs
        assert True
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_text` - Sample positive text
- `sample_negative_text` - Sample negative text
- `sample_sentences` - List of sample sentences
- `sample_long_text` - Long text for summarization

## Coverage Goals

- Target: > 70% overall coverage
- Unit tests should have high coverage
- Integration tests focus on critical paths

## Notes

- Slow tests are skipped in the fast test job to keep CI times reasonable
- Model downloads are cached in CI to speed up subsequent runs
- Some tests require the spacy model `en_core_web_sm` to be installed
