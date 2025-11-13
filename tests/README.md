# Test Suite

Comprehensive test suite for NLP with Transformers Examples.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   └── test_ui_config.py    # UI configuration tests
├── integration/             # Integration tests
│   └── __init__.py
└── ui/                      # UI smoke tests
    ├── __init__.py
    └── test_smoke.py        # Smoke tests for all UIs
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
```

### Unit Tests Only
```bash
pytest tests/unit/ -v
```

### Smoke Tests Only
```bash
pytest tests/ui/test_smoke.py -v -m smoke
```

### Fast Tests (Parallel Execution)
```bash
pytest tests/ -v -n auto
```

### Specific Test File
```bash
pytest tests/unit/test_ui_config.py -v
```

### Using Makefile
```bash
make test          # All tests with coverage
make test-unit     # Unit tests only
make test-smoke    # Smoke tests only
make test-fast     # Tests without coverage (faster)
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for workflows
- `@pytest.mark.smoke` - Smoke tests for basic functionality
- `@pytest.mark.ui` - UI-related tests
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.model` - Tests that require model loading

### Running Tests by Marker
```bash
pytest -m smoke       # Run only smoke tests
pytest -m unit        # Run only unit tests
pytest -m "not slow"  # Skip slow tests
```

## Test Coverage

Coverage reports are generated in multiple formats:

1. **Terminal** - Summary displayed after test run
2. **HTML** - Detailed report in `htmlcov/index.html`
3. **XML** - Machine-readable format in `coverage.xml`

View HTML coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests run automatically on:
- **Push** to main/master/develop branches
- **Pull Requests** to main/master/develop branches
- **Manual** workflow dispatch

See `.github/workflows/ci.yml` for CI configuration.

## Writing New Tests

### Test File Naming
- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_*()`
- Test classes: `Test*`

### Example Unit Test
```python
import pytest

def test_example():
    """Test description"""
    result = some_function()
    assert result == expected_value
```

### Example with Fixtures
```python
import pytest

def test_with_fixture(sample_text):
    """Test using fixture from conftest.py"""
    assert len(sample_text) > 0
```

### Example with Mocking
```python
from unittest.mock import patch, MagicMock

@patch('module.function')
def test_with_mock(mock_function):
    """Test with mocked dependencies"""
    mock_function.return_value = "mocked"
    result = call_function_that_uses_it()
    assert result == "expected"
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_text` - Single sample text for testing
- `sample_texts` - Multiple sample texts for batch testing
- `sample_article` - Long-form article for summarization
- `mock_model` - Mock transformer model
- `mock_pipeline` - Mock pipeline
- `temp_data_dir` - Temporary directory for test data

## Quality Checks

Beyond tests, we also run:

### Linting
```bash
make lint
```

### Code Formatting
```bash
make format
```

### Security Scan
```bash
make security
```

### All Quality Checks
```bash
make quality
```

## Pre-commit Hooks

Install pre-commit hooks to run checks before each commit:
```bash
pre-commit install
```

This will automatically run:
- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- bandit (security)
- YAML validation
- Trailing whitespace removal

## Troubleshooting

### Tests Fail Due to Missing Models

Some tests require Spacy models:
```bash
python -m spacy download en_core_web_sm
```

### ImportError in Tests

Ensure the project is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or run tests from project root:
```bash
cd /path/to/nlp-with-transformers-examples
pytest tests/
```

### Slow Tests

Use parallel execution:
```bash
pytest tests/ -n auto
```

Or skip slow tests:
```bash
pytest tests/ -m "not slow"
```

## Test Environment Variables

Set these environment variables for testing:

- `TESTING=1` - Indicates test environment
- `TRANSFORMERS_VERBOSITY=error` - Reduce transformers logging
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1` - Disable symlink warnings

These are automatically set in `conftest.py`.
