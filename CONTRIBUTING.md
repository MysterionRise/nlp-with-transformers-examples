# Contributing to NLP with Transformers Examples

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nlp-with-transformers-examples.git
   cd nlp-with-transformers-examples
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # Or manually:
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   python -m spacy download en_core_web_sm
   pre-commit install
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**

6. **Run tests and checks**
   ```bash
   make test
   make lint
   make quality
   ```

7. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**

## 📋 Development Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip
- git

### Installation

```bash
# Clone the repository
git clone https://github.com/MysterionRise/nlp-with-transformers-examples.git
cd nlp-with-transformers-examples

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
make install-dev

# Install pre-commit hooks
pre-commit install
```

## 🧪 Testing

### Running Tests

```bash
# All tests with coverage
make test

# Unit tests only
make test-unit

# Smoke tests only
make test-smoke

# Fast tests (no coverage)
make test-fast
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place UI smoke tests in `tests/ui/`
- Use descriptive test names: `test_function_does_something()`
- Use pytest fixtures from `tests/conftest.py`
- Mark tests appropriately: `@pytest.mark.smoke`, `@pytest.mark.unit`, etc.

Example:
```python
import pytest

@pytest.mark.unit
def test_analyze_sentiment_empty_input():
    """Test sentiment analysis with empty input"""
    result = analyze_sentiment("")
    assert "error" in result or "empty" in str(result).lower()
```

## 💅 Code Style

We use several tools to maintain code quality:

### Formatting

```bash
# Format code
make format

# Check formatting
make lint
```

### Tools Used

- **Black** - Code formatting (line length: 120)
- **isort** - Import sorting
- **flake8** - Linting
- **bandit** - Security checks
- **mypy** - Type checking (optional)

### Style Guidelines

1. **Line Length**: Maximum 120 characters
2. **Imports**:
   - Standard library first
   - Third-party libraries second
   - Local imports last
   - Alphabetically sorted within each group
3. **Docstrings**: Use Google-style docstrings
4. **Type Hints**: Encouraged but not required
5. **Comments**: Clear and concise

Example:
```python
"""
Module description here.
"""

import os
from typing import List, Dict

import numpy as np
from transformers import pipeline

from ui.ui_config import create_theme


def analyze_sentiment(text: str, model_name: str) -> Dict[str, float]:
    """
    Analyze sentiment of the input text.

    Args:
        text: Input text to analyze
        model_name: Name of the model to use

    Returns:
        Dictionary with sentiment scores

    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # Implementation here
    return results
```

## 🎯 Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### Examples
```bash
feat(ui): add dark mode toggle to sentiment playground
fix(ner): handle empty entity lists correctly
docs: update installation instructions
test: add unit tests for similarity explorer
```

## 🏗️ Project Structure

```
nlp-with-transformers-examples/
├── ui/                          # Interactive UIs
│   ├── sentiment_playground.py
│   ├── similarity_explorer.py
│   ├── ner_visualizer.py
│   ├── summarization_studio.py
│   └── ui_config.py            # Shared UI configuration
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── ui/                     # UI smoke tests
├── eval/                        # Evaluation utilities
├── .github/workflows/          # CI/CD workflows
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pytest.ini                  # Pytest configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
└── Makefile                    # Common commands
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS (Windows, macOS, Linux)
   - Python version
   - Package versions
6. **Logs**: Relevant error messages or logs
7. **Screenshots**: If applicable

Use the bug report template when creating an issue.

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Description**: What should the feature do?
3. **Alternatives**: Have you considered alternatives?
4. **Examples**: Show examples of similar features

## 📝 Pull Request Process

1. **Update Documentation**: Update README.md, docstrings, etc.
2. **Add Tests**: Include tests for new features
3. **Run All Checks**: Ensure all tests and quality checks pass
4. **Update CHANGELOG**: Add entry to CHANGELOG.md (if exists)
5. **Link Issues**: Reference related issues
6. **Request Review**: Tag maintainers for review

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] CI checks pass
- [ ] Commit messages follow conventions

## 🔍 Code Review

We review PRs for:

- **Correctness**: Does it work as intended?
- **Testing**: Are there adequate tests?
- **Style**: Does it follow our guidelines?
- **Documentation**: Is it well documented?
- **Performance**: Are there performance implications?
- **Security**: Are there security concerns?

## 📚 Adding New UIs

When adding a new UI:

1. Create file in `ui/` directory
2. Use `ui_config.py` for theming and styling
3. Add launcher entry in `launch_ui.py`
4. Add smoke tests in `tests/ui/test_smoke.py`
5. Update README.md with UI documentation
6. Add examples and documentation within the UI

Example structure:
```python
"""
UI Title - Description

Features:
- Feature 1
- Feature 2
"""

import gradio as gr
from ui.ui_config import create_theme, create_header, create_footer

def create_ui():
    theme = create_theme()

    with gr.Blocks(theme=theme) as demo:
        gr.HTML(create_header("UI Title", "Description", "🎨"))

        # UI components here

        gr.HTML(create_footer())

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
```

## 🎨 UI Design Guidelines

- Use consistent theming from `ui_config.py`
- Include example data
- Add clear descriptions and documentation
- Use tabs for different features
- Include "About" section
- Add visual feedback for actions
- Handle errors gracefully
- Support mobile views

## 🚦 CI/CD

Our CI/CD pipeline includes:

- **Linting**: Black, isort, flake8
- **Testing**: Pytest with coverage
- **Security**: Bandit, safety
- **Pre-commit**: Automated checks
- **Multi-platform**: Ubuntu, macOS, Windows
- **Multi-version**: Python 3.9, 3.10, 3.11

See `.github/workflows/` for details.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

- Open an issue for questions
- Check existing issues and PRs
- Review documentation

## 🙏 Thank You!

We appreciate your contributions to making this project better!
