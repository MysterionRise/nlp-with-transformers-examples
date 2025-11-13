"""
Pytest fixtures and configuration for the test suite
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a test sentence for NLP processing."


@pytest.fixture
def sample_texts():
    """Multiple sample texts for batch testing"""
    return [
        "This is a positive review. Great product!",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
    ]


@pytest.fixture
def sample_article():
    """Sample article for summarization testing"""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    as opposed to natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents,
    which refers to any system that perceives its environment and takes actions
    that maximize its chance of achieving its goals. The term artificial intelligence
    had previously been used to describe machines that mimic and display human cognitive
    skills that are associated with the human mind, such as learning and problem-solving.
    """


@pytest.fixture
def mock_model():
    """Mock transformer model for testing without loading actual models"""
    mock = MagicMock()
    mock.return_value = [{"label": "POSITIVE", "score": 0.95}]
    return mock


@pytest.fixture
def mock_pipeline(mock_model):
    """Mock pipeline for testing"""
    def _mock_pipeline(task, model, tokenizer, **kwargs):
        return mock_model
    return _mock_pipeline


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "error")


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for tests"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
