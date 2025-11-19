"""Pytest configuration and shared fixtures

This module provides reusable fixtures for testing the NLP transformers project.
It includes factory fixtures for creating test objects, sample data fixtures,
and configuration helpers.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Factory Fixtures - Create customizable test objects
# ============================================================================


@pytest.fixture
def model_config_factory():
    """Factory for creating ModelConfig instances with custom parameters.

    Returns:
        Callable that creates ModelConfig objects with defaults that can be overridden

    Example:
        config = model_config_factory(task="summarization", max_length=1024)
    """
    from config import ModelConfig

    def _create(
        name: str = "Test Model",
        model_id: str = "test/model",
        task: str = "text-classification",
        description: str = "A test model for unit testing",
        max_length: int = 512,
        **kwargs: Any,
    ) -> ModelConfig:
        """Create a ModelConfig with the given parameters"""
        return ModelConfig(
            name=name,
            model_id=model_id,
            task=task,
            description=description,
            max_length=max_length,
            **kwargs,
        )

    return _create


@pytest.fixture
def mock_pipeline_factory():
    """Factory for creating mock pipeline objects.

    Returns:
        Callable that creates configured mock pipeline instances

    Example:
        pipeline = mock_pipeline_factory(return_value=[{"label": "POSITIVE", "score": 0.99}])
    """

    def _create(return_value: Any = None, side_effect: Any = None) -> Mock:
        """Create a mock pipeline with specified behavior"""
        mock_pipeline = Mock()
        if return_value is not None:
            mock_pipeline.return_value = return_value
        if side_effect is not None:
            mock_pipeline.side_effect = side_effect
        return mock_pipeline

    return _create


@pytest.fixture
def temp_log_file(tmp_path: Path):
    """Create a temporary log file for testing.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to a temporary log file that will be cleaned up after the test

    Example:
        logger = setup_file_logger(temp_log_file)
    """
    log_file = tmp_path / "test.log"
    yield log_file
    # Cleanup is automatic with tmp_path


@pytest.fixture
def temp_config_file(tmp_path: Path):
    """Create a temporary config file for testing.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to a temporary config file

    Example:
        config = load_config(temp_config_file)
    """
    config_file = tmp_path / "config.yaml"
    yield config_file


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing logging calls.

    Returns:
        Mock logger object that can be used to verify logging calls

    Example:
        function_that_logs(mock_logger)
        mock_logger.info.assert_called_once()
    """
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


# ============================================================================
# Sample Data Fixtures - Provide realistic test data
# ============================================================================


@pytest.fixture
def sample_text():
    """Sample positive text for testing NLP operations.

    Returns:
        str: A positive sentiment text sample
    """
    return "This is a great product! I love it."


@pytest.fixture
def sample_negative_text():
    """Sample negative text for testing sentiment analysis.

    Returns:
        str: A negative sentiment text sample
    """
    return "This is terrible. I hate it."


@pytest.fixture
def sample_neutral_text():
    """Sample neutral text for testing sentiment analysis.

    Returns:
        str: A neutral sentiment text sample
    """
    return "The product arrived on time. It has standard features."


@pytest.fixture
def sample_sentences():
    """Sample sentences for similarity testing.

    Returns:
        list[str]: List of semantically varied sentences
    """
    return [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "Dogs are great pets.",
        "Python is a programming language.",
    ]


@pytest.fixture
def sample_long_text():
    """Sample long text for summarization testing.

    Returns:
        str: A multi-sentence text suitable for summarization
    """
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals. Colloquially, the term
    "artificial intelligence" is often used to describe machines (or computers)
    that mimic "cognitive" functions that humans associate with the human mind,
    such as "learning" and "problem solving".
    """


@pytest.fixture
def sample_entities_text():
    """Sample text containing named entities for NER testing.

    Returns:
        str: Text with person, location, and organization entities
    """
    return "Apple Inc. CEO Tim Cook announced the new product in Cupertino, California."


@pytest.fixture
def sample_multilingual_texts():
    """Sample texts in different languages for multilingual testing.

    Returns:
        dict[str, str]: Dictionary mapping language codes to sample texts
    """
    return {
        "en": "This is a wonderful day!",
        "es": "¡Este es un día maravilloso!",
        "fr": "C'est une journée merveilleuse!",
        "de": "Das ist ein wunderbarer Tag!",
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests.

    Returns:
        dict: Configuration parameters for performance benchmarks
    """
    return {
        "min_rounds": 5,
        "warmup": True,
        "max_time": 1.0,  # seconds
    }


# ============================================================================
# Test Isolation Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to ensure isolation.

    This fixture runs automatically for every test to prevent state leakage
    between tests that use singleton patterns.
    """
    # Reset ModelCache singleton
    from utils.model_cache import ModelCache

    ModelCache._instance = None

    # Reset Settings singleton
    from config import Settings

    Settings._instance = None

    yield

    # Cleanup after test
    if ModelCache._instance is not None:
        if hasattr(ModelCache._instance, "_cache"):
            ModelCache._instance._cache.clear()
    ModelCache._instance = None
    Settings._instance = None
