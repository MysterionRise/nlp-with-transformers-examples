"""Pytest configuration and shared fixtures"""
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_text():
    """Sample text for testing NLP operations"""
    return "This is a great product! I love it."


@pytest.fixture
def sample_negative_text():
    """Sample negative text for testing"""
    return "This is terrible. I hate it."


@pytest.fixture
def sample_sentences():
    """Sample sentences for similarity testing"""
    return [
        "The cat sat on the mat.",
        "A feline rested on a rug.",
        "Dogs are great pets.",
        "Python is a programming language.",
    ]


@pytest.fixture
def sample_long_text():
    """Sample long text for summarization testing"""
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
