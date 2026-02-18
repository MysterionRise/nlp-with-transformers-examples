"""Unit tests for sentiment-analysis.py

Note: The original sentiment-analysis.py script was removed during the
repository rebrand. These tests are preserved but skipped until the module
is restored or tests are updated to target the new sentiment UI/API.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SENTIMENT_SCRIPT = Path(__file__).parent.parent.parent / "sentiment-analysis.py"
SKIP_REASON = "sentiment-analysis.py was removed during repository rebrand"


def _load_sentiment_module():
    """Helper to dynamically load the sentiment-analysis module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sentiment_analysis", SENTIMENT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SENTIMENT_SCRIPT.exists(), reason=SKIP_REASON)
class TestPreprocessFunction:
    """Test the preprocess function"""

    def test_preprocess_import(self):
        """Test that preprocess function can be imported"""
        module = _load_sentiment_module()
        assert hasattr(module, "preprocess")
        assert callable(module.preprocess)

    def test_preprocess_returns_input(self):
        """Test that preprocess returns the input unchanged (current implementation)"""
        module = _load_sentiment_module()
        test_text = "This is a test"
        result = module.preprocess(test_text)
        assert result == test_text

    def test_preprocess_with_various_inputs(self):
        """Test preprocess with various input types"""
        module = _load_sentiment_module()

        # Test with empty string
        assert module.preprocess("") == ""

        # Test with special characters
        special_text = "Hello! @#$ %^&*"
        assert module.preprocess(special_text) == special_text

        # Test with newlines
        multiline_text = "Line 1\nLine 2"
        assert module.preprocess(multiline_text) == multiline_text
