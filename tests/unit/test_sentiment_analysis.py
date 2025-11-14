"""Unit tests for sentiment-analysis.py"""
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPreprocessFunction:
    """Test the preprocess function"""

    def test_preprocess_import(self):
        """Test that preprocess function can be imported"""
        # Import inline to avoid loading heavy models
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sentiment_analysis",
            Path(__file__).parent.parent.parent / "sentiment-analysis.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "preprocess")
        assert callable(module.preprocess)

    def test_preprocess_returns_input(self):
        """Test that preprocess returns the input unchanged (current implementation)"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sentiment_analysis",
            Path(__file__).parent.parent.parent / "sentiment-analysis.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_text = "This is a test"
        result = module.preprocess(test_text)
        assert result == test_text

    def test_preprocess_with_various_inputs(self):
        """Test preprocess with various input types"""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "sentiment_analysis",
            Path(__file__).parent.parent.parent / "sentiment-analysis.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Test with empty string
        assert module.preprocess("") == ""

        # Test with special characters
        special_text = "Hello! @#$ %^&*"
        assert module.preprocess(special_text) == special_text

        # Test with newlines
        multiline_text = "Line 1\nLine 2"
        assert module.preprocess(multiline_text) == multiline_text
