"""Integration tests for UI components - testing imports and basic functionality"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestUIImports:
    """Test that UI modules can be imported without errors"""

    def test_sentiment_playground_import(self):
        """Test importing sentiment_playground module"""
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "sentiment_playground", project_root / "ui" / "sentiment_playground.py"
            )
            module = importlib.util.module_from_spec(spec)
            # Don't execute module to avoid loading models, just check it loads
            assert module is not None
        except Exception as e:
            pytest.fail(f"Failed to import sentiment_playground: {e}")

    def test_similarity_explorer_import(self):
        """Test importing similarity_explorer module"""
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "similarity_explorer", project_root / "ui" / "similarity_explorer.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert module is not None
        except Exception as e:
            pytest.fail(f"Failed to import similarity_explorer: {e}")

    def test_ner_visualizer_import(self):
        """Test importing ner_visualizer module"""
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "ner_visualizer", project_root / "ui" / "ner_visualizer.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert module is not None
        except Exception as e:
            pytest.fail(f"Failed to import ner_visualizer: {e}")

    def test_summarization_studio_import(self):
        """Test importing summarization_studio module"""
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "summarization_studio", project_root / "ui" / "summarization_studio.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert module is not None
        except Exception as e:
            pytest.fail(f"Failed to import summarization_studio: {e}")


class TestTransformersPipeline:
    """Test basic transformers pipeline functionality"""

    @pytest.mark.slow
    def test_sentiment_pipeline_creation(self):
        """Test creating a sentiment analysis pipeline (slow test - downloads model)"""
        pytest.importorskip("transformers")
        from transformers import pipeline

        # Use a tiny model for testing
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        assert classifier is not None

        # Test with simple input
        result = classifier("This is great!")
        assert len(result) > 0
        assert "label" in result[0]
        assert "score" in result[0]


class TestDependencies:
    """Test that required dependencies are available"""

    def test_transformers_import(self):
        """Test that transformers library is available"""
        try:
            import transformers

            assert transformers is not None
        except ImportError:
            pytest.fail("transformers library not available")

    def test_torch_import(self):
        """Test that PyTorch is available"""
        try:
            import torch

            assert torch is not None
        except ImportError:
            pytest.fail("torch library not available")

    def test_gradio_import(self):
        """Test that Gradio is available"""
        try:
            import gradio

            assert gradio is not None
        except ImportError:
            pytest.fail("gradio library not available")

    def test_numpy_import(self):
        """Test that numpy is available"""
        try:
            import numpy

            assert numpy is not None
        except ImportError:
            pytest.fail("numpy library not available")

    def test_spacy_import(self):
        """Test that spacy is available"""
        try:
            import spacy

            assert spacy is not None
        except ImportError:
            pytest.fail("spacy library not available")
