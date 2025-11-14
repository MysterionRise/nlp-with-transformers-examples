"""Integration tests for model loading and basic operations"""

import pytest


class TestModelLoading:
    """Test basic model loading operations (marked as slow)"""

    @pytest.mark.slow
    def test_spacy_model_loading(self):
        """Test loading spacy model"""
        pytest.importorskip("spacy")
        import spacy

        try:
            # Try to load the model used in ner.py
            nlp = spacy.load("en_core_web_sm")
            assert nlp is not None

            # Test basic NER
            doc = nlp("Apple is a technology company founded by Steve Jobs.")
            assert len(list(doc.ents)) > 0

        except OSError:
            pytest.skip("Spacy model en_core_web_sm not installed")

    @pytest.mark.slow
    def test_sentence_transformers_import(self):
        """Test sentence transformers availability"""
        try:
            from sentence_transformers import SentenceTransformer

            assert SentenceTransformer is not None
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestBasicNLPOperations:
    """Test basic NLP operations without heavy models"""

    def test_text_preprocessing(self):
        """Test basic text preprocessing"""
        text = "  This is a test.  "
        cleaned = text.strip()
        assert cleaned == "This is a test."

    def test_tokenization_basic(self):
        """Test basic tokenization"""
        text = "This is a test."
        tokens = text.split()
        assert len(tokens) == 4
        assert tokens[0] == "This"
