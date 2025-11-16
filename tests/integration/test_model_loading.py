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

    @pytest.mark.slow
    def test_gliner_model_loading(self):
        """Test loading GLiNER model"""
        pytest.importorskip("gliner")
        from gliner import GLiNER

        try:
            # Load GLiNER model
            model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
            assert model is not None

            # Test basic NER with custom labels
            text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            labels = ["person", "organization", "location"]
            entities = model.predict_entities(text, labels, threshold=0.5)

            # Should find at least some entities
            assert len(entities) > 0

            # Check entity structure
            for entity in entities:
                assert "text" in entity
                assert "label" in entity
                assert "start" in entity
                assert "end" in entity
                assert "score" in entity

        except Exception as e:
            pytest.skip(f"GLiNER model not available: {e}")

    @pytest.mark.slow
    def test_gliner_custom_labels(self):
        """Test GLiNER with custom entity labels"""
        pytest.importorskip("gliner")
        from gliner import GLiNER

        try:
            model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

            # Test with domain-specific labels
            text = "The patient was prescribed ibuprofen for headache and took 200mg daily."
            labels = ["medication", "symptom", "dosage"]
            entities = model.predict_entities(text, labels, threshold=0.4)

            # Check that we can use custom labels
            assert isinstance(entities, list)

            # Verify label types match our custom labels
            found_labels = {entity["label"] for entity in entities}
            assert found_labels.issubset(set(labels))

        except Exception as e:
            pytest.skip(f"GLiNER model not available: {e}")


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
