"""
Unit tests for newly implemented UI modules.

Tests for:
- Question Answering System (qa_system.py)
- Text Generation Playground (generation_playground.py)
- Zero-Shot Classifier (zero_shot_classifier.py)
- Translation Hub (translation_hub.py)
- Vision-Language Explorer (vision_language_explorer.py)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestQASystem:
    """Test the Question Answering System UI"""

    @patch("ui.qa_system.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful model loading for QA"""
        from ui.qa_system import load_model

        mock_pipeline.return_value = MagicMock()

        model = load_model("DistilBERT SQuAD")
        assert model is not None
        mock_pipeline.assert_called_once()

    @patch("ui.qa_system.pipeline")
    def test_load_model_caching(self, mock_pipeline):
        """Test that models are properly cached"""
        from ui.qa_system import load_model, model_cache

        mock_pipeline.return_value = MagicMock()

        # Clear cache
        model_cache.clear()

        # Load twice, should only call pipeline once
        load_model("DistilBERT SQuAD")
        load_model("DistilBERT SQuAD")

        assert mock_pipeline.call_count == 1
        assert "DistilBERT SQuAD" in model_cache

    def test_answer_question_empty_context(self):
        """Test handling of empty context"""
        from ui.qa_system import answer_question

        result, _, _ = answer_question("", "What is AI?", "DistilBERT SQuAD")
        assert "error" in result

    def test_answer_question_empty_question(self):
        """Test handling of empty question"""
        from ui.qa_system import answer_question

        context = "Artificial intelligence is a field of computer science."
        result, _, _ = answer_question(context, "", "DistilBERT SQuAD")
        assert "error" in result

    @patch("ui.qa_system.load_model")
    def test_answer_question_success(self, mock_load):
        """Test successful question answering"""
        from ui.qa_system import answer_question

        mock_model = MagicMock()
        mock_model.return_value = {
            "answer": "Paris",
            "score": 0.95,
            "start": 20,
            "end": 25,
        }
        mock_load.return_value = mock_model

        context = "The Eiffel Tower is located in Paris, France."
        question = "Where is the Eiffel Tower?"

        result, html, fig = answer_question(context, question, "DistilBERT SQuAD")

        assert "Answer" in result
        assert result["Answer"] == "Paris"
        assert "95.00%" in result["Confidence"]
        assert html is not None

    def test_highlight_answer(self):
        """Test answer highlighting HTML generation"""
        from ui.qa_system import highlight_answer

        context = "The Eiffel Tower is in Paris"
        answer = "Paris"
        html = highlight_answer(context, answer, 20, 25)

        assert "mark" in html
        assert "Paris" in html
        assert "FFD700" in html  # Gold color


class TestGenerationPlayground:
    """Test the Text Generation Playground UI"""

    @patch("ui.generation_playground.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful generation model loading"""
        from ui.generation_playground import load_model

        mock_pipeline.return_value = MagicMock()

        model = load_model("GPT-2")
        assert model is not None
        mock_pipeline.assert_called_once()

    def test_generate_text_empty_prompt(self):
        """Test handling of empty prompt"""
        from ui.generation_playground import generate_text

        result = generate_text("", "GPT-2")
        assert "error" in result

    @patch("ui.generation_playground.load_model")
    @patch("ui.generation_playground.torch")
    def test_generate_text_success(self, mock_torch, mock_load):
        """Test successful text generation"""
        from ui.generation_playground import generate_text

        mock_model = MagicMock()
        mock_model.return_value = [{"generated_text": "The future of AI is bright and promising"}]
        mock_load.return_value = mock_model

        result = generate_text("The future of AI", "GPT-2", max_length=50, temperature=0.7)

        assert "error" not in result
        assert "Prompt" in result
        assert "Samples" in result
        assert len(result["Samples"]) > 0

    @patch("ui.generation_playground.load_model")
    def test_batch_generate(self, mock_load):
        """Test batch text generation"""
        from ui.generation_playground import batch_generate

        mock_model = MagicMock()
        mock_model.return_value = [{"generated_text": "Generated text here"}]
        mock_load.return_value = mock_model

        prompts = "Once upon a time\nIn a galaxy far away"
        result = batch_generate(prompts, "GPT-2")

        assert "1. Prompt:" in result
        assert "2. Prompt:" in result
        assert "Generated:" in result


class TestZeroShotClassifier:
    """Test the Zero-Shot Classification UI"""

    @patch("ui.zero_shot_classifier.pipeline")
    def test_load_model_success(self, mock_pipeline):
        """Test successful zero-shot model loading"""
        from ui.zero_shot_classifier import load_model

        mock_pipeline.return_value = MagicMock()

        model = load_model("BART Large MNLI")
        assert model is not None
        mock_pipeline.assert_called_once()

    def test_classify_text_empty_text(self):
        """Test handling of empty text"""
        from ui.zero_shot_classifier import classify_text

        result, _ = classify_text("", "positive, negative", "BART Large MNLI")
        assert "error" in result

    def test_classify_text_empty_labels(self):
        """Test handling of empty labels"""
        from ui.zero_shot_classifier import classify_text

        result, _ = classify_text("This is great!", "", "BART Large MNLI")
        assert "error" in result

    @patch("ui.zero_shot_classifier.load_model")
    def test_classify_text_success(self, mock_load):
        """Test successful text classification"""
        from ui.zero_shot_classifier import classify_text

        mock_model = MagicMock()
        mock_model.return_value = {
            "labels": ["positive", "negative", "neutral"],
            "scores": [0.9, 0.08, 0.02],
        }
        mock_load.return_value = mock_model

        result, fig = classify_text(
            "This movie was great!",
            "positive, negative, neutral",
            "BART Large MNLI",
        )

        assert "Top Classification" in result
        assert result["Top Classification"] == "positive"
        assert "90.00%" in result["Confidence"]
        assert "All Scores" in result

    @patch("ui.zero_shot_classifier.load_model")
    def test_batch_classify(self, mock_load):
        """Test batch classification"""
        from ui.zero_shot_classifier import batch_classify

        mock_model = MagicMock()
        mock_model.return_value = {
            "labels": ["positive", "negative"],
            "scores": [0.85, 0.15],
        }
        mock_load.return_value = mock_model

        texts = "Great!\nTerrible."
        result = batch_classify(texts, "positive, negative", "BART Large MNLI")

        assert "1. Text:" in result
        assert "2. Text:" in result


class TestTranslationHub:
    """Test the Translation Hub UI"""

    def test_translate_empty_text(self):
        """Test handling of empty text"""
        from ui.translation_hub import translate_mbart

        result = translate_mbart("", "English", "Spanish")
        assert "error" in result

    @patch("ui.translation_hub.load_mbart_model")
    def test_translate_mbart_success(self, mock_load):
        """Test successful mBART translation"""
        from ui.translation_hub import translate_mbart

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Hola, ¿cómo estás?"}]
        mock_model.tokenizer = MagicMock()
        mock_load.return_value = mock_model

        result = translate_mbart("Hello, how are you?", "English", "Spanish")

        assert "error" not in result
        assert "Original" in result
        assert "Translation" in result
        assert "Model" in result

    def test_translate_helsinki_empty_text(self):
        """Test Helsinki translation with empty text"""
        from ui.translation_hub import translate_helsinki

        result = translate_helsinki("", "en-es")
        assert "error" in result

    @patch("ui.translation_hub.load_helsinki_model")
    def test_translate_helsinki_success(self, mock_load):
        """Test successful Helsinki translation"""
        from ui.translation_hub import translate_helsinki

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Hola mundo"}]
        mock_load.return_value = mock_model

        result = translate_helsinki("Hello world", "en-es")

        assert "error" not in result
        assert "Translation" in result
        assert "Language Pair" in result

    @patch("ui.translation_hub.load_mbart_model")
    def test_batch_translate(self, mock_load):
        """Test batch translation"""
        from ui.translation_hub import batch_translate

        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Translated text"}]
        mock_model.tokenizer = MagicMock()
        mock_load.return_value = mock_model

        texts = "Hello\nWorld"
        result = batch_translate(texts, "mbart", "English", "Spanish")

        assert "1. Original:" in result
        assert "2. Original:" in result


class TestVisionLanguageExplorer:
    """Test the Vision-Language Explorer UI"""

    def test_generate_caption_no_image(self):
        """Test caption generation with no image"""
        from ui.vision_language_explorer import generate_caption

        result = generate_caption(None)
        assert "error" in result

    @patch("ui.vision_language_explorer.load_image_captioning_model")
    def test_generate_caption_success(self, mock_load):
        """Test successful image caption generation"""
        from PIL import Image

        from ui.vision_language_explorer import generate_caption

        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="red")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor.batch_decode = MagicMock(return_value=["a red square on white background"])
        mock_model.generate = MagicMock(return_value=MagicMock())

        mock_load.return_value = (mock_model, mock_processor)

        result = generate_caption(img)

        assert "error" not in result
        assert "Caption" in result
        assert "Model" in result
        assert "100x100" in result["Image Size"]

    def test_calculate_clip_similarity_no_image(self):
        """Test similarity calculation with no image"""
        from ui.vision_language_explorer import calculate_clip_similarity

        result, html = calculate_clip_similarity(None, ["a cat"])
        assert "error" in result

    def test_calculate_clip_similarity_no_texts(self):
        """Test similarity calculation with no texts"""
        from PIL import Image

        from ui.vision_language_explorer import calculate_clip_similarity

        img = Image.new("RGB", (100, 100))
        result, html = calculate_clip_similarity(img, [])
        assert "error" in result

    def test_batch_caption_no_images(self):
        """Test batch captioning with no images"""
        from ui.vision_language_explorer import batch_caption_images

        result = batch_caption_images([])
        assert "Please upload" in result or "No" in result


# Import verification tests


class TestUIImports:
    """Test that all new UI modules can be imported successfully"""

    def test_import_qa_system(self):
        """Test QA system module can be imported"""
        try:
            from ui import qa_system

            assert hasattr(qa_system, "create_ui")
            assert hasattr(qa_system, "answer_question")
        except ImportError as e:
            pytest.fail(f"Failed to import qa_system: {e}")

    def test_import_generation_playground(self):
        """Test text generation module can be imported"""
        try:
            from ui import generation_playground

            assert hasattr(generation_playground, "create_ui")
            assert hasattr(generation_playground, "generate_text")
        except ImportError as e:
            pytest.fail(f"Failed to import generation_playground: {e}")

    def test_import_zero_shot_classifier(self):
        """Test zero-shot classifier module can be imported"""
        try:
            from ui import zero_shot_classifier

            assert hasattr(zero_shot_classifier, "create_ui")
            assert hasattr(zero_shot_classifier, "classify_text")
        except ImportError as e:
            pytest.fail(f"Failed to import zero_shot_classifier: {e}")

    def test_import_translation_hub(self):
        """Test translation hub module can be imported"""
        try:
            from ui import translation_hub

            assert hasattr(translation_hub, "create_ui")
            assert hasattr(translation_hub, "translate_mbart")
        except ImportError as e:
            pytest.fail(f"Failed to import translation_hub: {e}")

    def test_import_vision_language_explorer(self):
        """Test vision-language explorer module can be imported"""
        try:
            from ui import vision_language_explorer

            assert hasattr(vision_language_explorer, "create_ui")
            assert hasattr(vision_language_explorer, "generate_caption")
        except ImportError as e:
            pytest.fail(f"Failed to import vision_language_explorer: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
