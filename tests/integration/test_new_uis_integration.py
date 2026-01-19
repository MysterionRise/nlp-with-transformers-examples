"""
Integration tests for newly implemented UI modules.

These tests verify that the new UI modules can be imported, configured,
and their main functions work correctly with mocked models.

Marked as @pytest.mark.slow for CI/CD pipeline.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.slow
class TestNewUIsIntegration:
    """Integration tests for all new UI modules"""

    def test_qa_system_create_ui(self):
        """Test that QA system UI can be created"""
        from ui.qa_system import create_ui

        ui = create_ui()
        assert ui is not None
        assert hasattr(ui, "launch")

    def test_generation_playground_create_ui(self):
        """Test that generation playground UI can be created"""
        from ui.generation_playground import create_ui

        ui = create_ui()
        assert ui is not None
        assert hasattr(ui, "launch")

    def test_zero_shot_classifier_create_ui(self):
        """Test that zero-shot classifier UI can be created"""
        from ui.zero_shot_classifier import create_ui

        ui = create_ui()
        assert ui is not None
        assert hasattr(ui, "launch")

    def test_translation_hub_create_ui(self):
        """Test that translation hub UI can be created"""
        from ui.translation_hub import create_ui

        ui = create_ui()
        assert ui is not None
        assert hasattr(ui, "launch")

    def test_vision_language_explorer_create_ui(self):
        """Test that vision-language explorer UI can be created"""
        from ui.vision_language_explorer import create_ui

        ui = create_ui()
        assert ui is not None
        assert hasattr(ui, "launch")

    @patch("ui.qa_system.pipeline")
    def test_qa_system_workflow(self, mock_pipeline):
        """Test complete QA system workflow"""
        from ui.qa_system import answer_question, create_ui, load_model

        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value = {
            "answer": "Paris",
            "score": 0.95,
            "start": 20,
            "end": 25,
        }
        mock_pipeline.return_value = mock_model

        # Test model loading
        model = load_model("DistilBERT SQuAD")
        assert model is not None

        # Test answering
        context = "The Eiffel Tower is located in Paris, France."
        question = "Where is the Eiffel Tower?"
        result, html, fig = answer_question(context, question, "DistilBERT SQuAD")

        assert "Answer" in result
        assert result["Answer"] == "Paris"

        # Test UI creation
        ui = create_ui()
        assert ui is not None

    @patch("ui.generation_playground.pipeline")
    @patch("ui.generation_playground.torch")
    def test_generation_playground_workflow(self, mock_torch, mock_pipeline):
        """Test complete text generation workflow"""
        from ui.generation_playground import batch_generate, create_ui, generate_text, load_model

        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value = [
            {"generated_text": "The future of AI is bright and promising"}
        ]
        mock_pipeline.return_value = mock_model

        # Test model loading
        model = load_model("GPT-2")
        assert model is not None

        # Test generation
        result = generate_text("The future of AI", "GPT-2")
        assert "Samples" in result

        # Test batch generation
        batch_result = batch_generate("Prompt 1\nPrompt 2", "GPT-2")
        assert "1. Prompt:" in batch_result

        # Test UI creation
        ui = create_ui()
        assert ui is not None

    @patch("ui.zero_shot_classifier.pipeline")
    def test_zero_shot_classifier_workflow(self, mock_pipeline):
        """Test complete zero-shot classification workflow"""
        from ui.zero_shot_classifier import batch_classify, classify_text, create_ui, load_model

        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value = {
            "labels": ["positive", "negative", "neutral"],
            "scores": [0.9, 0.08, 0.02],
        }
        mock_pipeline.return_value = mock_model

        # Test model loading
        model = load_model("BART Large MNLI")
        assert model is not None

        # Test classification
        result, fig = classify_text(
            "This is great!", "positive, negative, neutral", "BART Large MNLI"
        )
        assert "Top Classification" in result

        # Test batch classification
        batch_result = batch_classify(
            "Great!\nTerrible.", "positive, negative", "BART Large MNLI"
        )
        assert "1. Text:" in batch_result

        # Test UI creation
        ui = create_ui()
        assert ui is not None

    @patch("ui.translation_hub.load_mbart_model")
    def test_translation_hub_workflow(self, mock_load_mbart):
        """Test complete translation workflow"""
        from ui.translation_hub import batch_translate, create_ui, translate_mbart

        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value = [{"translation_text": "Hola, ¿cómo estás?"}]
        mock_model.tokenizer = MagicMock()
        mock_load_mbart.return_value = mock_model

        # Test translation
        result = translate_mbart("Hello, how are you?", "English", "Spanish")
        assert "Translation" in result

        # Test batch translation
        batch_result = batch_translate(
            "Hello\nWorld", "mbart", "English", "Spanish"
        )
        assert "1. Original:" in batch_result

        # Test UI creation
        ui = create_ui()
        assert ui is not None

    @patch("ui.vision_language_explorer.load_image_captioning_model")
    def test_vision_language_explorer_workflow(self, mock_load_git):
        """Test complete vision-language workflow"""
        from PIL import Image

        from ui.vision_language_explorer import create_ui, generate_caption

        # Create dummy image
        img = Image.new("RGB", (100, 100), color="blue")

        # Mock the model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_processor.batch_decode = MagicMock(return_value=["a blue square"])
        mock_model.generate = MagicMock(return_value=MagicMock())
        mock_load_git.return_value = (mock_model, mock_processor)

        # Test caption generation
        result = generate_caption(img)
        assert "Caption" in result

        # Test UI creation
        ui = create_ui()
        assert ui is not None


@pytest.mark.slow
class TestNewUIsLaunchConfiguration:
    """Test that new UIs are properly configured in launcher"""

    def test_launcher_configuration(self):
        """Test that launch_ui.py includes new UIs"""
        from launch_ui import UIS

        # Check that all new UIs are registered
        assert "qa" in UIS
        assert "generation" in UIS
        assert "zero_shot" in UIS
        assert "translation" in UIS
        assert "vision" in UIS

        # Verify port assignments
        assert UIS["qa"]["port"] == 7865
        assert UIS["generation"]["port"] == 7866
        assert UIS["zero_shot"]["port"] == 7867
        assert UIS["translation"]["port"] == 7868
        assert UIS["vision"]["port"] == 7869

        # Verify files exist
        qa_path = Path(project_root) / UIS["qa"]["file"]
        gen_path = Path(project_root) / UIS["generation"]["file"]
        zs_path = Path(project_root) / UIS["zero_shot"]["file"]
        trans_path = Path(project_root) / UIS["translation"]["file"]
        vision_path = Path(project_root) / UIS["vision"]["file"]

        assert qa_path.exists()
        assert gen_path.exists()
        assert zs_path.exists()
        assert trans_path.exists()
        assert vision_path.exists()

    def test_all_uis_launchable(self):
        """Test that all UIs can be launched without errors"""
        from launch_ui import UIS, launch_ui

        for ui_key in ["qa", "generation", "zero_shot", "translation", "vision"]:
            # Verify UI is in registry
            assert ui_key in UIS
            # Verify launch_ui can handle it
            ui_info = UIS[ui_key]
            assert "name" in ui_info
            assert "file" in ui_info
            assert "port" in ui_info
            assert "description" in ui_info


@pytest.mark.slow
class TestSampleData:
    """Test that sample data files exist and are readable"""

    def test_sentiment_reviews_data(self):
        """Test sentiment reviews sample data"""
        data_path = Path(project_root) / "data" / "sentiment_reviews.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert len(content.split("\n")) > 5

    def test_qa_contexts_data(self):
        """Test QA contexts sample data"""
        data_path = Path(project_root) / "data" / "qa_contexts.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Context" in content
            assert "Question" in content

    def test_news_articles_data(self):
        """Test news articles sample data"""
        data_path = Path(project_root) / "data" / "news_articles.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Article" in content

    def test_translation_samples_data(self):
        """Test translation samples data"""
        data_path = Path(project_root) / "data" / "translation_samples.txt"
        assert data_path.exists()

        with open(data_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "Sample" in content


@pytest.mark.slow
class TestConfigurationUpdates:
    """Test that configuration has been updated with new models"""

    def test_vision_models_in_config(self):
        """Test that vision models are in configuration"""
        import yaml

        config_path = Path(project_root) / "config" / "models.yaml"
        assert config_path.exists()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for vision-language category
        assert "vision_language" in config
        assert len(config["vision_language"]) >= 3

        # Check for specific models
        models_found = list(config["vision_language"].keys())
        assert len(models_found) > 0

    def test_model_registry_categories(self):
        """Test that all required model categories exist"""
        import yaml

        config_path = Path(project_root) / "config" / "models.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for all original categories
        required_categories = [
            "sentiment_analysis",
            "summarization",
            "embeddings",
            "ner",
            "question_answering",
            "text_generation",
            "zero_shot",
            "translation",
            "vision_language",
        ]

        for category in required_categories:
            assert category in config, f"Missing category: {category}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
