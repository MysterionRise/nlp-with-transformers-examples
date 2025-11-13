"""
Smoke tests for UI modules

These tests verify that all UI modules can be imported and basic
functions work without crashing. They don't load actual models.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.mark.smoke
@pytest.mark.ui
class TestUISmokeTests:
    """Smoke tests for all UI modules"""

    def test_sentiment_playground_import(self):
        """Test sentiment playground can be imported"""
        try:
            from ui import sentiment_playground
            assert hasattr(sentiment_playground, 'create_ui')
            assert hasattr(sentiment_playground, 'SENTIMENT_MODELS')
        except ImportError as e:
            pytest.fail(f"Failed to import sentiment_playground: {e}")

    def test_similarity_explorer_import(self):
        """Test similarity explorer can be imported"""
        try:
            from ui import similarity_explorer
            assert hasattr(similarity_explorer, 'create_ui')
            assert hasattr(similarity_explorer, 'MODEL_NAME')
        except ImportError as e:
            pytest.fail(f"Failed to import similarity_explorer: {e}")

    def test_ner_visualizer_import(self):
        """Test NER visualizer can be imported"""
        try:
            from ui import ner_visualizer
            assert hasattr(ner_visualizer, 'create_ui')
        except ImportError as e:
            pytest.fail(f"Failed to import ner_visualizer: {e}")

    def test_summarization_studio_import(self):
        """Test summarization studio can be imported"""
        try:
            from ui import summarization_studio
            assert hasattr(summarization_studio, 'create_ui')
            assert hasattr(summarization_studio, 'SUMMARIZATION_MODELS')
        except ImportError as e:
            pytest.fail(f"Failed to import summarization_studio: {e}")

    def test_ui_config_import(self):
        """Test UI config can be imported"""
        try:
            from ui import ui_config
            assert hasattr(ui_config, 'create_theme')
            assert hasattr(ui_config, 'create_header')
            assert hasattr(ui_config, 'create_footer')
        except ImportError as e:
            pytest.fail(f"Failed to import ui_config: {e}")

    @patch('ui.sentiment_playground.pipeline')
    def test_sentiment_analyze_empty_input(self, mock_pipeline):
        """Test sentiment analysis with empty input"""
        from ui.sentiment_playground import analyze_sentiment

        result, plot = analyze_sentiment("", "Twitter RoBERTa (Multilingual)")
        assert "error" in result or "Please provide" in str(result)

    @patch('ui.similarity_explorer.AutoTokenizer')
    @patch('ui.similarity_explorer.AutoModel')
    def test_similarity_compare_empty(self, mock_model, mock_tokenizer):
        """Test similarity comparison with empty input"""
        from ui.similarity_explorer import compare_two_sentences

        score, interpretation, plot = compare_two_sentences("", "")
        assert score == 0.0
        assert "Please provide" in interpretation

    def test_launcher_script_exists(self):
        """Test launcher script exists and is valid Python"""
        launcher_path = project_root / "launch_ui.py"
        assert launcher_path.exists()

        # Try to import it
        import launch_ui
        assert hasattr(launch_ui, 'main')
        assert hasattr(launch_ui, 'UIS')
        assert isinstance(launch_ui.UIS, dict)
        assert len(launch_ui.UIS) == 4  # Should have 4 UIs

    def test_all_ui_files_exist(self):
        """Test all UI files exist"""
        ui_dir = project_root / "ui"
        expected_files = [
            "sentiment_playground.py",
            "similarity_explorer.py",
            "ner_visualizer.py",
            "summarization_studio.py",
            "ui_config.py"
        ]

        for file_name in expected_files:
            file_path = ui_dir / file_name
            assert file_path.exists(), f"Missing UI file: {file_name}"

    def test_sentiment_models_configured(self):
        """Test sentiment models are properly configured"""
        from ui.sentiment_playground import SENTIMENT_MODELS

        assert isinstance(SENTIMENT_MODELS, dict)
        assert len(SENTIMENT_MODELS) > 0
        for name, model_id in SENTIMENT_MODELS.items():
            assert isinstance(name, str)
            assert isinstance(model_id, str)
            assert "/" in model_id  # HuggingFace model format

    def test_summarization_models_configured(self):
        """Test summarization models are properly configured"""
        from ui.summarization_studio import SUMMARIZATION_MODELS

        assert isinstance(SUMMARIZATION_MODELS, dict)
        assert len(SUMMARIZATION_MODELS) > 0
        for name, model_id in SUMMARIZATION_MODELS.items():
            assert isinstance(name, str)
            assert isinstance(model_id, str)


@pytest.mark.smoke
class TestLauncherScript:
    """Test the launcher script functionality"""

    def test_launcher_uis_complete(self):
        """Test launcher has all required UI definitions"""
        import launch_ui

        assert "sentiment" in launch_ui.UIS
        assert "similarity" in launch_ui.UIS
        assert "ner" in launch_ui.UIS
        assert "summarization" in launch_ui.UIS

    def test_launcher_ui_metadata(self):
        """Test launcher UI metadata is complete"""
        import launch_ui

        for key, ui_info in launch_ui.UIS.items():
            assert "name" in ui_info
            assert "file" in ui_info
            assert "port" in ui_info
            assert "description" in ui_info

            # Verify file path
            file_path = project_root / ui_info["file"]
            assert file_path.exists(), f"UI file missing: {ui_info['file']}"

            # Verify port is a number
            assert isinstance(ui_info["port"], int)
            assert 7000 < ui_info["port"] < 8000  # Reasonable port range
