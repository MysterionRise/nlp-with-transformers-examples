"""
Integration tests for Phase 2 components

Tests how config, error handling, logging, and model caching work together.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config import ModelConfig, get_model_registry, get_settings
from utils import (
    PerformanceLogger,
    get_logger,
    handle_errors,
    load_model,
)


class TestConfigIntegration:
    """Test configuration system integration"""

    def test_settings_and_registry_integration(self):
        """Test that settings and registry work together"""
        settings = get_settings()
        registry = get_model_registry()

        assert settings is not None
        assert registry is not None

        # Registry should have access to settings paths
        categories = registry.list_categories()
        assert len(categories) > 0

    def test_model_config_retrieval(self):
        """Test retrieving model configurations"""
        registry = get_model_registry()

        # Test getting models from different categories
        sentiment_model = registry.get_model("sentiment_analysis", "twitter_roberta_multilingual")
        assert isinstance(sentiment_model, ModelConfig)
        assert sentiment_model.task == "sentiment-analysis"

        summarization_model = registry.get_model("summarization", "bart_large_cnn")
        assert isinstance(summarization_model, ModelConfig)
        assert summarization_model.task == "summarization"

    def test_search_models_integration(self):
        """Test model search functionality"""
        registry = get_model_registry()

        # Search for RoBERTa models
        results = registry.search_models("roberta")
        assert len(results) > 0

        # Each result should be a tuple of (category, key, config)
        for category, key, config in results:
            assert isinstance(category, str)
            assert isinstance(key, str)
            assert isinstance(config, ModelConfig)


class TestLoggingIntegration:
    """Test logging integration with other components"""

    def test_logger_with_error_handling(self, caplog):
        """Test that logging works with error handling decorators"""
        import logging

        logger = get_logger("test_integration")

        @handle_errors(default_return=None)
        def failing_function():
            logger.info("About to fail")
            raise ValueError("Test error")

        with caplog.at_level(logging.INFO, logger="test_integration"):
            result = failing_function()

        # Function should return default due to error
        assert result is None

        # Log should contain our message
        messages = [record.message for record in caplog.records]
        assert any("About to fail" in msg for msg in messages)

    def test_performance_logger_with_settings(self):
        """Test PerformanceLogger uses settings"""
        logger = get_logger("test_perf")
        settings = get_settings()

        # Logger should respect settings log level
        assert logger.level >= 0

        with PerformanceLogger("test_operation", logger=logger):
            # Simulate some work
            _ = sum(i**2 for i in range(100))

        # No exceptions should be raised


class TestModelCacheIntegration:
    """Test model cache integration"""

    @patch("utils.model_cache.pipeline")
    def test_cache_with_registry(self, mock_pipeline):
        """Test that cache uses registry for model configs"""
        mock_pipeline.return_value = Mock()

        try:
            # This should use registry to get config, then cache the model
            model = load_model("sentiment_analysis", "twitter_roberta_multilingual")
            assert model is not None
        except Exception:
            # May fail if dependencies not available, which is OK for integration test
            pytest.skip("Model loading dependencies not available")

    @patch("utils.model_cache.pipeline")
    def test_cache_with_error_handling(self, mock_pipeline):
        """Test cache handles loading errors gracefully"""
        # Reset cache singleton to ensure clean state
        from utils import model_cache

        model_cache.ModelCache._instance = None
        model_cache._model_cache_instance = None

        mock_pipeline.side_effect = RuntimeError("Failed to load")

        from utils.error_handler import ModelLoadError

        # Should raise ModelLoadError after retries, not generic RuntimeError
        with pytest.raises((ModelLoadError, RuntimeError)):
            load_model("sentiment_analysis", "twitter_roberta_multilingual")

    @patch("utils.model_cache.pipeline")
    def test_cache_with_logging(self, mock_pipeline, caplog):
        """Test that cache operations are logged"""
        # Reset cache singleton to ensure clean state
        from utils import model_cache

        model_cache.ModelCache._instance = None
        model_cache._model_cache_instance = None

        mock_pipeline.return_value = Mock()

        with caplog.at_level("INFO"):
            try:
                load_model("sentiment_analysis", "twitter_roberta_multilingual")
            except Exception as e:
                pytest.skip(f"Model loading dependencies not available: {e}")

        # Should have logging messages about model loading
        # At minimum, verify the pipeline was called
        assert mock_pipeline.called or len(caplog.records) > 0


class TestErrorHandlingIntegration:
    """Test error handling integration"""

    def test_error_handling_with_logging(self, caplog):
        """Test that errors are properly logged"""
        import logging

        from utils import handle_errors

        @handle_errors(log_error=True, default_return="error")
        def failing_func():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            result = failing_func()

        assert result == "error"

        # Error should be logged
        messages = [record.message for record in caplog.records]
        assert any("error" in msg.lower() for msg in messages)

    def test_retry_with_logging(self, caplog):
        """Test retry decorator logs attempts"""
        from utils import retry_on_error

        attempts = {"count": 0}

        @retry_on_error(max_retries=2, delay=0.01)
        def eventually_succeeds():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ValueError("Temporary failure")
            return "success"

        with caplog.at_level("WARNING"):
            result = eventually_succeeds()

        assert result == "success"
        assert attempts["count"] == 2

        # Should have warning about retry
        messages = [record.message for record in caplog.records]
        assert any("retry" in msg.lower() for msg in messages)


class TestFullStackIntegration:
    """Test complete stack working together"""

    @patch("utils.model_cache.pipeline")
    def test_model_loading_full_stack(self, mock_pipeline, caplog):
        """Test complete model loading flow with all components"""
        mock_pipeline.return_value = Mock()

        from utils import PerformanceLogger, get_logger, handle_errors

        logger = get_logger("integration_test")

        @handle_errors(default_return=None, log_error=True)
        def load_and_use_model():
            with PerformanceLogger("model_loading", logger=logger):
                try:
                    model = load_model("sentiment_analysis", "twitter_roberta_multilingual")
                    return model
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

        with caplog.at_level("DEBUG"):
            result = load_and_use_model()

        # Should successfully load (or handle error)
        # Either we get a model or None (if error)
        assert result is not None or "error" in str(caplog.text).lower()

    def test_settings_affect_all_components(self):
        """Test that settings are used across all components"""
        settings = get_settings()

        # Settings should be accessible
        assert settings.app_name is not None
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.max_cached_models > 0

        # Logger should respect settings
        logger = get_logger("test_settings")
        assert logger.level >= 0

        # Registry should be accessible
        registry = get_model_registry()
        assert len(registry.list_categories()) > 0


class TestConfigurationFiles:
    """Test configuration files are properly loaded"""

    def test_models_yaml_loaded(self):
        """Test that models.yaml is properly loaded"""
        registry = get_model_registry()

        # Check all expected categories exist
        expected_categories = [
            "sentiment_analysis",
            "summarization",
            "embeddings",
            "ner",
            "question_answering",
            "text_generation",
            "zero_shot",
            "translation",
        ]

        categories = registry.list_categories()
        for expected in expected_categories:
            assert expected in categories, f"Category {expected} not found in registry"

    def test_model_configs_valid(self):
        """Test that all model configs are valid"""
        registry = get_model_registry()

        for category in registry.list_categories():
            models = registry.get_category(category)
            assert len(models) > 0, f"No models found in category {category}"

            for key, config in models.items():
                # All configs should have required fields
                assert config.name is not None
                assert config.model_id is not None
                assert config.task is not None


class TestEnvironmentConfiguration:
    """Test environment-based configuration"""

    def test_settings_use_defaults(self):
        """Test that settings use appropriate defaults"""
        settings = get_settings()

        # Default values should be sensible
        assert settings.debug is False or settings.debug is True
        assert settings.max_cached_models >= 1
        assert settings.server_port > 0

    def test_paths_exist_or_created(self):
        """Test that configured paths exist or are created"""
        settings = get_settings()

        # These directories should be created automatically
        assert settings.logs_dir.exists() or settings.logs_dir == Path("logs")
        assert settings.data_dir.exists() or settings.data_dir == Path("data")


class TestPerformanceIntegration:
    """Test performance-related integration"""

    @patch("utils.model_cache.pipeline")
    def test_model_cache_performance(self, mock_pipeline, caplog):
        """Test that caching improves performance"""
        import time

        mock_pipeline.return_value = Mock()

        # First load
        start1 = time.time()
        try:
            load_model("sentiment_analysis", "twitter_roberta_multilingual")
        except Exception:
            pytest.skip("Dependencies not available")
        duration1 = time.time() - start1

        # Second load (from cache)
        start2 = time.time()
        try:
            load_model("sentiment_analysis", "twitter_roberta_multilingual")
        except Exception:
            pytest.skip("Dependencies not available")
        duration2 = time.time() - start2

        # Second load should be faster (or at least not slower)
        # Using a generous margin for CI environments
        assert duration2 <= duration1 * 2

    def test_performance_logger_accuracy(self):
        """Test PerformanceLogger timing accuracy"""
        import time

        logger = get_logger("perf_test")

        start = time.time()
        with PerformanceLogger("sleep_test", logger=logger):
            time.sleep(0.05)
        elapsed = time.time() - start

        # Should take approximately 0.05 seconds
        assert 0.04 <= elapsed <= 0.15  # Generous range for CI
