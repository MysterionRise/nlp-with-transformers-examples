"""
Unit tests for model caching system
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from config import ModelConfig
from utils.error_handler import ModelLoadError, ModelNotFoundError
from utils.model_cache import ModelCache, clear_model_cache, get_cache_info, get_model_cache, load_model


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline"""
    return Mock()


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration"""
    return ModelConfig(
        name="Test Model",
        model_id="test/model",
        task="text-classification",
        description="A test model",
        max_length=512,
    )


@pytest.fixture
def fresh_cache():
    """Reset ModelCache singleton for each test"""
    # Reset the singleton
    ModelCache._instance = None
    yield
    # Clean up after test
    if ModelCache._instance is not None:
        ModelCache._instance._cache.clear()
    ModelCache._instance = None


class TestModelCacheInitialization:
    """Test ModelCache initialization"""

    def test_model_cache_singleton(self, fresh_cache):
        """Test that ModelCache is a singleton"""
        cache1 = ModelCache()
        cache2 = ModelCache()
        assert cache1 is cache2

    def test_model_cache_device_detection(self, fresh_cache):
        """Test automatic device detection"""
        with patch("utils.model_cache.torch") as mock_torch:
            # Simulate no GPU available
            mock_torch.cuda.is_available.return_value = False
            if hasattr(mock_torch.backends, "mps"):
                mock_torch.backends.mps.is_available.return_value = False

            cache = ModelCache()
            assert cache.device in ["cpu", "cuda", "mps", "auto"]

    def test_model_cache_max_size(self, fresh_cache):
        """Test cache size configuration"""
        cache = ModelCache()
        assert cache._max_size > 0


class TestModelCacheLoading:
    """Test model loading and caching"""

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_first_load(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test loading a model for the first time"""
        # Setup mocks
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        cache = ModelCache()
        model = cache.get_model("sentiment_analysis", "test_model")

        # Verify model was loaded
        assert model == mock_pipeline_instance
        mock_pipeline.assert_called_once()

        # Verify model is in cache
        assert len(cache._cache) == 1

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_from_cache(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test loading a model from cache (second access)"""
        # Setup mocks
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        cache = ModelCache()

        # First load
        model1 = cache.get_model("sentiment_analysis", "test_model")
        call_count_after_first = mock_pipeline.call_count

        # Second load (should come from cache)
        model2 = cache.get_model("sentiment_analysis", "test_model")

        # Pipeline should not be called again
        assert mock_pipeline.call_count == call_count_after_first
        assert model1 is model2

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_force_reload(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test forcing model reload"""
        # Setup mocks
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()

        # First load
        cache.get_model("sentiment_analysis", "test_model")
        first_call_count = mock_pipeline.call_count

        # Force reload
        cache.get_model("sentiment_analysis", "test_model", force_reload=True)

        # Pipeline should be called again
        assert mock_pipeline.call_count > first_call_count

    @patch("utils.model_cache.get_model_registry")
    def test_get_model_not_found(self, mock_registry, fresh_cache):
        """Test loading a non-existent model"""
        # Setup mock to raise KeyError
        mock_registry.return_value.get_model.side_effect = KeyError("Model not found")

        cache = ModelCache()

        with pytest.raises(ModelNotFoundError):
            cache.get_model("sentiment_analysis", "nonexistent_model")

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_load_model_failure(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test handling of model loading failures"""
        # Setup mocks
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline.side_effect = RuntimeError("Failed to load")

        cache = ModelCache()

        with pytest.raises(ModelLoadError):
            cache.get_model("sentiment_analysis", "test_model")


class TestModelCacheEviction:
    """Test LRU cache eviction"""

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_lru_eviction(self, mock_registry, mock_pipeline, fresh_cache):
        """Test that least recently used model is evicted"""
        # Create mock configs
        def create_config(name):
            return ModelConfig(
                name=name, model_id=f"test/{name}", task="text-classification", description="Test"
            )

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.side_effect = lambda **kwargs: Mock()

        cache = ModelCache()
        cache._max_size = 2  # Set small cache size

        # Load 3 models (should evict the first one)
        cache.get_model("test", "model1")
        cache.get_model("test", "model2")
        cache.get_model("test", "model3")  # This should evict model1

        # Cache should only have 2 models
        assert len(cache._cache) == 2

        # model1 should not be in cache
        cache_keys = list(cache._cache.keys())
        assert not any("model1" in key for key in cache_keys)

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_lru_access_updates_order(self, mock_registry, mock_pipeline, fresh_cache):
        """Test that accessing a model updates its position in LRU"""
        def create_config(name):
            return ModelConfig(
                name=name, model_id=f"test/{name}", task="text-classification", description="Test"
            )

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.side_effect = lambda **kwargs: Mock()

        cache = ModelCache()
        cache._max_size = 2

        # Load 2 models
        cache.get_model("test", "model1")
        cache.get_model("test", "model2")

        # Access model1 again (should move it to end)
        cache.get_model("test", "model1")

        # Load model3 (should evict model2, not model1)
        cache.get_model("test", "model3")

        # Cache should still contain model1
        cache_keys = list(cache._cache.keys())
        assert any("model1" in key for key in cache_keys)
        assert not any("model2" in key for key in cache_keys)


class TestModelCacheManagement:
    """Test cache management operations"""

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_clear_cache(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test clearing the cache"""
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 1

        cache.clear_cache()

        assert len(cache._cache) == 0

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_remove_specific_model(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test removing a specific model from cache"""
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 1

        cache.remove_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 0

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_cache_stats(self, mock_registry, mock_pipeline, fresh_cache, mock_model_config):
        """Test getting cache statistics"""
        mock_registry.return_value.get_model.return_value = mock_model_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        stats = cache.get_cache_stats()

        assert isinstance(stats, dict)
        assert "cached_models" in stats
        assert "max_size" in stats
        assert "device" in stats
        assert "models" in stats
        assert stats["cached_models"] == 1

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_warmup(self, mock_registry, mock_pipeline, fresh_cache):
        """Test warming up cache with multiple models"""

        def create_config(key):
            return ModelConfig(
                name=key, model_id=f"test/{key}", task="text-classification", description="Test"
            )

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        models_to_warmup = [("sentiment", "model1"), ("sentiment", "model2")]

        cache.warmup(models_to_warmup)

        # Both models should be in cache
        assert len(cache._cache) == 2


class TestModelCacheByID:
    """Test loading models by HuggingFace ID"""

    @patch("utils.model_cache.pipeline")
    def test_get_model_by_id(self, mock_pipeline, fresh_cache):
        """Test loading model by ID (bypass registry)"""
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        model = cache.get_model_by_id("bert-base-uncased", "text-classification")

        assert model is not None
        mock_pipeline.assert_called_once()

    @patch("utils.model_cache.pipeline")
    def test_get_model_by_id_caching(self, mock_pipeline, fresh_cache):
        """Test that models loaded by ID are cached"""
        mock_pipeline.return_value = Mock()

        cache = ModelCache()

        # Load twice
        cache.get_model_by_id("bert-base-uncased", "text-classification")
        cache.get_model_by_id("bert-base-uncased", "text-classification")

        # Pipeline should only be called once (second load from cache)
        assert mock_pipeline.call_count == 1


class TestConvenienceFunctions:
    """Test convenience functions"""

    @patch("utils.model_cache.get_model_cache")
    def test_load_model_function(self, mock_get_cache):
        """Test load_model convenience function"""
        mock_cache = Mock()
        mock_cache.get_model.return_value = Mock()
        mock_get_cache.return_value = mock_cache

        result = load_model("sentiment_analysis", "test_model")

        mock_cache.get_model.assert_called_once_with("sentiment_analysis", "test_model")
        assert result is not None

    @patch("utils.model_cache.get_model_cache")
    def test_clear_model_cache_function(self, mock_get_cache):
        """Test clear_model_cache convenience function"""
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        clear_model_cache()

        mock_cache.clear_cache.assert_called_once()

    @patch("utils.model_cache.get_model_cache")
    def test_get_cache_info_function(self, mock_get_cache):
        """Test get_cache_info convenience function"""
        mock_cache = Mock()
        mock_cache.get_cache_stats.return_value = {"cached_models": 2}
        mock_get_cache.return_value = mock_cache

        info = get_cache_info()

        mock_cache.get_cache_stats.assert_called_once()
        assert info["cached_models"] == 2
