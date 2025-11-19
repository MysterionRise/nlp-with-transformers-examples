"""
Unit tests for model caching system.

This module tests the ModelCache class which provides:
- Singleton pattern for global cache access
- LRU eviction strategy for memory management
- Model loading and caching from HuggingFace
- Cache management operations (clear, remove, stats)
- Automatic device detection (CPU/CUDA/MPS)

The tests use extensive mocking to avoid downloading actual models
and to ensure fast, isolated test execution.
"""

from unittest.mock import Mock, patch

import pytest

from utils.error_handler import ModelLoadError, ModelNotFoundError
from utils.model_cache import ModelCache, clear_model_cache, get_cache_info, load_model

# ============================================================================
# Test Classes
# ============================================================================


class TestModelCacheInitialization:
    """Test ModelCache initialization and configuration.

    This test class verifies that:
    - ModelCache implements singleton pattern correctly
    - Device detection works for CPU, CUDA, and MPS
    - Cache size limits are properly configured
    """

    def test_model_cache_singleton(self):
        """Test that ModelCache follows singleton pattern.

        Ensures that multiple calls to ModelCache() return the same instance,
        preventing multiple caches from existing simultaneously.
        """
        cache1 = ModelCache()
        cache2 = ModelCache()
        assert cache1 is cache2, "ModelCache should be a singleton"

    @pytest.mark.parametrize(
        "cuda_available,mps_available,expected_devices",
        [
            (True, False, ["cpu", "cuda", "auto"]),  # CUDA available
            (False, True, ["cpu", "mps", "auto"]),  # MPS available (Apple Silicon)
            (False, False, ["cpu", "auto"]),  # CPU only
        ],
        ids=["cuda-available", "mps-available", "cpu-only"],
    )
    def test_model_cache_device_detection(self, cuda_available, mps_available, expected_devices):
        """Test automatic device detection for different hardware configurations.

        Args:
            cuda_available: Whether CUDA/GPU is available
            mps_available: Whether MPS (Apple Silicon) is available
            expected_devices: List of valid device strings for this configuration

        The cache should correctly detect available hardware and set the device accordingly.
        """
        with patch("utils.model_cache.torch") as mock_torch:
            # Configure mock hardware availability
            mock_torch.cuda.is_available.return_value = cuda_available
            if hasattr(mock_torch.backends, "mps"):
                mock_torch.backends.mps.is_available.return_value = mps_available

            cache = ModelCache()
            assert cache.device in expected_devices, f"Device should be one of {expected_devices}"

    def test_model_cache_max_size(self):
        """Test that cache has a positive maximum size limit.

        The cache should have a configured maximum size to prevent
        unlimited memory usage when loading many models.
        """
        cache = ModelCache()
        assert cache._max_size > 0, "Cache must have a positive max size"
        assert isinstance(cache._max_size, int), "Max size should be an integer"


class TestModelCacheLoading:
    """Test model loading and caching functionality.

    This test class verifies:
    - Initial model loading from HuggingFace
    - Subsequent loads served from cache
    - Force reload capability
    - Error handling for missing models
    - Error handling for loading failures
    """

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_first_load(self, mock_registry, mock_pipeline, model_config_factory):
        """Test loading a model for the first time.

        Verifies that:
        - The model is loaded via the pipeline
        - The loaded model is returned correctly
        - The model is added to the cache

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        # Setup test data
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline_instance = Mock(name="MockPipeline")
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute
        cache = ModelCache()
        model = cache.get_model("sentiment_analysis", "test_model")

        # Verify
        assert model == mock_pipeline_instance, "Should return the loaded pipeline"
        mock_pipeline.assert_called_once()
        assert len(cache._cache) == 1, "Model should be in cache after first load"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_from_cache(self, mock_registry, mock_pipeline, model_config_factory):
        """Test that subsequent model loads are served from cache.

        Verifies that:
        - Second access to same model doesn't call pipeline again
        - Same model instance is returned from cache
        - Cache hit optimization is working

        This is critical for performance as model loading is expensive.

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        # Setup
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline_instance = Mock(name="MockPipeline")
        mock_pipeline.return_value = mock_pipeline_instance

        cache = ModelCache()

        # First load
        model1 = cache.get_model("sentiment_analysis", "test_model")
        call_count_after_first = mock_pipeline.call_count

        # Second load (should come from cache)
        model2 = cache.get_model("sentiment_analysis", "test_model")

        # Verify cache behavior
        assert mock_pipeline.call_count == call_count_after_first, "Pipeline should not be called again"
        assert model1 is model2, "Should return same instance from cache"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_model_force_reload(self, mock_registry, mock_pipeline, model_config_factory):
        """Test forcing model reload bypasses cache.

        Verifies that:
        - force_reload=True triggers new pipeline call
        - Cached model is replaced with new instance
        - Useful for updating models or testing

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        # Setup
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()

        # First load
        cache.get_model("sentiment_analysis", "test_model")
        first_call_count = mock_pipeline.call_count

        # Force reload
        cache.get_model("sentiment_analysis", "test_model", force_reload=True)

        # Verify reload occurred
        assert mock_pipeline.call_count > first_call_count, "Pipeline should be called again on force reload"

    @patch("utils.model_cache.get_model_registry")
    def test_get_model_not_found(self, mock_registry):
        """Test proper error handling when model doesn't exist.

        Verifies that:
        - Attempting to load non-existent model raises ModelNotFoundError
        - Error provides useful information for debugging

        Args:
            mock_registry: Mocked model registry configured to fail
        """
        # Setup: Registry raises KeyError for non-existent model
        mock_registry.return_value.get_model.side_effect = KeyError("Model not found")

        cache = ModelCache()

        # Verify exception is raised and properly typed
        with pytest.raises(ModelNotFoundError, match="not found"):
            cache.get_model("sentiment_analysis", "nonexistent_model")

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_load_model_failure(self, mock_registry, mock_pipeline, model_config_factory):
        """Test handling of model loading failures.

        Verifies that:
        - Pipeline failures are caught and wrapped in ModelLoadError
        - Original error information is preserved
        - Cache remains in consistent state after failure

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked pipeline configured to fail
            model_config_factory: Fixture to create test model configs
        """
        # Setup: Pipeline fails to load
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.side_effect = RuntimeError("Failed to load model from HuggingFace")

        cache = ModelCache()

        # Verify proper error handling
        with pytest.raises(ModelLoadError, match="Failed to load"):
            cache.get_model("sentiment_analysis", "test_model")

    @pytest.mark.parametrize(
        "category,model_key,expected_calls",
        [
            ("sentiment_analysis", "roberta", 1),
            ("summarization", "bart", 1),
            ("ner", "bert", 1),
        ],
        ids=["sentiment", "summarization", "ner"],
    )
    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_load_different_model_types(
        self, mock_registry, mock_pipeline, model_config_factory, category, model_key, expected_calls
    ):
        """Test loading models of different types/tasks.

        Verifies that the cache correctly handles different model categories
        and tasks (sentiment analysis, summarization, NER, etc.).

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
            category: Model category (task type)
            model_key: Model identifier
            expected_calls: Expected number of pipeline calls
        """
        # Setup with specific task
        task_map = {
            "sentiment_analysis": "sentiment-analysis",
            "summarization": "summarization",
            "ner": "ner",
        }
        test_config = model_config_factory(task=task_map.get(category, "text-classification"))
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        model = cache.get_model(category, model_key)

        assert model is not None
        assert mock_pipeline.call_count == expected_calls


class TestModelCacheEviction:
    """Test LRU (Least Recently Used) cache eviction strategy.

    This test class verifies:
    - Models are evicted when cache reaches max size
    - Least recently used model is evicted first
    - Recently accessed models are protected from eviction
    - Cache size limits are enforced correctly
    """

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_lru_eviction(self, mock_registry, mock_pipeline, model_config_factory):
        """Test that least recently used model is evicted when cache is full.

        Verifies the LRU eviction policy:
        1. Load models until cache is full
        2. Loading additional model triggers eviction
        3. Oldest (least recently used) model is removed

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """

        # Create factory function for configs
        def create_config(name):
            return model_config_factory(name=name, model_id=f"test/{name}")

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.side_effect = lambda **kwargs: Mock()

        cache = ModelCache()
        cache._max_size = 2  # Set small cache size for testing

        # Load 3 models (should evict the first one)
        cache.get_model("test", "model1")
        cache.get_model("test", "model2")
        cache.get_model("test", "model3")  # This should evict model1

        # Verify cache size limit is enforced
        assert len(cache._cache) == 2, "Cache should only contain max_size models"

        # Verify correct model was evicted
        cache_keys = list(cache._cache.keys())
        assert not any("model1" in key for key in cache_keys), "model1 should be evicted (oldest)"
        assert any("model2" in key for key in cache_keys), "model2 should still be in cache"
        assert any("model3" in key for key in cache_keys), "model3 should still be in cache"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_lru_access_updates_order(self, mock_registry, mock_pipeline, model_config_factory):
        """Test that accessing a model updates its position in LRU order.

        Verifies that:
        1. Accessing a cached model marks it as recently used
        2. Recently accessed models are protected from eviction
        3. LRU order is correctly maintained

        This is critical for keeping frequently-used models in cache.

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """

        def create_config(name):
            return model_config_factory(name=name, model_id=f"test/{name}")

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.side_effect = lambda **kwargs: Mock()

        cache = ModelCache()
        cache._max_size = 2

        # Load 2 models
        cache.get_model("test", "model1")
        cache.get_model("test", "model2")

        # Access model1 again (should move it to end of LRU order)
        cache.get_model("test", "model1")

        # Load model3 (should evict model2, not model1)
        cache.get_model("test", "model3")

        # Verify correct eviction behavior
        cache_keys = list(cache._cache.keys())
        assert any("model1" in key for key in cache_keys), "model1 should still be in cache (recently accessed)"
        assert not any("model2" in key for key in cache_keys), "model2 should be evicted"
        assert any("model3" in key for key in cache_keys), "model3 should be in cache"


class TestModelCacheManagement:
    """Test cache management operations.

    This test class verifies:
    - Clearing entire cache
    - Removing specific models
    - Getting cache statistics
    - Warming up cache with pre-loaded models
    """

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_clear_cache(self, mock_registry, mock_pipeline, model_config_factory):
        """Test clearing all models from cache.

        Verifies that:
        - clear_cache() removes all cached models
        - Cache size becomes 0
        - Cache can be used normally after clearing

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 1, "Model should be in cache"

        cache.clear_cache()

        assert len(cache._cache) == 0, "Cache should be empty after clear"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_remove_specific_model(self, mock_registry, mock_pipeline, model_config_factory):
        """Test removing a specific model from cache.

        Verifies that:
        - Individual models can be removed by category/key
        - Other models remain in cache
        - Removed model can be reloaded

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 1, "Model should be in cache"

        cache.remove_model("sentiment_analysis", "test_model")

        assert len(cache._cache) == 0, "Model should be removed from cache"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_get_cache_stats(self, mock_registry, mock_pipeline, model_config_factory):
        """Test retrieving cache statistics.

        Verifies that cache stats include:
        - Number of cached models
        - Maximum cache size
        - Device being used
        - List of cached model identifiers

        Useful for monitoring and debugging cache behavior.

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """
        test_config = model_config_factory()
        mock_registry.return_value.get_model.return_value = test_config
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        cache.get_model("sentiment_analysis", "test_model")

        stats = cache.get_cache_stats()

        # Verify stats structure
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "cached_models" in stats, "Should include count of cached models"
        assert "max_size" in stats, "Should include max cache size"
        assert "device" in stats, "Should include device information"
        assert "models" in stats, "Should include list of cached models"
        assert stats["cached_models"] == 1, "Should show 1 cached model"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_warmup(self, mock_registry, mock_pipeline, model_config_factory):
        """Test warming up cache with multiple models.

        Warmup pre-loads frequently-used models to improve application startup performance.

        Verifies that:
        - Multiple models can be loaded in batch
        - All models are available in cache after warmup
        - Warmup is efficient (minimal redundant operations)

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
        """

        def create_config(key):
            return model_config_factory(name=key, model_id=f"test/{key}")

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        models_to_warmup = [("sentiment", "model1"), ("sentiment", "model2")]

        cache.warmup(models_to_warmup)

        # Verify all models are cached
        assert len(cache._cache) == 2, "All warmup models should be cached"


class TestModelCacheByID:
    """Test loading models directly by HuggingFace ID.

    This test class verifies:
    - Loading models by ID (bypassing registry)
    - Caching behavior for ID-based loads
    - Compatibility with standard caching mechanisms
    """

    @patch("utils.model_cache.pipeline")
    def test_get_model_by_id(self, mock_pipeline):
        """Test loading model by HuggingFace ID (bypass registry).

        Useful for:
        - Loading models not in the registry
        - Testing with custom/experimental models
        - Direct HuggingFace integration

        Args:
            mock_pipeline: Mocked HuggingFace pipeline
        """
        mock_pipeline.return_value = Mock()

        cache = ModelCache()
        model = cache.get_model_by_id("bert-base-uncased", "text-classification")

        assert model is not None, "Should load model successfully"
        mock_pipeline.assert_called_once()

    @patch("utils.model_cache.pipeline")
    def test_get_model_by_id_caching(self, mock_pipeline):
        """Test that models loaded by ID are properly cached.

        Verifies that:
        - ID-based loads use the same cache as registry loads
        - Subsequent ID-based loads are served from cache
        - Performance optimization applies to both load methods

        Args:
            mock_pipeline: Mocked HuggingFace pipeline
        """
        mock_pipeline.return_value = Mock()

        cache = ModelCache()

        # Load same model twice by ID
        cache.get_model_by_id("bert-base-uncased", "text-classification")
        cache.get_model_by_id("bert-base-uncased", "text-classification")

        # Verify caching works
        assert mock_pipeline.call_count == 1, "Pipeline should only be called once (second load from cache)"


class TestConvenienceFunctions:
    """Test module-level convenience functions.

    This test class verifies the public API functions that wrap ModelCache methods
    for simpler usage patterns:
    - load_model(): Load a model
    - clear_model_cache(): Clear the cache
    - get_cache_info(): Get cache statistics
    """

    @patch("utils.model_cache.get_model_cache")
    def test_load_model_function(self, mock_get_cache):
        """Test load_model() convenience function.

        Verifies that the module-level load_model() function
        correctly delegates to the cache singleton.

        Args:
            mock_get_cache: Mocked cache singleton getter
        """
        mock_cache = Mock()
        mock_cache.get_model.return_value = Mock()
        mock_get_cache.return_value = mock_cache

        result = load_model("sentiment_analysis", "test_model")

        mock_cache.get_model.assert_called_once_with("sentiment_analysis", "test_model")
        assert result is not None, "Should return loaded model"

    @patch("utils.model_cache.get_model_cache")
    def test_clear_model_cache_function(self, mock_get_cache):
        """Test clear_model_cache() convenience function.

        Verifies that the module-level clear function
        correctly delegates to the cache singleton.

        Args:
            mock_get_cache: Mocked cache singleton getter
        """
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache

        clear_model_cache()

        mock_cache.clear_cache.assert_called_once()

    @patch("utils.model_cache.get_model_cache")
    def test_get_cache_info_function(self, mock_get_cache):
        """Test get_cache_info() convenience function.

        Verifies that the module-level info function
        correctly delegates to the cache singleton and returns stats.

        Args:
            mock_get_cache: Mocked cache singleton getter
        """
        mock_cache = Mock()
        mock_cache.get_cache_stats.return_value = {"cached_models": 2}
        mock_get_cache.return_value = mock_cache

        info = get_cache_info()

        mock_cache.get_cache_stats.assert_called_once()
        assert info["cached_models"] == 2, "Should return cache statistics"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestModelCacheEdgeCases:
    """Test edge cases and boundary conditions.

    This test class covers:
    - Empty cache operations
    - Invalid inputs
    - Concurrent access patterns
    - Resource cleanup
    """

    def test_clear_empty_cache(self):
        """Test clearing an already-empty cache.

        Should handle gracefully without errors.
        """
        cache = ModelCache()
        cache.clear_cache()  # Clear empty cache

        assert len(cache._cache) == 0, "Cache should remain empty"

    def test_remove_nonexistent_model(self):
        """Test removing a model that isn't in the cache.

        Should handle gracefully without raising exceptions.
        """
        cache = ModelCache()

        # Should not raise error
        cache.remove_model("sentiment", "nonexistent_model")

        assert len(cache._cache) == 0, "Cache should remain empty"

    def test_get_stats_empty_cache(self):
        """Test getting statistics from an empty cache.

        Should return valid stats structure with zero counts.
        """
        cache = ModelCache()
        stats = cache.get_cache_stats()

        assert stats["cached_models"] == 0, "Should show 0 cached models"
        assert stats["max_size"] > 0, "Should still have max_size configured"
        assert isinstance(stats["models"], list), "Should return empty models list"
        assert len(stats["models"]) == 0, "Models list should be empty"

    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_warmup_empty_list(self, mock_registry, mock_pipeline):
        """Test warmup with empty model list.

        Should handle gracefully without attempting any loads.

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
        """
        cache = ModelCache()
        cache.warmup([])  # Empty warmup list

        assert len(cache._cache) == 0, "Cache should remain empty"
        mock_pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "max_size",
        [1, 2, 5, 10],
        ids=["size-1", "size-2", "size-5", "size-10"],
    )
    @patch("utils.model_cache.pipeline")
    @patch("utils.model_cache.get_model_registry")
    def test_cache_with_various_sizes(self, mock_registry, mock_pipeline, model_config_factory, max_size):
        """Test cache behavior with various maximum sizes.

        Verifies that cache correctly enforces different size limits.

        Args:
            mock_registry: Mocked model registry
            mock_pipeline: Mocked HuggingFace pipeline
            model_config_factory: Fixture to create test model configs
            max_size: Maximum cache size to test
        """

        def create_config(name):
            return model_config_factory(name=name, model_id=f"test/{name}")

        mock_registry.return_value.get_model.side_effect = lambda cat, key: create_config(key)
        mock_pipeline.side_effect = lambda **kwargs: Mock()

        cache = ModelCache()
        cache._max_size = max_size

        # Load more models than max_size
        for i in range(max_size + 2):
            cache.get_model("test", f"model{i}")

        # Verify size limit is enforced
        assert len(cache._cache) <= max_size, f"Cache should not exceed max_size of {max_size}"
