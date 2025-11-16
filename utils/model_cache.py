"""
Model Cache and Optimization utilities

Provides singleton pattern for model loading, caching, and memory management.
Supports automatic device detection (CPU, CUDA, MPS) and lazy loading.
"""

import gc
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
from transformers import Pipeline, pipeline

from config.settings import ModelConfig, get_model_registry, get_settings
from utils.error_handler import ErrorContext, ModelLoadError, ModelNotFoundError, retry_on_error
from utils.logger import PerformanceLogger, get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    Singleton cache for managing transformer models

    Features:
    - Lazy loading (models loaded on first use)
    - LRU eviction when cache is full
    - Automatic device detection
    - Thread-safe operations
    - Memory cleanup
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the cache"""
        if self._initialized:
            return

        self.settings = get_settings()
        self.model_registry = get_model_registry()

        # OrderedDict for LRU cache
        self._cache: OrderedDict[str, Pipeline] = OrderedDict()
        self._max_size = self.settings.max_cached_models
        self._device = self._detect_device()

        logger.info(f"ModelCache initialized with max_size={self._max_size}, device={self._device}")
        self._initialized = True

    def _detect_device(self) -> str:
        """
        Detect the best available device for model inference

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.settings.device != "auto":
            logger.info(f"Using configured device: {self.settings.device}")
            return self.settings.device

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available - using GPU: {gpu_name}")
            return device

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple MPS available - using Metal GPU")
            return device

        # Fallback to CPU
        device = "cpu"
        logger.info("No GPU available - using CPU")
        return device

    def _generate_cache_key(self, category: str, model_key: str, task_override: Optional[str] = None) -> str:
        """Generate unique cache key for a model"""
        task = task_override or f"{category}_{model_key}"
        return f"{category}::{model_key}::{task}"

    @retry_on_error(max_retries=2, delay=1.0)
    def _load_model(self, model_config: ModelConfig, task: Optional[str] = None) -> Pipeline:
        """
        Load a model from HuggingFace

        Args:
            model_config: Model configuration
            task: Optional task override

        Returns:
            Loaded pipeline

        Raises:
            ModelLoadError: If model fails to load
        """
        task = task or model_config.task
        model_id = model_config.model_id

        logger.info(f"Loading model: {model_config.name} ({model_id}) for task: {task}")

        try:
            with PerformanceLogger(f"load_model_{model_id}", logger=logger):
                # Determine device
                device = None if self._device == "cpu" else 0  # Use first GPU if available

                # Load pipeline with appropriate parameters
                model = pipeline(
                    task=task,
                    model=model_id,
                    device=device,
                    model_kwargs={"cache_dir": str(self.settings.cache_dir)},
                )

                logger.info(f"Successfully loaded: {model_config.name}")
                return model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise ModelLoadError(model_id, original_error=e)

    def _evict_lru(self):
        """Evict the least recently used model from cache"""
        if not self._cache:
            return

        # Remove oldest item (first in OrderedDict)
        cache_key, model = self._cache.popitem(last=False)
        logger.info(f"Evicting model from cache: {cache_key}")

        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model(
        self,
        category: str,
        model_key: str,
        task_override: Optional[str] = None,
        force_reload: bool = False,
    ) -> Pipeline:
        """
        Get a model from cache or load it

        Args:
            category: Model category (e.g., 'sentiment_analysis')
            model_key: Model key within category
            task_override: Optional task type override
            force_reload: Force reload even if cached

        Returns:
            Loaded model pipeline

        Raises:
            ModelNotFoundError: If model not found in registry
            ModelLoadError: If model fails to load
        """
        cache_key = self._generate_cache_key(category, model_key, task_override)

        # Check cache (unless force reload)
        if not force_reload and cache_key in self._cache:
            logger.debug(f"Cache hit: {cache_key}")
            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        logger.debug(f"Cache miss: {cache_key}")

        # Get model configuration
        try:
            model_config = self.model_registry.get_model(category, model_key)
        except KeyError as e:
            raise ModelNotFoundError(model_key, category) from e

        # Evict if cache is full
        while len(self._cache) >= self._max_size:
            self._evict_lru()

        # Load model
        with ErrorContext("model_loading", category=category, model_key=model_key):
            model = self._load_model(model_config, task=task_override)

        # Add to cache
        self._cache[cache_key] = model
        logger.info(f"Added to cache: {cache_key} (cache size: {len(self._cache)}/{self._max_size})")

        return model

    def get_model_by_id(self, model_id: str, task: str, force_reload: bool = False) -> Pipeline:
        """
        Get a model by HuggingFace model ID (bypass registry)

        Args:
            model_id: HuggingFace model ID
            task: Task type
            force_reload: Force reload even if cached

        Returns:
            Loaded model pipeline
        """
        cache_key = f"direct::{model_id}::{task}"

        # Check cache
        if not force_reload and cache_key in self._cache:
            logger.debug(f"Cache hit: {cache_key}")
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Evict if cache is full
        while len(self._cache) >= self._max_size:
            self._evict_lru()

        # Load model directly
        logger.info(f"Loading model by ID: {model_id} (task: {task})")
        try:
            device = None if self._device == "cpu" else 0
            model = pipeline(task=task, model=model_id, device=device)
        except Exception as e:
            raise ModelLoadError(model_id, original_error=e)

        # Add to cache
        self._cache[cache_key] = model
        logger.info(f"Added to cache: {cache_key}")

        return model

    def clear_cache(self):
        """Clear all cached models"""
        logger.info(f"Clearing model cache ({len(self._cache)} models)")
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_model(self, category: str, model_key: str):
        """Remove a specific model from cache"""
        cache_key = self._generate_cache_key(category, model_key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.info(f"Removed from cache: {cache_key}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def warmup(self, models: list[tuple[str, str]]):
        """
        Pre-load models into cache

        Args:
            models: List of (category, model_key) tuples to preload
        """
        logger.info(f"Warming up cache with {len(models)} models")
        for category, model_key in models:
            try:
                self.get_model(category, model_key)
            except Exception as e:
                logger.warning(f"Failed to warmup {category}::{model_key}: {str(e)}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_models": len(self._cache),
            "max_size": self._max_size,
            "device": self._device,
            "models": list(self._cache.keys()),
        }

    @property
    def device(self) -> str:
        """Get current device"""
        return self._device


# Singleton instance getter
_model_cache_instance: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """
    Get the global model cache instance

    Returns:
        ModelCache singleton
    """
    global _model_cache_instance
    if _model_cache_instance is None:
        _model_cache_instance = ModelCache()
    return _model_cache_instance


# Convenience functions
def load_model(category: str, model_key: str, **kwargs) -> Pipeline:
    """
    Load a model from the cache

    Args:
        category: Model category
        model_key: Model key
        **kwargs: Additional arguments passed to get_model

    Returns:
        Loaded model pipeline
    """
    cache = get_model_cache()
    return cache.get_model(category, model_key, **kwargs)


def load_model_by_id(model_id: str, task: str, **kwargs) -> Pipeline:
    """
    Load a model by HuggingFace ID

    Args:
        model_id: HuggingFace model ID
        task: Task type
        **kwargs: Additional arguments

    Returns:
        Loaded model pipeline
    """
    cache = get_model_cache()
    return cache.get_model_by_id(model_id, task, **kwargs)


def clear_model_cache():
    """Clear all cached models"""
    cache = get_model_cache()
    cache.clear_cache()


def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics"""
    cache = get_model_cache()
    return cache.get_cache_stats()
