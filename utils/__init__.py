"""
Utilities package for NLP Transformers Examples

This package provides shared utilities for error handling, logging,
model caching, and other common functionality.
"""

from .error_handler import (
    ConfigurationError,
    ErrorContext,
    InferenceError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    NLPError,
    ResourceError,
    format_error_message,
    get_error_suggestion,
    handle_errors,
    retry_on_error,
    validate_input,
)
from .logger import PerformanceLogger, get_logger, init_logging, log_function_call, setup_logger
from .model_cache import (
    ModelCache,
    clear_model_cache,
    get_cache_info,
    get_model_cache,
    load_model,
    load_model_by_id,
)

__all__ = [
    # Error handling
    "NLPError",
    "ModelLoadError",
    "ModelNotFoundError",
    "InferenceError",
    "ConfigurationError",
    "InvalidInputError",
    "ResourceError",
    "handle_errors",
    "retry_on_error",
    "validate_input",
    "ErrorContext",
    "format_error_message",
    "get_error_suggestion",
    # Logging
    "get_logger",
    "setup_logger",
    "init_logging",
    "log_function_call",
    "PerformanceLogger",
    # Model caching
    "ModelCache",
    "get_model_cache",
    "load_model",
    "load_model_by_id",
    "clear_model_cache",
    "get_cache_info",
]
