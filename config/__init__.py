"""
Configuration package for NLP Transformers Examples

This package provides centralized configuration management for all models,
settings, and application parameters.
"""

from .settings import ModelConfig, ModelRegistry, Settings, get_model_registry, get_settings

__all__ = ["Settings", "get_settings", "ModelRegistry", "get_model_registry", "ModelConfig"]
