"""
Pydantic schemas for API requests and responses
"""

from .requests import (
    BatchTextRequest,
    CustomerIntelligenceRequest,
    CustomerTextItem,
    NERRequest,
    QARequest,
    SentimentRequest,
    SimilarityRequest,
    SummarizationRequest,
)
from .responses import (
    APIError,
    CustomerInsightItem,
    CustomerIntelligenceResponse,
    CustomerIntelligenceSummary,
    EntityResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ModelsListResponse,
    NERResponse,
    QAResponse,
    SentimentResponse,
    SimilarityResponse,
    SummarizationResponse,
    TopEntity,
)

__all__ = [
    # Requests
    "SentimentRequest",
    "SummarizationRequest",
    "NERRequest",
    "SimilarityRequest",
    "QARequest",
    "BatchTextRequest",
    "CustomerTextItem",
    "CustomerIntelligenceRequest",
    # Responses
    "SentimentResponse",
    "SummarizationResponse",
    "NERResponse",
    "EntityResponse",
    "SimilarityResponse",
    "QAResponse",
    "TopEntity",
    "CustomerInsightItem",
    "CustomerIntelligenceSummary",
    "CustomerIntelligenceResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelsListResponse",
    "ModelInfo",
    "APIError",
]
