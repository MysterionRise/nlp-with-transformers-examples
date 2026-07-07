"""
Text Summarization API Router

Provides endpoints for generating text summaries using transformer models.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import SummarizationRequest
from api.schemas.responses import SummarizationResponse
from utils.inference import DEFAULT_SUMMARIZATION_MODEL, get_inference_service
from utils.metrics import record_error

router = APIRouter(prefix="/api/v1", tags=["Summarization"])


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    summary="Summarize text",
    description="Generate a concise summary of the provided text using transformer models. "
    "Supports adjustable summary length parameters.",
    responses={
        200: {
            "description": "Successful summarization",
            "content": {
                "application/json": {
                    "example": {
                        "original_text": "A long article about AI...",
                        "summary": "AI has transformed industries through automation and decision-making.",
                        "original_length": 450,
                        "summary_length": 85,
                        "compression_ratio": 0.189,
                        "model": "facebook/bart-large-cnn",
                        "processing_time_ms": 1250.5,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def summarize_text(
    request: Request,
    body: SummarizationRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> SummarizationResponse:
    """
    Generate a summary of the provided text

    Uses state-of-the-art summarization models like BART, T5, or Pegasus.
    You can control the summary length with min_length and max_length parameters.

    **Authentication required:** API key or JWT token

    **Rate limits apply based on your API key tier.**
    """
    model_key = body.model or DEFAULT_SUMMARIZATION_MODEL

    try:
        result = await get_inference_service().summarize(
            body.text,
            model_key=model_key,
            min_length=body.min_length,
            max_length=body.max_length,
        )

        return SummarizationResponse(
            original_text=result["original_text"],
            summary=result["summary"],
            original_length=result["original_length"],
            summary_length=result["summary_length"],
            compression_ratio=result["compression_ratio"],
            model=result["model"],
            processing_time_ms=result["processing_time_ms"],
        )

    except KeyError:
        from config.settings import get_model_registry

        registry = get_model_registry()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found in summarization category",
                "available_models": registry.list_models("summarization"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/summarize")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to summarize text: {str(e)}",
            },
        )
