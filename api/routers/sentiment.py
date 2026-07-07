"""
Sentiment Analysis API Router

Provides endpoints for analyzing text sentiment using transformer models.
"""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import BatchTextRequest, SentimentRequest
from api.schemas.responses import SentimentResponse, SentimentResult
from utils.inference import DEFAULT_SENTIMENT_MODEL, get_inference_service
from utils.metrics import record_error

router = APIRouter(prefix="/api/v1", tags=["Sentiment Analysis"])


async def run_sentiment_inference(text: str, model_key: str) -> tuple[dict, str, float]:
    """
    Run sentiment inference asynchronously

    Args:
        text: Text to analyze
        model_key: Model key from registry

    Returns:
        Tuple of (result, model_id, processing_time_ms)
    """
    # Retained for backward-compatible internal imports; endpoints use normalized service output.
    service = get_inference_service()
    result = await service.analyze_sentiment(text, model_key)
    return result, result["model"], result["processing_time_ms"]


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze text sentiment",
    description="Analyze the sentiment of a text using transformer models. "
    "Returns sentiment label (positive/negative/neutral) with confidence scores.",
    responses={
        200: {
            "description": "Successful sentiment analysis",
            "content": {
                "application/json": {
                    "example": {
                        "text": "I love this product!",
                        "sentiment": {"label": "positive", "score": 0.9876},
                        "all_scores": [
                            {"label": "positive", "score": 0.9876},
                            {"label": "neutral", "score": 0.0098},
                            {"label": "negative", "score": 0.0026},
                        ],
                        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
                        "processing_time_ms": 45.2,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def analyze_sentiment(
    request: Request,
    body: SentimentRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> SentimentResponse:
    """
    Analyze sentiment of provided text

    Supports multiple sentiment models. If no model is specified,
    uses the default Twitter RoBERTa multilingual model.

    **Authentication required:** API key or JWT token

    **Rate limits apply based on your API key tier.**
    """
    model_key = body.model or DEFAULT_SENTIMENT_MODEL

    try:
        result = await get_inference_service().analyze_sentiment(body.text, model_key)

        return SentimentResponse(
            text=body.text,
            sentiment=SentimentResult(**result["sentiment"]),
            all_scores=[SentimentResult(**score) for score in result["all_scores"]] if result["all_scores"] else None,
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
                "message": f"Model '{model_key}' not found in sentiment_analysis category",
                "available_models": registry.list_models("sentiment_analysis"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/sentiment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to analyze sentiment: {str(e)}",
            },
        )


@router.post(
    "/sentiment/batch",
    response_model=List[SentimentResponse],
    summary="Batch sentiment analysis",
    description="Analyze sentiment for multiple texts in a single request. "
    "More efficient than multiple individual requests.",
    responses={
        200: {"description": "Successful batch analysis"},
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def analyze_sentiment_batch(
    request: Request,
    body: BatchTextRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> List[SentimentResponse]:
    """
    Batch sentiment analysis for multiple texts

    Processes up to 100 texts in a single request.
    Results are returned in the same order as input texts.

    **Authentication required:** API key or JWT token

    **Rate limits apply - batch requests count as one request.**
    """
    model_key = body.model or DEFAULT_SENTIMENT_MODEL

    try:
        results = await get_inference_service().analyze_sentiment_batch(body.texts, model_key)
        return [
            SentimentResponse(
                text=result["text"],
                sentiment=SentimentResult(**result["sentiment"]),
                all_scores=(
                    [SentimentResult(**score) for score in result["all_scores"]] if result["all_scores"] else None
                ),
                model=result["model"],
                processing_time_ms=result["processing_time_ms"],
            )
            for result in results
        ]
    except KeyError:
        from config.settings import get_model_registry

        registry = get_model_registry()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": f"Model '{model_key}' not found",
                "available_models": registry.list_models("sentiment_analysis"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/sentiment/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to analyze sentiment batch: {str(e)}",
            },
        )
