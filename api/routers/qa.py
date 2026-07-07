"""
Question Answering API Router

Provides endpoints for extractive question answering.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import QARequest
from api.schemas.responses import QAAnswer, QAResponse
from utils.inference import DEFAULT_QA_MODEL, get_inference_service
from utils.metrics import record_error

router = APIRouter(prefix="/api/v1", tags=["Question Answering"])


@router.post(
    "/qa",
    response_model=QAResponse,
    summary="Answer questions from context",
    description="Extract answers to questions from a given context passage. "
    "Uses extractive QA models trained on datasets like SQuAD.",
    responses={
        200: {
            "description": "Successful question answering",
            "content": {
                "application/json": {
                    "example": {
                        "question": "What is the capital of France?",
                        "context": "France is a country in Western Europe. Its capital is Paris.",
                        "answers": [{"answer": "Paris", "score": 0.9234, "start": 52, "end": 57}],
                        "model": "distilbert-base-cased-distilled-squad",
                        "processing_time_ms": 52.3,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def answer_question(
    request: Request,
    body: QARequest,
    user: Annotated[User, Depends(get_current_user)],
) -> QAResponse:
    """
    Answer a question based on the provided context

    Uses extractive question answering to find the answer span within the context.
    The model will identify the most relevant portion of the context that answers
    the question.

    **Tips for best results:**
    - Provide clear, specific questions
    - Ensure the context contains the answer
    - Use factual questions (who, what, when, where)

    **Authentication required:** API key or JWT token
    """
    model_key = body.model or DEFAULT_QA_MODEL

    try:
        result = await get_inference_service().answer_question(
            question=body.question,
            context=body.context,
            model_key=model_key,
            top_k=body.top_k or 1,
        )

        return QAResponse(
            question=body.question,
            context=body.context,
            answers=[QAAnswer(**answer) for answer in result["answers"]],
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
                "message": f"Model '{model_key}' not found in question_answering category",
                "available_models": registry.list_models("question_answering"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/qa")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to answer question: {str(e)}",
            },
        )
