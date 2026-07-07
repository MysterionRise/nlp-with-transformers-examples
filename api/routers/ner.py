"""
Named Entity Recognition (NER) API Router

Provides endpoints for extracting named entities from text.
"""

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import NERRequest
from api.schemas.responses import EntityResponse, NERResponse
from utils.inference import DEFAULT_NER_MODEL, get_inference_service
from utils.metrics import record_error

router = APIRouter(prefix="/api/v1", tags=["Named Entity Recognition"])


async def run_spacy_ner(text: str, entity_types: Optional[List[str]] = None) -> tuple[list, str, float]:
    """
    Run NER using spaCy

    Args:
        text: Text to process
        entity_types: Optional filter for entity types

    Returns:
        Tuple of (entities, model_name, processing_time_ms)
    """
    result = await get_inference_service().extract_entities(text, DEFAULT_NER_MODEL, entity_types)
    return result["entities"], result["model"], result["processing_time_ms"]


async def run_transformer_ner(
    text: str, model_key: str, entity_types: Optional[List[str]] = None
) -> tuple[list, str, float]:
    """
    Run NER using transformer models

    Args:
        text: Text to process
        model_key: Model key from registry
        entity_types: Optional filter for entity types

    Returns:
        Tuple of (entities, model_id, processing_time_ms)
    """
    result = await get_inference_service().extract_entities(text, model_key, entity_types)
    return result["entities"], result["model"], result["processing_time_ms"]


@router.post(
    "/ner",
    response_model=NERResponse,
    summary="Extract named entities",
    description="Extract named entities (people, organizations, locations, dates, etc.) from text. "
    "Supports both spaCy and transformer-based models.",
    responses={
        200: {
            "description": "Successful entity extraction",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Tim Cook is the CEO of Apple Inc.",
                        "entities": [
                            {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8, "score": 0.99},
                            {"text": "Apple Inc.", "label": "ORG", "start": 23, "end": 33, "score": 0.98},
                        ],
                        "entity_counts": {"PERSON": 1, "ORG": 1},
                        "model": "en_core_web_sm",
                        "processing_time_ms": 23.5,
                    }
                }
            },
        },
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def extract_entities(
    request: Request,
    body: NERRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> NERResponse:
    """
    Extract named entities from text

    Supports multiple NER backends:
    - **spacy_sm**: Fast spaCy model (default)
    - **spacy_trf**: Transformer-based spaCy model (more accurate)
    - **bert_ner**: BERT-based NER model

    You can filter results to specific entity types using the entity_types parameter.

    **Common entity types:**
    - PERSON: People names
    - ORG: Organizations
    - GPE: Geopolitical entities (countries, cities)
    - LOC: Locations
    - DATE: Dates
    - TIME: Times
    - MONEY: Monetary values
    - PERCENT: Percentages

    **Authentication required:** API key or JWT token
    """
    model_key = body.model or DEFAULT_NER_MODEL

    try:
        result = await get_inference_service().extract_entities(body.text, model_key, body.entity_types)

        # Convert to response format
        entity_responses = [
            EntityResponse(
                text=e["text"],
                label=e["label"],
                start=e["start"],
                end=e["end"],
                score=e.get("score"),
            )
            for e in result["entities"]
        ]

        return NERResponse(
            text=body.text,
            entities=entity_responses,
            entity_counts=result["entity_counts"],
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
                "message": f"Model '{model_key}' not found in ner category",
                "available_models": registry.list_models("ner"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/ner")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to extract entities: {str(e)}",
            },
        )
