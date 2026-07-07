"""
Customer Intelligence API Router

Combines core NLP tasks into a portfolio-grade workflow for customer feedback,
reviews, support notes, and market text.
"""

import time
from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.middleware.auth import User, get_current_user
from api.schemas.requests import CustomerIntelligenceRequest
from api.schemas.responses import (
    CustomerInsightItem,
    CustomerIntelligenceResponse,
    CustomerIntelligenceSummary,
    EntityResponse,
    SentimentResult,
    TopEntity,
)
from utils.inference import (
    DEFAULT_NER_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SUMMARIZATION_MODEL,
    get_inference_service,
)
from utils.metrics import record_error

router = APIRouter(prefix="/api/v1/customer-intelligence", tags=["Customer Intelligence"])


@router.post(
    "/analyze",
    response_model=CustomerIntelligenceResponse,
    summary="Analyze customer intelligence signals",
    description=(
        "Analyze customer feedback or market text with sentiment, entity extraction, "
        "aggregate trends, and an optional summary."
    ),
    responses={
        200: {"description": "Successful customer intelligence analysis"},
        400: {"description": "Invalid model selection"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def analyze_customer_intelligence(
    request: Request,
    body: CustomerIntelligenceRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> CustomerIntelligenceResponse:
    """Analyze a batch of customer text and return item-level plus aggregate signals."""
    from config.settings import get_model_registry

    registry = get_model_registry()
    sentiment_model = body.sentiment_model or DEFAULT_SENTIMENT_MODEL
    ner_model = body.ner_model or DEFAULT_NER_MODEL
    summary_model = body.summary_model or DEFAULT_SUMMARIZATION_MODEL

    try:
        sentiment_config = registry.get_model("sentiment_analysis", sentiment_model)
        ner_config = registry.get_model("ner", ner_model)
        summary_config = registry.get_model("summarization", summary_model)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_model",
                "message": str(e),
                "available_models": {
                    "sentiment_analysis": registry.list_models("sentiment_analysis"),
                    "ner": registry.list_models("ner"),
                    "summarization": registry.list_models("summarization"),
                },
            },
        ) from e

    service = get_inference_service()
    start_time = time.time()
    response_items = []
    sentiment_counts: Counter[str] = Counter()
    entity_counts: Counter[tuple[str, str]] = Counter()

    try:
        for item in body.items:
            item_start = time.time()
            sentiment = await service.analyze_sentiment(item.text, sentiment_model)
            entities = await service.extract_entities(item.text, ner_model)

            sentiment_label = sentiment["sentiment"]["label"]
            sentiment_counts[sentiment_label] += 1

            entity_responses = []
            for entity in entities["entities"]:
                entity_counts[(entity["text"], entity["label"])] += 1
                entity_responses.append(
                    EntityResponse(
                        text=entity["text"],
                        label=entity["label"],
                        start=entity["start"],
                        end=entity["end"],
                        score=entity.get("score"),
                    )
                )

            response_items.append(
                CustomerInsightItem(
                    id=item.id,
                    text=item.text,
                    metadata=item.metadata,
                    sentiment=SentimentResult(**sentiment["sentiment"]),
                    entities=entity_responses,
                    processing_time_ms=(time.time() - item_start) * 1000,
                )
            )

        summary_text = None
        if body.include_summary:
            joined_text = "\n".join(item.text for item in body.items)
            if len(joined_text) >= 50:
                summary = await service.summarize(
                    joined_text[:50000],
                    model_key=summary_model,
                    min_length=30,
                    max_length=160,
                )
                summary_text = summary["summary"]

        top_entities = [
            TopEntity(text=text, label=label, count=count) for (text, label), count in entity_counts.most_common(10)
        ]

        return CustomerIntelligenceResponse(
            items=response_items,
            aggregate=CustomerIntelligenceSummary(
                item_count=len(body.items),
                sentiment_distribution=dict(sentiment_counts),
                top_entities=top_entities,
                summary=summary_text,
                model_ids={
                    "sentiment": sentiment_config.model_id,
                    "ner": ner_config.model_id,
                    "summary": summary_config.model_id if body.include_summary else "",
                },
            ),
            processing_time_ms=(time.time() - start_time) * 1000,
        )
    except HTTPException:
        raise
    except Exception as e:
        record_error("inference_error", "/api/v1/customer-intelligence/analyze")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_error",
                "message": f"Failed to analyze customer intelligence: {str(e)}",
            },
        )
