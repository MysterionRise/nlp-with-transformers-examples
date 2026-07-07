# Architecture

## Product Positioning

Customer Intelligence NLP Platform is an API-first service for turning reviews, support notes, and market text into structured signals: sentiment, entities, similarity, question answering, and summaries. Gradio UIs remain optional demos; the FastAPI service is the primary production proof point.

## Runtime Shape

- FastAPI exposes task endpoints and the portfolio workflow endpoint `/api/v1/customer-intelligence/analyze`.
- Pydantic settings load from `.env`, including nested `NLP_API__...` values for auth.
- A shared inference service owns model lookup, output normalization, and model-cache access.
- The central model cache supports transformers pipelines, sentence-transformers models, and spaCy pipelines.
- Redis is used for shared rate limiting when `NLP_REDIS_URL` is configured; memory fallback is used for tests and local development.
- Prometheus metrics, structured logs, request IDs, and `/api/v1/status` expose runtime behavior.

## Key Tradeoffs

- The API path is optimized first; UIs are intentionally secondary to avoid duplicate architecture decisions.
- Models are loaded lazily to keep startup fast and support small local machines.
- Redis rate limiting uses a fixed-window strategy for operational simplicity.
- Real-model integration tests are separated from mocked fast tests to avoid downloads in normal CI.

## Operational Boundary

This repo demonstrates production-minded engineering, not a regulated production deployment. Public deployments must replace demo secrets, configure HTTPS, add persistent telemetry export, and review model/data risks for the target domain.
