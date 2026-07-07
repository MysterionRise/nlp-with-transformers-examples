# ADR 0001: Shared Inference Cache

## Status

Accepted

## Context

The original API and UIs loaded models in multiple local dictionaries. That made cache behavior, metrics, defaults, and memory management inconsistent.

## Decision

Use one central model cache for transformers pipelines, sentence-transformers models, and spaCy pipelines. API routers call a shared inference service rather than loading models directly.

## Consequences

- API behavior is easier to test and instrument.
- Cache hit/miss/eviction metrics are centralized.
- UIs can migrate gradually without changing API contracts.
- The cache remains process-local; horizontal production deployments still need per-replica model memory planning.
