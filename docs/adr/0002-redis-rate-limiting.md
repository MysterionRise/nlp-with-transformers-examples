# ADR 0002: Redis-Backed Rate Limiting With Memory Fallback

## Status

Accepted

## Context

In-memory rate limiting works for local development but fails across multiple API replicas. A portfolio deployment needs a shared backend while keeping tests simple.

## Decision

Use Redis when `NLP_REDIS_URL` is configured and reachable. Fall back to memory when Redis is not configured or unavailable.

## Consequences

- Docker/cloud deployments can share rate limits across replicas.
- Local tests do not require Redis.
- The Redis strategy uses fixed windows for simplicity; stricter production use may require sliding windows or token buckets.
