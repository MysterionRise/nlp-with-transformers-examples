# ADR 0003: API-Based Evaluation

## Status

Accepted

## Context

Portfolio reviewers need proof that the deployed API works, not only that internal Python functions can be called. Evaluation should exercise auth, routing, validation, inference, aggregation, and observability.

## Decision

Run evaluation and benchmark scripts against the public API surface.

## Consequences

- The same scripts work for local Docker and cloud deployments.
- Results include real runtime overhead and cache behavior.
- Unit tests still mock inference for speed; real-model checks remain slow/integration tests.
