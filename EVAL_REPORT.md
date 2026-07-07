# Evaluation Report

## Purpose

The evaluation path is designed to prove that the public API works end to end and to document known model limitations. It is not a claim of benchmark-grade model quality.

## Current Eval Set

The tracked smoke set lives at `data/customer_intelligence_eval.json` and covers:

- Positive feedback with a negative operational concern.
- Support feedback with expected organization/location entities.
- Negative service feedback.

Run it against a local or cloud API:

```bash
export NLP_API_KEY=<demo key from .env.example>
python scripts/run_eval.py --api-url http://localhost:8000
```

## Metrics Captured

- Sentiment label sanity checks.
- Required entity presence for simple fixtures.
- Workflow processing time.
- Aggregate sentiment distribution and top entities.

## Known Limitations

- The current eval set is intentionally small and should be expanded before any production claim.
- Sentiment labels differ across model families; tests allow acceptable label sets where customer text is mixed.
- spaCy small models are fast but can miss domain-specific entities.
- Generated summaries require qualitative review and should not be treated as ground truth.

## Next Evaluation Improvements

- Add at least 50 labeled customer feedback examples across retail, support, finance, and product feedback.
- Track latency and quality by model key.
- Add regression thresholds for pass rate, p95 latency, and error rate.
- Add prompt-injection-style and sensitive-data logging checks for generative paths.
