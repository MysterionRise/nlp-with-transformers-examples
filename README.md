# Customer Intelligence NLP Platform

[![CI](https://github.com/MysterionRise/transformers-nlp-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/transformers-nlp-suite/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

API-first NLP platform for turning customer reviews, support notes, and market text into structured intelligence signals. The main proof point is the FastAPI service: authenticated endpoints, shared model caching, Redis-backed rate limiting, Prometheus metrics, structured logs, Docker Compose runtime, and CI-covered tests.

Gradio UIs remain available as optional demos, but the product story is the API.

## What It Does

- Analyzes sentiment for individual or batched customer text.
- Extracts named entities with spaCy or transformer-backed NER models.
- Computes semantic similarity with sentence-transformers embeddings.
- Generates summaries and extractive QA answers from supplied text.
- Combines these into `/api/v1/customer-intelligence/analyze` for portfolio-grade customer feedback analysis.
- Exposes runtime state through `/health`, `/ready`, `/metrics`, and `/api/v1/status`.

## Current Status

Implemented and verified:

- Docker Compose API stack: `docker compose up api redis`
- API key and JWT authentication
- Nested environment loading for `NLP_API__...` settings
- Shared inference layer over transformers, sentence-transformers, and spaCy
- LRU model cache with hit, miss, eviction, and size metrics
- Redis-backed rate limiting when `NLP_REDIS_URL` is configured, with memory fallback for local/test use
- Fast mocked API tests and slow real-model integration tests in CI
- Architecture, operations, evaluation, model-card, and ADR documentation

Important limitations:

- The bundled demo key is for local portfolio review only.
- First real-model requests can be slow because models are downloaded and loaded lazily.
- The tracked eval set is a smoke check, not a production quality benchmark.
- This is not intended for high-stakes or regulated automated decisions.

## Quick Start

### Docker Compose

The canonical local proof is the API plus Redis:

```bash
docker compose up api redis
```

Then open:

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Readiness: http://localhost:8000/ready
- Metrics: http://localhost:8000/metrics

The Compose service uses `.env.example`, which contains the local demo API key used in the examples below. For any shared deployment, copy `.env.example` to `.env` or configure equivalent platform secrets, then replace `NLP_API__JWT_SECRET` and `NLP_API__API_KEYS`.

### Local Python

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export NLP_API__JWT_SECRET=replace-with-a-long-random-secret
export NLP_API__API_KEYS='{"demo-customer-intel-key":{"name":"Portfolio Demo","role":"admin","rate_limit":1000,"enabled":true}}'

python launch_ui.py api
```

By default, local API docs are available at http://localhost:8000/docs.

## API Examples

### Customer Intelligence Workflow

```bash
curl -X POST http://localhost:8000/api/v1/customer-intelligence/analyze \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "review-001",
        "text": "The product is excellent, but delivery took too long.",
        "metadata": {"source": "reviews"}
      },
      {
        "id": "ticket-002",
        "text": "Support resolved my issue quickly and the mobile app is easy to use.",
        "metadata": {"source": "support"}
      }
    ],
    "include_summary": true
  }'
```

Returns per-item sentiment and entities, plus aggregate sentiment distribution, top entities, summary, model IDs, and processing time.

### Sentiment

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded my expectations.",
    "model": "twitter_roberta_multilingual"
  }'
```

### Summarization

```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence has transformed customer support operations by helping teams triage feedback, summarize conversations, and identify recurring product issues. The strongest systems still require human review, observability, and domain-specific evaluation before production rollout.",
    "min_length": 20,
    "max_length": 60
  }'
```

### Named Entity Recognition

```bash
curl -X POST http://localhost:8000/api/v1/ner \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple CEO Tim Cook announced new products in Cupertino.",
    "entity_types": ["PERSON", "ORG", "GPE"]
  }'
```

### Semantic Similarity

```bash
curl -X POST http://localhost:8000/api/v1/similarity \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The onboarding flow was easy to complete.",
    "text2": "The setup process was simple and clear.",
    "model": "all_minilm_l6"
  }'
```

### Question Answering

```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -H "X-API-Key: demo-customer-intel-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What needs human review?",
    "context": "Generated summaries can help analysts review customer feedback, but important business decisions still need human review."
  }'
```

### JWT Flow

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "X-API-Key: demo-customer-intel-key" | jq -r ".access_token")

curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "The new workflow is much faster."}'
```

## Endpoint Reference

| Endpoint | Method | Auth | Purpose |
| --- | --- | --- | --- |
| `/api/v1/customer-intelligence/analyze` | POST | Required | Combined sentiment, NER, aggregates, and optional summary |
| `/api/v1/sentiment` | POST | Required | Single-text sentiment |
| `/api/v1/sentiment/batch` | POST | Required | Batch sentiment |
| `/api/v1/summarize` | POST | Required | Text summarization |
| `/api/v1/ner` | POST | Required | Named entity extraction |
| `/api/v1/similarity` | POST | Required | Semantic similarity |
| `/api/v1/qa` | POST | Required | Extractive question answering |
| `/api/v1/auth/token` | POST | API key | Exchange API key for JWT |
| `/api/v1/models` | GET | No | List model registry categories |
| `/api/v1/models/{category}` | GET | No | List models in one category |
| `/api/v1/status` | GET | Optional | Request, latency, cache, and uptime summary |
| `/health` | GET | No | Liveness probe |
| `/ready` | GET | No | Config, registry, auth, cache, and Redis readiness |
| `/metrics` | GET | No | Prometheus exposition |

## Configuration

Key environment variables:

| Variable | Purpose |
| --- | --- |
| `NLP_API__JWT_SECRET` | JWT signing secret |
| `NLP_API__API_KEYS` | JSON map of API keys to name, role, rate limit, and enabled flag |
| `NLP_REDIS_URL` | Redis backend for shared rate limiting |
| `NLP_MAX_CACHED_MODELS` | LRU model-cache size |
| `NLP_LOG_LEVEL` | Logging level |
| `NLP_JSON_LOGS` | Emit JSON logs when `true` |
| `NLP_OTEL_ENABLED` | Enable optional OpenTelemetry FastAPI instrumentation |

Model defaults live in `utils/inference.py` and model definitions live in `config/models.yaml`.

Default API models:

- Sentiment: `twitter_roberta_multilingual`
- NER: `spacy_sm`
- Similarity: `all_minilm_l6`
- Summarization: `bart_large_cnn`
- QA: `distilbert_squad`

## Architecture

The API routers do not load models directly. They call `utils/inference.py`, which performs model lookup, output normalization, timing, and access to the shared cache in `utils/model_cache.py`.

The cache supports:

- Hugging Face transformers pipelines
- sentence-transformers embedding models
- spaCy language pipelines

Operational instrumentation includes:

- Request logging with request ID, auth method, user role, status code, latency, and error type
- Inference logs with task, model key, model ID, status, and latency
- Prometheus counters and histograms for HTTP requests, inference, cache, auth attempts, errors, and rate-limit hits
- `/api/v1/status` summaries from real in-process request and cache state

For more detail, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing And Evaluation

CI runs:

- Black, isort, and flake8
- Fast tests on Python 3.11, 3.12, and 3.13
- Slow real-model integration tests on Python 3.11
- Full test suite with coverage on Python 3.11

Useful local commands:

```bash
pytest -m "not slow"
pytest -m slow
python scripts/run_eval.py --api-url http://localhost:8000 --api-key <key>
python scripts/benchmark_api.py --api-url http://localhost:8000 --api-key <key>
```

The tracked eval data is in `data/customer_intelligence_eval.json`. It is intentionally small and should be treated as a smoke/regression set, not as evidence of production model quality. See [EVAL_REPORT.md](EVAL_REPORT.md).

## Optional Gradio UIs

The UI layer is secondary but still available:

```bash
python launch_ui.py
python launch_ui.py sentiment
python launch_ui.py similarity
python launch_ui.py ner
python launch_ui.py summarization
python launch_ui.py performance
python launch_ui.py qa
python launch_ui.py generation
python launch_ui.py zero_shot
python launch_ui.py translation
python launch_ui.py vision
```

Docker Compose UI profiles:

```bash
docker compose --profile ui up ui-launcher
docker compose --profile individual up sentiment qa generation
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design and tradeoffs
- [OPERATIONS.md](OPERATIONS.md): local, Docker, cloud, monitoring, and runbooks
- [EVAL_REPORT.md](EVAL_REPORT.md): current eval set, CI coverage, limitations, next steps
- [MODEL_CARD.md](MODEL_CARD.md): intended use, out-of-scope use, default models, risks
- [ADR 0001](docs/adr/0001-shared-inference-cache.md): shared inference cache
- [ADR 0002](docs/adr/0002-redis-rate-limiting.md): Redis rate limiting
- [ADR 0003](docs/adr/0003-api-based-evaluation.md): API-based evaluation

## Portfolio Read

This repo is meant to show AI engineering judgment, not only model demos:

- coherent product framing
- API-first runtime
- real auth/rate-limit/metrics/logging paths
- model-cache and inference abstractions
- tests that separate fast mocked behavior from slow model downloads
- explicit limitations and risk boundaries

## License

MIT License. See `LICENSE`.
