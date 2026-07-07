# Operations Guide

## Local API

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python launch_ui.py api
```

Local docs are available at `http://localhost:8000/docs`.

## Docker Compose

Canonical portfolio run:

```bash
docker compose up api redis
```

The Compose API service uses `.env.example` for a local demo. For any shared environment:

```bash
cp .env.example .env
# Replace NLP_API__JWT_SECRET and NLP_API__API_KEYS
```

Then point the Compose service at `.env` or export equivalent environment variables in the deployment platform.

## Health And Readiness

- `/health`: liveness probe.
- `/ready`: validates config, model registry, model cache, auth configuration, and Redis when configured.
- `/api/v1/status`: request counts, average latency, cache stats, and uptime.
- `/metrics`: Prometheus exposition.

## Security Defaults

- No API keys are hardcoded in Python source.
- Demo credentials exist only in `.env.example`.
- API endpoints require API key or JWT.
- Public deployments must configure HTTPS and replace demo credentials.

## Cloud Deployment Path

Recommended simple path:

1. Build the Docker image in CI.
2. Push to a registry.
3. Deploy one API container plus managed Redis.
4. Configure `NLP_REDIS_URL`, `NLP_API__JWT_SECRET`, and `NLP_API__API_KEYS` as platform secrets.
5. Expose port `8000` behind a managed HTTPS load balancer.
6. Scrape `/metrics` with the platform monitoring stack.

## Runbooks

- High latency: check `/api/v1/status`, cache size, cold-start behavior, and `scripts/benchmark_api.py`.
- 401s: verify API key JSON and JWT secret configuration.
- 429s: inspect rate-limit headers and Redis availability.
- NER failures: verify `en_core_web_sm` is installed in the image/runtime.
- Model load failures: check network access, Hugging Face cache volume, memory limits, and model registry keys.
