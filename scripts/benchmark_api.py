#!/usr/bin/env python3
"""
Benchmark the public API path for portfolio and regression checks.
"""

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List

import httpx

SAMPLE_REQUEST = {
    "text1": "The weather is beautiful today.",
    "text2": "It is a lovely sunny day outside.",
    "model": "all_minilm_l6",
}


def timed_post(client: httpx.Client, url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    response = client.post(url, headers={"X-API-Key": api_key}, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return {"status_code": response.status_code, "elapsed_ms": elapsed_ms, "body": response.json()}


def benchmark(api_url: str, api_key: str, iterations: int) -> Dict[str, Any]:
    base_url = api_url.rstrip("/")
    endpoint = f"{base_url}/api/v1/similarity"

    with httpx.Client(timeout=180) as client:
        before_status = client.get(f"{base_url}/api/v1/status", headers={"X-API-Key": api_key}).json()
        results: List[Dict[str, Any]] = []
        for _ in range(iterations):
            results.append(timed_post(client, endpoint, api_key, SAMPLE_REQUEST))
        after_status = client.get(f"{base_url}/api/v1/status", headers={"X-API-Key": api_key}).json()

    latencies = [result["elapsed_ms"] for result in results]
    return {
        "api_url": api_url,
        "endpoint": "/api/v1/similarity",
        "iterations": iterations,
        "cold_start_ms": round(latencies[0], 2) if latencies else None,
        "warm_min_ms": round(min(latencies[1:] or latencies), 2) if latencies else None,
        "warm_median_ms": round(statistics.median(latencies[1:] or latencies), 2) if latencies else None,
        "warm_max_ms": round(max(latencies[1:] or latencies), 2) if latencies else None,
        "cache_before": before_status.get("model_cache_stats", {}),
        "cache_after": after_status.get("model_cache_stats", {}),
        "requests_after": after_status.get("requests_total"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark customer-intelligence API runtime")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--api-key", default=os.getenv("NLP_API_KEY"), help="API key, or set NLP_API_KEY")
    parser.add_argument("--iterations", type=int, default=3, help="Number of similarity requests")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key is required unless NLP_API_KEY is set")

    result = benchmark(args.api_url, args.api_key, args.iterations)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
