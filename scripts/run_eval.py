#!/usr/bin/env python3
"""
Run a small customer-intelligence evaluation against a running API.

This script is intentionally API-based so the same check works for local Docker,
cloud deployments, and portfolio demos.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx

DEFAULT_EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "customer_intelligence_eval.json"


def load_eval_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(api_url: str, api_key: str, eval_file: Path) -> Dict[str, Any]:
    eval_items = load_eval_items(eval_file)
    request_items = [{"id": item["id"], "text": item["text"]} for item in eval_items]

    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/api/v1/customer-intelligence/analyze",
            headers={"X-API-Key": api_key},
            json={"items": request_items, "include_summary": True},
        )
        response.raise_for_status()
        result = response.json()

    by_id = {item["id"]: item for item in result["items"]}
    checks = []

    for expected in eval_items:
        actual = by_id[expected["id"]]
        sentiment_label = actual["sentiment"]["label"]
        expected_sentiments = expected.get("expected_sentiment_any", [])
        sentiment_passed = not expected_sentiments or sentiment_label in expected_sentiments

        actual_entities = {entity["text"] for entity in actual.get("entities", [])}
        expected_entities = set(expected.get("expected_entities", []))
        entity_passed = expected_entities.issubset(actual_entities)

        checks.append(
            {
                "id": expected["id"],
                "sentiment_label": sentiment_label,
                "sentiment_passed": sentiment_passed,
                "expected_entities": sorted(expected_entities),
                "actual_entities": sorted(actual_entities),
                "entity_passed": entity_passed,
                "passed": sentiment_passed and entity_passed,
            }
        )

    passed = sum(1 for check in checks if check["passed"])
    return {
        "api_url": api_url,
        "eval_file": str(eval_file),
        "passed": passed,
        "total": len(checks),
        "pass_rate": round(passed / len(checks), 4) if checks else 0,
        "workflow_processing_time_ms": result["processing_time_ms"],
        "aggregate": result["aggregate"],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run customer-intelligence API evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--api-key", default=os.getenv("NLP_API_KEY"), help="API key, or set NLP_API_KEY")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Evaluation JSON file")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key is required unless NLP_API_KEY is set")

    result = run_eval(args.api_url, args.api_key, args.eval_file)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
