.PHONY: help install install-dev test test-unit test-smoke lint format clean check quality

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_sm
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-unit:  ## Run unit tests only
	pytest tests/unit/ -v

test-smoke:  ## Run smoke tests only
	pytest tests/ui/test_smoke.py -v -m smoke

test-fast:  ## Run tests without coverage
	pytest tests/ -v -n auto

lint:  ## Run linting checks
	black --check --line-length 120 .
	isort --check-only --profile black --line-length 120 .
	flake8 . --max-line-length=120 --exclude=.git,__pycache__,venv,env,.venv

format:  ## Format code with black and isort
	black --line-length 120 .
	isort --profile black --line-length 120 .

clean:  ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check:  ## Run pre-commit checks on all files
	pre-commit run --all-files

quality:  ## Run code quality checks
	@echo "=== Running code quality checks ==="
	@echo ""
	@echo "1. Code formatting..."
	black --check --line-length 120 . || true
	@echo ""
	@echo "2. Import sorting..."
	isort --check-only --profile black --line-length 120 . || true
	@echo ""
	@echo "3. Linting..."
	flake8 . --max-line-length=120 --exclude=.git,__pycache__,venv,env,.venv --statistics || true
	@echo ""
	@echo "4. Security check..."
	bandit -r . --exclude ./venv,./env,./.venv,./tests -ll || true
	@echo ""
	@echo "Quality check complete!"

security:  ## Run security checks
	bandit -r . --exclude ./venv,./env,./.venv,./tests
	safety check

coverage:  ## Generate coverage report
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

launch-sentiment:  ## Launch Sentiment Analysis UI
	python ui/sentiment_playground.py

launch-similarity:  ## Launch Similarity Explorer UI
	python ui/similarity_explorer.py

launch-ner:  ## Launch NER Visualizer UI
	python ui/ner_visualizer.py

launch-summarization:  ## Launch Summarization Studio UI
	python ui/summarization_studio.py

all-checks: lint test quality  ## Run all checks (lint + test + quality)
