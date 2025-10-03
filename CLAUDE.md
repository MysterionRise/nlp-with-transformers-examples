# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A collection of NLP examples using Hugging Face Transformers library, inspired by "Natural Language Processing with Transformers" book. The repository contains standalone Python scripts demonstrating various NLP tasks including sentiment analysis, named entity recognition, summarization, and embeddings.

## Environment Setup

Python virtual environment with transformers and PyTorch:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development tools
pip install -r requirements-dev.txt

# Alternative: Install from README
pip install "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2"
```

For spaCy NER examples:
```bash
python -m spacy download en_core_web_trf
```

## Code Quality Tools

The project uses pre-commit hooks with strict formatting standards:

- **Black** formatter with line length 120
- **isort** with black profile and line length 120
- **flake8** linter with max line length 120, ignoring W605 and E203
- **Commitizen** for conventional commits
- **Bandit** for security checks

Run pre-commit manually:
```bash
pre-commit run --all-files
```

## Repository Structure

### Root-level Scripts
Standalone examples demonstrating specific NLP tasks:

- `embeddings_test.py` - Sentence embeddings using sentence-transformers/all-MiniLM-L6-v2
- `local_llm.py` - Local LLM inference with Mistral-7B
- `ner.py` - Named entity recognition using spaCy transformer model
- `sentiment-analysis.py` - Sentiment analysis on CSV reviews using cardiffnlp/twitter-xlm-roberta-base-sentiment
- `summarisation_llm_test.py` - Batch summarization testing with multiple models (BART, FLAN-T5, Pegasus)
- `summarisation_dolly.py` - Text generation with Databricks Dolly-v2-12b
- `scrap_reviews.py` - Google Play Store review scraping utility

### Data Directory
Contains JSON files (1.json through 8.json) with structured data for summarization tasks, and CSV files with review data.

### Subdirectories
- `huggingface_course/` - Tutorial materials from HuggingFace course
- `eval/` - Evaluation scripts with metrics (BLEU, ROUGE, METEOR, BERTScore, cosine similarity)

## Running Examples

Each script is self-contained and can be run directly:

```bash
python embeddings_test.py
python ner.py
python local_llm.py
```

For sentiment analysis on reviews:
```bash
python sentiment-analysis.py  # Requires data/all_reviews.csv
```

For batch summarization:
```bash
python summarisation_llm_test.py  # Processes data/1.json through data/8.json
```

## Model Usage Patterns

1. **Pipeline API** (sentiment-analysis.py, summarisation_dolly.py):
   ```python
   from transformers import pipeline
   task = pipeline("task-name", model=model_path, tokenizer=model_path)
   ```

2. **Direct Model Loading** (embeddings_test.py, local_llm.py):
   ```python
   from transformers import AutoModel, AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModel.from_pretrained(model_id)
   ```

3. **Batch Processing** (summarisation_llm_test.py):
   Iterates through multiple models and files to generate comparative results.

## Data Flow

- Input data: CSV files (`data/all_reviews.csv`) or JSON files (`data/1-8.json`)
- Output: Printed results or JSON files (`results.json` for summarization comparisons)
- Evaluation: `eval/eval.py` provides comprehensive metrics for text similarity and quality

## Important Notes

- Scripts create pipeline objects inside loops (sentiment-analysis.py:25) - may be inefficient for production
- Some scripts use older model versions or configurations that may need updating
- The summarization script processes 8 JSON files across 5 different models sequentially
- Pre-commit hooks enforce conventional commits via commitizen
