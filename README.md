# NLP with Transformers - Examples

A collection of practical Natural Language Processing (NLP) examples using the Hugging Face Transformers library. This repository demonstrates various NLP tasks including sentiment analysis, named entity recognition, text summarization, and sentence embeddings using state-of-the-art transformer models.

Inspired by [Natural Language Processing with Transformers: Building Language Applications with Hugging Face](https://www.amazon.co.uk/gp/product/1098103246/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1).

## Purpose

This repository serves as a learning resource and reference implementation for common NLP tasks. Each script is self-contained and demonstrates:
- How to load and use pre-trained transformer models
- Different approaches to NLP tasks (pipeline API vs direct model usage)
- Working with real-world data (reviews, structured text)
- Evaluating model outputs with various metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MysterionRise/nlp-with-transformers-examples.git
cd nlp-with-transformers-examples

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (includes code formatting and linting tools)
pip install -r requirements-dev.txt
```

### Additional Dependencies

For NER examples using spaCy:
```bash
python -m spacy download en_core_web_trf
```

## Examples

### Sentence Embeddings
Generate semantic embeddings for sentences using sentence-transformers:
```bash
python embeddings_test.py
```

### Named Entity Recognition (NER)
Extract named entities from text using spaCy's transformer model:
```bash
python ner.py
```

### Sentiment Analysis
Analyze sentiment of customer reviews:
```bash
python sentiment_analysis.py
```
*Requires `data/all_reviews.csv`*

### Text Summarization
Compare multiple summarization models (BART, FLAN-T5, Pegasus):
```bash
python summarisation_llm_test.py
```
*Processes JSON files from `data/` directory*

### Local LLM Inference
Run inference with Mistral-7B locally:
```bash
python local_llm.py
```

### Text Generation
Generate text using Databricks Dolly model:
```bash
python summarisation_dolly.py
```

### Evaluation Metrics
Evaluate text similarity and quality using BLEU, ROUGE, METEOR, and BERTScore:
```bash
cd eval
python eval.py
```

## Repository Structure

```
├── embeddings_test.py          # Sentence embeddings with mean pooling
├── ner.py                       # Named entity recognition
├── sentiment_analysis.py        # Sentiment analysis on reviews
├── summarisation_llm_test.py    # Batch summarization comparison
├── summarisation_dolly.py       # Text generation with Dolly
├── local_llm.py                 # Local LLM inference
├── scrap_reviews.py            # Google Play review scraper
├── eval/
│   └── eval.py                 # Text evaluation metrics
├── huggingface_course/         # HuggingFace course materials
└── data/                       # Data files (JSON, CSV)
```

## Development

This project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Code formatting standards:
- **Black** formatter (line length: 120)
- **isort** for import sorting
- **flake8** for linting
- **Commitizen** for conventional commits
- **Bandit** for security checks

## Models Used

- **Sentiment Analysis**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Summarization**: `facebook/bart-large-cnn`, `google/flan-t5-xxl`, `google/pegasus-xsum`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **NER**: spaCy `en_core_web_trf`
- **Text Generation**: `databricks/dolly-v2-12b`, `mistralai/Mistral-7B-v0.3`

## Contributing

Feel free to open issues or submit pull requests with improvements or additional examples.

## License

See [LICENSE](LICENSE) file for details.
