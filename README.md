# NLP with Transformers - Interactive Examples

[![CI](https://github.com/MysterionRise/nlp-with-transformers-examples/workflows/CI/badge.svg)](https://github.com/MysterionRise/nlp-with-transformers-examples/actions)
[![Code Quality](https://github.com/MysterionRise/nlp-with-transformers-examples/workflows/Code%20Quality/badge.svg)](https://github.com/MysterionRise/nlp-with-transformers-examples/actions)
[![Pre-commit](https://github.com/MysterionRise/nlp-with-transformers-examples/workflows/Pre-commit/badge.svg)](https://github.com/MysterionRise/nlp-with-transformers-examples/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of practical NLP examples inspired by [Natural Language Processing with Transformers: Building Language Applications with Hugging Face](https://www.amazon.co.uk/gp/product/1098103246/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1) with **interactive web UIs** for hands-on experimentation.

## 🎯 Features

### Interactive UIs (New!)
- 🎭 **Sentiment Analysis Playground** - Analyze sentiment with multiple models
- 🔍 **Sentence Similarity Explorer** - Compare embeddings and semantic similarity
- 🏷️ **NER Visualizer** - Extract and visualize named entities
- 📝 **Text Summarization Studio** - Generate and compare text summaries

### CLI Examples
- Sentiment analysis on reviews
- Named entity recognition with Spacy
- Text summarization with multiple models
- Sentence embeddings and similarity
- Data scraping from Google Play
- Comprehensive evaluation metrics (BLEU, ROUGE, METEOR, BERTScore)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MysterionRise/nlp-with-transformers-examples.git
cd nlp-with-transformers-examples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For NER functionality, download Spacy model:
```bash
python -m spacy download en_core_web_sm
# Or for better accuracy (larger model):
python -m spacy download en_core_web_trf
```

### Launch Interactive UIs

**Option 1: Interactive Menu**
```bash
python launch_ui.py
```

**Option 2: Launch Specific UI**
```bash
python launch_ui.py sentiment       # Sentiment Analysis
python launch_ui.py similarity      # Sentence Similarity
python launch_ui.py ner             # Named Entity Recognition
python launch_ui.py summarization   # Text Summarization
```

**Option 3: Direct Launch**
```bash
python ui/sentiment_playground.py      # Port 7860
python ui/similarity_explorer.py       # Port 7861
python ui/ner_visualizer.py           # Port 7862
python ui/summarization_studio.py     # Port 7863
```

## 📚 Interactive UIs Overview

### 🎭 Sentiment Analysis Playground
**Port:** 7860 | **File:** `ui/sentiment_playground.py`

Analyze text sentiment using state-of-the-art transformer models.

**Features:**
- Multiple model comparison (RoBERTa, DistilBERT, BERT)
- Real-time sentiment classification
- Confidence scores with visual bars
- Batch processing support
- Example texts included

**Models:**
- Twitter RoBERTa (Multilingual)
- Twitter RoBERTa (English)
- DistilBERT SST-2
- BERT Base SST-2

---

### 🔍 Sentence Similarity Explorer
**Port:** 7861 | **File:** `ui/similarity_explorer.py`

Explore semantic similarity between sentences using embeddings.

**Features:**
- Pairwise sentence comparison
- Semantic search (find most similar sentences)
- 2D embedding visualization (t-SNE, PCA)
- Similarity heatmaps
- Interactive plots

**Use Cases:**
- Duplicate detection
- Semantic search
- Content clustering
- Paraphrase detection

---

### 🏷️ NER Visualizer
**Port:** 7862 | **File:** `ui/ner_visualizer.py`

Extract and visualize named entities from text using Spacy.

**Features:**
- Interactive entity highlighting
- Entity type filtering (18+ entity types)
- Entity statistics and charts
- JSON export
- Pre-loaded examples

**Entity Types:**
- PERSON, ORG, GPE, LOC, DATE, TIME
- MONEY, PERCENT, PRODUCT, EVENT
- And more...

---

### 📝 Text Summarization Studio
**Port:** 7863 | **File:** `ui/summarization_studio.py`

Generate concise summaries with multiple transformer models.

**Features:**
- Single model summarization
- Multi-model comparison
- Adjustable parameters (min/max length)
- Summary statistics (compression ratio)
- Example articles

**Models:**
- BART Large CNN
- T5 Large/Base
- Pegasus XSum
- DistilBART CNN

## 🛠️ CLI Examples

### Sentiment Analysis
```bash
python sentiment-analysis.py
```
Analyzes sentiment of reviews from `data/all_reviews.csv`.

### Named Entity Recognition
```bash
python ner.py
```
Extracts entities from text and generates visualizations.

### Text Summarization
```bash
python summarisation_llm_test.py
```
Compares 5 summarization models on JSON files in `data/` directory.

### Sentence Embeddings
```bash
python embeddings_test.py
```
Generates sentence embeddings using sentence-transformers.

### Data Scraping
```bash
python scrap_reviews.py
```
Scrapes app reviews from Google Play Store.

### Evaluation
```bash
python eval/eval.py
```
Comprehensive NLP evaluation with BLEU, ROUGE, METEOR, BERTScore.

## 📦 Requirements

### Core Dependencies
- transformers[torch] - Hugging Face Transformers
- torch - PyTorch
- spacy - Industrial NLP
- numpy - Numerical computing

### UI Dependencies
- gradio>=4.0.0 - Interactive UIs
- plotly>=5.17.0 - Visualizations
- scikit-learn - ML utilities
- umap-learn - Dimensionality reduction

### Evaluation
- nltk - BLEU scores
- rouge-score - ROUGE metrics
- bert-score - BERT-based evaluation

See `requirements.txt` for full list.

## 🧪 Testing & Quality

This project maintains high code quality with comprehensive testing and CI/CD.

### Running Tests

```bash
# All tests with coverage
make test

# Unit tests only
make test-unit

# Smoke tests only
make test-smoke

# Fast tests (parallel execution)
make test-fast
```

### Code Quality Checks

```bash
# Format code
make format

# Lint code
make lint

# Run all quality checks
make quality

# Security scan
make security
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run checks before each commit:
```bash
pre-commit install
```

### Test Coverage

- Unit tests for UI components
- Integration tests for workflows
- Smoke tests for all UIs
- Code coverage reporting
- Multi-platform testing (Ubuntu, macOS, Windows)
- Multi-version testing (Python 3.9, 3.10, 3.11)

See [tests/README.md](tests/README.md) for detailed testing documentation.

## 🎓 Use Cases

- **Education**: Learn NLP concepts interactively
- **Prototyping**: Quickly test transformer models
- **Research**: Compare model performance
- **Development**: Build NLP applications
- **Demo**: Showcase NLP capabilities

## 📖 Additional Resources

- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed feature roadmap
- [Testing Guide](tests/README.md) - Comprehensive testing documentation
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Spacy Documentation](https://spacy.io/usage)
- [Gradio Documentation](https://gradio.app/docs/)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process
- Commit message conventions

Quick start:
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/nlp-with-transformers-examples.git

# Install development dependencies
make install-dev

# Make your changes and run tests
make test
make lint

# Submit a pull request
```

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by "Natural Language Processing with Transformers" book
- Built with Hugging Face Transformers
- UI powered by Gradio
- NER powered by Spacy
