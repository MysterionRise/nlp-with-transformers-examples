# NLP with Transformers - Interactive Examples

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

## 🎓 Use Cases

- **Education**: Learn NLP concepts interactively
- **Prototyping**: Quickly test transformer models
- **Research**: Compare model performance
- **Development**: Build NLP applications
- **Demo**: Showcase NLP capabilities

## 📖 Additional Resources

- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed feature roadmap
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Spacy Documentation](https://spacy.io/usage)
- [Gradio Documentation](https://gradio.app/docs/)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by "Natural Language Processing with Transformers" book
- Built with Hugging Face Transformers
- UI powered by Gradio
- NER powered by Spacy
