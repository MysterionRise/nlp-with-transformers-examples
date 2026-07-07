# Model Card

## System

Customer Intelligence NLP Platform combines open-source NLP models for sentiment analysis, named entity recognition, semantic similarity, extractive question answering, and summarization.

## Intended Use

- Analyze customer reviews and support notes.
- Extract aggregate sentiment and frequently mentioned entities.
- Compare short text similarity for duplicate or related feedback detection.
- Produce draft summaries for human review.

## Out Of Scope

- Automated high-stakes decisions.
- Medical, legal, credit, employment, or insurance decisions.
- Fully autonomous customer messaging without human review.
- Regulated production use without domain-specific evaluation and risk review.

## Default Models

- Sentiment: `twitter_roberta_multilingual`
- NER: `spacy_sm`
- Similarity: `all_minilm_l6`
- Summarization: `bart_large_cnn`
- QA: `distilbert_squad`

## Limitations

- Model outputs may be wrong, biased, stale, or inconsistent across domains.
- Mixed sentiment text can produce a single coarse label that hides nuance.
- spaCy small NER prioritizes speed over recall.
- Summaries can omit important details or overstate weak evidence.
- Performance depends on local hardware, cache warmth, model size, and network/cache availability.

## Evaluation

See `EVAL_REPORT.md` and `data/customer_intelligence_eval.json` for the current smoke evaluation. The current eval set proves wiring and basic sanity only; it does not establish production model quality.
