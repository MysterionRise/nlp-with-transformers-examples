"""
Sentiment Analysis Playground - Interactive UI for sentiment classification

Features:
- Real-time sentiment analysis on user input
- Multiple model comparison
- Confidence scores with visual bars
- Batch processing support
- Example texts for quick testing
"""

import gradio as gr
from transformers import pipeline
import plotly.graph_objects as go
from typing import Dict, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available sentiment analysis models
SENTIMENT_MODELS = {
    "Twitter RoBERTa (Multilingual)": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "Twitter RoBERTa (English)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "DistilBERT SST-2": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT Base (SST-2)": "textattack/bert-base-uncased-SST-2",
}

# Cache for loaded models
model_cache = {}


def load_model(model_name: str):
    """Load or retrieve cached sentiment analysis model"""
    if model_name not in model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            model_cache[model_name] = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODELS[model_name],
                tokenizer=SENTIMENT_MODELS[model_name],
                top_k=None  # Return all scores
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    return model_cache[model_name]


def analyze_sentiment(text: str, model_name: str) -> Tuple[Dict, str]:
    """
    Analyze sentiment of the input text

    Args:
        text: Input text to analyze
        model_name: Name of the model to use

    Returns:
        Tuple of (results dict, visualization HTML)
    """
    if not text or not text.strip():
        return {"error": "Please provide some text to analyze"}, ""

    try:
        # Load model
        classifier = load_model(model_name)

        # Get predictions
        results = classifier(text[:512])[0]  # Limit to 512 chars for speed

        # Format results
        formatted_results = {}
        labels = []
        scores = []

        for result in results:
            label = result['label']
            score = result['score']
            formatted_results[label] = f"{score:.4f} ({score*100:.2f}%)"
            labels.append(label)
            scores.append(score)

        # Get the top prediction
        top_result = max(results, key=lambda x: x['score'])
        formatted_results['Prediction'] = top_result['label']
        formatted_results['Confidence'] = f"{top_result['score']*100:.2f}%"

        # Create visualization
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=labels,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale='RdYlGn',
                    showscale=False
                ),
                text=[f"{s*100:.1f}%" for s in scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Sentiment Confidence Scores",
            xaxis_title="Confidence",
            yaxis_title="Sentiment",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 1])
        )

        return formatted_results, fig

    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"error": f"Error: {str(e)}"}, None


def analyze_batch(text: str, model_name: str, delimiter: str = "\n") -> str:
    """
    Analyze multiple texts in batch

    Args:
        text: Multiple texts separated by delimiter
        model_name: Name of the model to use
        delimiter: Delimiter to split texts (default: newline)

    Returns:
        Formatted results as string
    """
    if not text or not text.strip():
        return "Please provide some text to analyze"

    try:
        # Split texts
        texts = [t.strip() for t in text.split(delimiter) if t.strip()]

        if not texts:
            return "No valid text found"

        # Load model
        classifier = load_model(model_name)

        # Analyze each text
        results = []
        for i, txt in enumerate(texts, 1):
            pred = classifier(txt[:512])[0]
            top = max(pred, key=lambda x: x['score'])
            results.append(f"{i}. {txt[:50]}{'...' if len(txt) > 50 else ''}\n"
                         f"   → {top['label']} ({top['score']*100:.2f}%)\n")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        return f"Error: {str(e)}"


# Example texts for testing
EXAMPLES = [
    ["This movie was absolutely fantastic! I loved every minute of it.", "Twitter RoBERTa (Multilingual)"],
    ["I'm so disappointed with this product. Complete waste of money.", "Twitter RoBERTa (Multilingual)"],
    ["The service was okay, nothing special but not terrible either.", "Twitter RoBERTa (Multilingual)"],
    ["Best experience ever! Highly recommend to everyone!", "DistilBERT SST-2"],
    ["This is the worst thing I've ever purchased. Terrible quality.", "BERT Base (SST-2)"],
]


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Sentiment Analysis Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 Sentiment Analysis Playground

            Analyze the sentiment of text using state-of-the-art transformer models.
            Try different models and see how they perform on your text!
            """
        )

        with gr.Tab("Single Text Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Type or paste your text here...",
                        lines=5
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(SENTIMENT_MODELS.keys()),
                        value="Twitter RoBERTa (Multilingual)",
                        label="Select Model"
                    )
                    analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

                with gr.Column(scale=2):
                    output_json = gr.JSON(label="Results")
                    output_plot = gr.Plot(label="Confidence Visualization")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, model_dropdown],
                label="Try these examples"
            )

        with gr.Tab("Batch Analysis"):
            gr.Markdown(
                """
                ### Process multiple texts at once
                Enter multiple texts separated by new lines. Each line will be analyzed separately.
                """
            )

            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Enter multiple texts (one per line)",
                        placeholder="Text 1\nText 2\nText 3\n...",
                        lines=10
                    )
                    batch_model = gr.Dropdown(
                        choices=list(SENTIMENT_MODELS.keys()),
                        value="Twitter RoBERTa (Multilingual)",
                        label="Select Model"
                    )
                    batch_btn = gr.Button("Analyze All", variant="primary")

                with gr.Column():
                    batch_output = gr.Textbox(
                        label="Batch Results",
                        lines=15,
                        max_lines=20
                    )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Tool

                This sentiment analysis playground allows you to experiment with different
                transformer-based sentiment analysis models.

                ### Available Models:

                1. **Twitter RoBERTa (Multilingual)** - Trained on multilingual Twitter data
                2. **Twitter RoBERTa (English)** - Optimized for English Twitter text
                3. **DistilBERT SST-2** - Fast and efficient, trained on Stanford Sentiment Treebank
                4. **BERT Base (SST-2)** - BERT model fine-tuned on SST-2 dataset

                ### Features:
                - Real-time sentiment analysis
                - Confidence scores for all sentiment classes
                - Visual confidence distribution
                - Batch processing support
                - Multiple model comparison

                ### Tips:
                - Shorter texts (< 512 characters) work best
                - Try the same text with different models to compare
                - Use batch mode for analyzing multiple reviews or comments
                """
            )

        # Connect the components
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=[text_input, model_dropdown],
            outputs=[output_json, output_plot]
        )

        batch_btn.click(
            fn=analyze_batch,
            inputs=[batch_input, batch_model],
            outputs=[batch_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
