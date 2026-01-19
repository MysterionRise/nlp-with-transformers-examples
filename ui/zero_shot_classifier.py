"""
Zero-Shot Classifier - Interactive UI for zero-shot text classification

Features:
- Real-time zero-shot classification without training data
- Multiple model comparison
- Dynamic candidate label input
- Multi-label classification support
- Confidence visualization
- Batch processing support
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr
import plotly.graph_objects as go
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available zero-shot classification models
ZS_MODELS = {
    "BART Large MNLI": "facebook/bart-large-mnli",
    "DeBERTa MNLI": "microsoft/deberta-v3-base",
}

# Cache for loaded models
model_cache = {}

# Pre-defined label sets for common tasks
LABEL_PRESETS = {
    "Sentiment": "positive, negative, neutral",
    "Intent": "question, statement, command, small talk",
    "Category": "sports, politics, technology, entertainment, science",
    "Emotion": "joy, sadness, anger, surprise, fear, disgust",
    "Toxicity": "toxic, hateful, offensive, safe",
    "Language": "English, Spanish, French, German, Chinese",
}


def load_model(model_name: str):
    """Load or retrieve cached zero-shot classification model"""
    if model_name not in model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            model_cache[model_name] = pipeline(
                "zero-shot-classification",
                model=ZS_MODELS[model_name],
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    return model_cache[model_name]


def classify_text(
    text: str,
    candidate_labels: str,
    model_name: str,
    multi_class: bool = False,
) -> Tuple[Dict, str]:
    """
    Classify text into candidate labels without training data

    Args:
        text: Input text to classify
        candidate_labels: Comma-separated candidate labels
        model_name: Name of the model to use
        multi_class: Whether to treat as multi-label problem

    Returns:
        Tuple of (results dict, visualization HTML)
    """
    if not text or not text.strip():
        return {"error": "Please provide text to classify"}, ""

    if not candidate_labels or not candidate_labels.strip():
        return {"error": "Please provide candidate labels (comma-separated)"}, ""

    try:
        # Parse labels
        labels = [label.strip() for label in candidate_labels.split(",")]
        labels = [l for l in labels if l]  # Remove empty strings

        if not labels:
            return {"error": "Invalid label format"}, ""

        # Load model
        classifier = load_model(model_name)

        # Classify
        result = classifier(
            text,
            labels,
            multi_class=multi_class,
        )

        # Format results
        formatted_results = {
            "Text": text[:200] + ("..." if len(text) > 200 else ""),
            "Top Classification": result["labels"][0],
            "Confidence": f"{result['scores'][0]*100:.2f}%",
            "Model": model_name,
            "Multi-class": multi_class,
            "All Scores": {label: f"{score*100:.2f}%" for label, score in zip(result["labels"], result["scores"])},
        }

        # Create visualization
        fig = go.Figure(
            data=[
                go.Bar(
                    x=result["scores"],
                    y=result["labels"],
                    orientation="h",
                    marker=dict(color=result["scores"], colorscale="Viridis", showscale=False),
                    text=[f"{s*100:.1f}%" for s in result["scores"]],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Classification Scores",
            xaxis_title="Confidence",
            yaxis_title="Labels",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 1]),
        )

        return formatted_results, fig

    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        return {"error": f"Error: {str(e)}"}, None


def batch_classify(
    batch_text: str,
    candidate_labels: str,
    model_name: str,
    multi_class: bool = False,
    delimiter: str = "\n",
) -> str:
    """
    Classify multiple texts

    Args:
        batch_text: Multiple texts separated by delimiter
        candidate_labels: Comma-separated candidate labels
        model_name: Model to use
        multi_class: Multi-class mode
        delimiter: Delimiter for splitting texts

    Returns:
        Formatted results as string
    """
    if not batch_text or not batch_text.strip():
        return "Please provide texts to classify"

    if not candidate_labels or not candidate_labels.strip():
        return "Please provide candidate labels"

    try:
        # Parse labels
        labels = [label.strip() for label in candidate_labels.split(",")]
        labels = [l for l in labels if l]

        if not labels:
            return "Invalid label format"

        # Split texts
        texts = [t.strip() for t in batch_text.split(delimiter) if t.strip()]

        if not texts:
            return "No valid texts found"

        # Load model
        classifier = load_model(model_name)

        results = []
        for i, txt in enumerate(texts, 1):
            try:
                result = classifier(txt, labels, multi_class=multi_class)

                top_label = result["labels"][0]
                top_score = result["scores"][0]

                results.append(
                    f"{i}. Text: {txt[:60]}{'...' if len(txt) > 60 else ''}\n"
                    f"   Classification: {top_label} ({top_score*100:.2f}%)\n"
                )
            except Exception as e:
                results.append(f"{i}. Error: {str(e)}\n")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return f"Error: {str(e)}"


# Example texts and labels for testing
EXAMPLES = [
    [
        "This movie was amazing! I loved every second of it.",
        "positive, negative, neutral",
        "BART Large MNLI",
        False,
    ],
    [
        "What time does the store open tomorrow?",
        "question, statement, command, small talk",
        "BART Large MNLI",
        False,
    ],
    [
        "I'm really excited about the new AI developments in tech.",
        "sports, politics, technology, entertainment, science",
        "DeBERTa MNLI",
        False,
    ],
    [
        "I feel so sad and disappointed right now.",
        "joy, sadness, anger, surprise, fear, disgust",
        "BART Large MNLI",
        False,
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Zero-Shot Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏷️ Zero-Shot Classifier

            Classify text into any categories without training data!
            Simply provide your text and the labels you want to classify into.
            """
        )

        with gr.Tab("Single Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Enter text to classify",
                        placeholder="Type or paste your text here...",
                        lines=5,
                    )
                    labels_input = gr.Textbox(
                        label="Candidate Labels (comma-separated)",
                        placeholder="positive, negative, neutral",
                        lines=2,
                    )

                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=list(ZS_MODELS.keys()),
                        value="BART Large MNLI",
                        label="Select Model",
                    )
                    multi_class_toggle = gr.Checkbox(
                        label="Multi-class (text can belong to multiple labels)",
                        value=False,
                    )
                    classify_btn = gr.Button("Classify", variant="primary")

            with gr.Row():
                with gr.Column():
                    output_json = gr.JSON(label="Classification Results")
                with gr.Column():
                    output_plot = gr.Plot(label="Score Visualization")

            gr.Markdown("### Label Presets")
            with gr.Row():
                preset_buttons = []
                for preset_name, preset_labels in LABEL_PRESETS.items():
                    preset_buttons.append(gr.Button(preset_name, size="sm"))

                # Connect preset buttons
                for preset_labels_text, btn in zip(LABEL_PRESETS.values(), preset_buttons):
                    btn.click(lambda x=preset_labels_text: x, outputs=labels_input)

            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, labels_input, model_dropdown, multi_class_toggle],
                label="Try these examples",
            )

        with gr.Tab("Batch Classification"):
            gr.Markdown(
                """
                ### Classify multiple texts at once
                Enter multiple texts separated by new lines and provide candidate labels.
                """
            )

            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Enter texts (one per line)",
                        placeholder="Text 1\nText 2\nText 3\n...",
                        lines=10,
                    )
                    batch_labels = gr.Textbox(
                        label="Candidate Labels",
                        placeholder="label1, label2, label3",
                        lines=2,
                    )

                with gr.Column():
                    batch_model = gr.Dropdown(
                        choices=list(ZS_MODELS.keys()),
                        value="BART Large MNLI",
                        label="Select Model",
                    )
                    batch_multi_class = gr.Checkbox(
                        label="Multi-class",
                        value=False,
                    )
                    batch_btn = gr.Button("Classify All", variant="primary")

            batch_output = gr.Textbox(label="Batch Results", lines=15, max_lines=20)

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Tool

                Zero-shot classification allows you to classify text into any categories
                without needing training examples. This is powered by natural language inference models.

                ### Available Models:

                1. **BART Large MNLI** - Trained on MNLI, good balance of speed and accuracy
                2. **DeBERTa MNLI** - More advanced, typically better accuracy

                ### How It Works:

                - The model treats classification as an entailment problem
                - For each label, it checks if the text entails "This text is about [label]"
                - Scores represent the probability of entailment for each label

                ### Parameters:

                **Multi-class Mode:**
                - Off (default): Each text assigned to most likely single label
                - On: Text can be assigned to multiple labels with confidence scores

                ### Use Cases:

                - **Sentiment Analysis:** positive, negative, neutral
                - **Intent Detection:** question, statement, command, small talk
                - **Category Classification:** sports, technology, politics, etc.
                - **Emotion Detection:** joy, sadness, anger, fear, disgust, surprise
                - **Content Moderation:** toxic, hateful, safe
                - **Language Identification:** English, Spanish, French, etc.

                ### Tips:

                - Use clear, descriptive labels for best results
                - Shorter labels typically work better than longer ones
                - Be specific with labels (e.g., "sports news" vs just "sports")
                - Experiment with different label sets for the same text
                - Use multi-class mode when texts can belong to multiple categories

                ### Limitations:

                - Works best with clear, unambiguous text
                - May struggle with very short texts
                - Requires good label descriptions
                - Computationally more expensive than supervised classification
                """
            )

        # Connect the components
        classify_btn.click(
            fn=classify_text,
            inputs=[text_input, labels_input, model_dropdown, multi_class_toggle],
            outputs=[output_json, output_plot],
        )

        batch_btn.click(
            fn=batch_classify,
            inputs=[batch_input, batch_labels, batch_model, batch_multi_class],
            outputs=[batch_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7867, share=False, show_error=True)
