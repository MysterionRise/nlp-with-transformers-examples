"""
Question Answering System - Interactive UI for extractive QA

Features:
- Real-time question answering on provided context
- Multiple model comparison
- Answer span highlighting with confidence scores
- Batch processing support
- Visual confidence display
"""

import logging
from typing import Dict, Tuple

import gradio as gr
import plotly.graph_objects as go
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available question answering models
QA_MODELS = {
    "DistilBERT SQuAD": "distilbert-base-cased-distilled-squad",
    "RoBERTa SQuAD": "deepset/roberta-base-squad2",
    "BERT SQuAD": "bert-large-uncased-whole-word-masking-finetuned-squad",
}

# Cache for loaded models
model_cache = {}


def load_model(model_name: str):
    """Load or retrieve cached question answering model"""
    if model_name not in model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            model_cache[model_name] = pipeline(
                "question-answering",
                model=QA_MODELS[model_name],
                tokenizer=QA_MODELS[model_name],
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    return model_cache[model_name]


def highlight_answer(context: str, answer: str, start_char: int, end_char: int) -> str:
    """
    Create HTML with highlighted answer span

    Args:
        context: Full context text
        answer: The extracted answer
        start_char: Start character index
        end_char: End character index

    Returns:
        HTML string with highlighted answer
    """
    try:
        before = context[:start_char]
        highlighted = context[start_char:end_char]
        after = context[end_char:]

        mark_style = "background-color: #FFD700; padding: 2px 4px; border-radius: 3px;"
        html = (
            f'<div style="padding: 10px; background-color: #f5f5f5; '
            f'border-radius: 5px; line-height: 1.8;">'
            f"{before}"
            f'<mark style="{mark_style}">{highlighted}</mark>'
            f"{after}"
            f"</div>"
        )
        return html
    except Exception as e:
        logger.error(f"Error highlighting answer: {e}")
        return context


def answer_question(context: str, question: str, model_name: str) -> Tuple[Dict, str]:
    """
    Answer a question based on the provided context

    Args:
        context: Context text to extract answer from
        question: Question to answer
        model_name: Name of the model to use

    Returns:
        Tuple of (results dict, highlighted HTML)
    """
    if not context or not context.strip():
        return {"error": "Please provide context text"}, ""

    if not question or not question.strip():
        return {"error": "Please provide a question"}, ""

    try:
        # Limit context to 512 tokens for efficiency
        context_limited = context[:2000]

        # Load model
        qa_model = load_model(model_name)

        # Get answer
        result = qa_model(question=question, context=context_limited)

        # Format results
        formatted_results = {
            "Answer": result["answer"],
            "Confidence": f"{result['score'] * 100:.2f}%",
            "Start": result["start"],
            "End": result["end"],
        }

        # Create highlighted HTML
        highlighted_html = highlight_answer(context_limited, result["answer"], result["start"], result["end"])

        # Create confidence visualization
        fig = go.Figure(
            data=[
                go.Indicator(
                    mode="gauge+number+delta",
                    value=result["score"] * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Answer Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "#FFE6E6"},
                            {"range": [50, 80], "color": "#FFF9E6"},
                            {"range": [80, 100], "color": "#E6F9E6"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 70,
                        },
                    },
                )
            ]
        )

        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))

        return formatted_results, highlighted_html, fig

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"error": f"Error: {str(e)}"}, "", None


def batch_answer(batch_text: str, model_name: str, delimiter: str = "\n---\n") -> str:
    """
    Answer questions from a batch of context+question pairs

    Args:
        batch_text: Batch text with context and questions separated by delimiter
        model_name: Name of the model to use
        delimiter: Delimiter separating context-question pairs

    Returns:
        Formatted results as string
    """
    if not batch_text or not batch_text.strip():
        return "Please provide context and questions"

    try:
        qa_model = load_model(model_name)

        # Split batch
        pairs = [p.strip() for p in batch_text.split(delimiter) if p.strip()]

        if not pairs:
            return "No valid pairs found"

        results = []
        for i, pair in enumerate(pairs, 1):
            lines = [l.strip() for l in pair.split("\n") if l.strip()]

            if len(lines) < 2:
                results.append(f"{i}. Invalid format (need context + question)")
                continue

            context = lines[0]
            question = lines[1]

            try:
                answer_result = qa_model(question=question, context=context[:2000])
                results.append(
                    f"{i}. Q: {question[:60]}{'...' if len(question) > 60 else ''}\n"
                    f"   A: {answer_result['answer']}\n"
                    f"   Confidence: {answer_result['score'] * 100:.2f}%\n"
                )
            except Exception as e:
                results.append(f"{i}. Error processing pair: {str(e)}\n")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return f"Error: {str(e)}"


# Example context and questions for testing
EXAMPLES = [
    [
        (
            "The Eiffel Tower is located in Paris, France. It was built in 1889 for "
            "the World's Fair. The tower stands 330 meters tall and is made of iron "
            "lattice work. It has become one of the most recognizable landmarks in "
            "the world."
        ),
        "Where is the Eiffel Tower located?",
        "DistilBERT SQuAD",
    ],
    [
        (
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn from data without being explicitly programmed. It "
            "uses algorithms to identify patterns and make predictions based on "
            "input data."
        ),
        "What is machine learning?",
        "DistilBERT SQuAD",
    ],
    [
        (
            "Python is a high-level programming language known for its simplicity "
            "and readability. It was created by Guido van Rossum and first released "
            "in 1991. Python is widely used in web development, data analysis, and "
            "artificial intelligence."
        ),
        "When was Python first released?",
        "RoBERTa SQuAD",
    ],
    [
        (
            "The Great Wall of China is one of the most impressive structures in "
            "the world. Construction began in the 7th century BC and continued for "
            "centuries. The wall stretches over 13,000 miles and was built to "
            "protect against invasions."
        ),
        "How long is the Great Wall of China?",
        "BERT SQuAD",
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Question Answering System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # ❓ Question Answering System

            Answer questions based on a provided context using state-of-the-art extractive QA models.
            Simply provide context and ask a question, and the model will find the answer!
            """)

        with gr.Tab("Single Question"):
            with gr.Row():
                with gr.Column(scale=1):
                    context_input = gr.Textbox(
                        label="Context (provide the text to search for answers)",
                        placeholder="Paste your context here...",
                        lines=8,
                    )

                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="What would you like to know?",
                        lines=3,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(QA_MODELS.keys()),
                        value="DistilBERT SQuAD",
                        label="Select Model",
                    )
                    answer_btn = gr.Button("Get Answer", variant="primary")

            with gr.Row():
                with gr.Column():
                    output_json = gr.JSON(label="Answer Details")
                with gr.Column():
                    output_confidence = gr.Plot(label="Confidence Score")

            output_highlight = gr.HTML(label="Context with Answer Highlighted")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[context_input, question_input, model_dropdown],
                label="Try these examples",
            )

        with gr.Tab("Batch Processing"):
            gr.Markdown("""
                ### Process multiple context-question pairs
                Format your input as:
                ```
                Context text here
                Question 1?
                ---
                Context text here
                Question 2?
                ```
                Separate each pair with `---`
                """)

            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Context-Question Pairs",
                        placeholder="Context...\nQuestion?\n---\nContext...\nQuestion?\n",
                        lines=15,
                        max_lines=25,
                    )
                    batch_model = gr.Dropdown(
                        choices=list(QA_MODELS.keys()),
                        value="DistilBERT SQuAD",
                        label="Select Model",
                    )
                    batch_btn = gr.Button("Process Batch", variant="primary")

                with gr.Column():
                    batch_output = gr.Textbox(label="Batch Results", lines=15, max_lines=25)

        with gr.Tab("About"):
            gr.Markdown("""
                ## About This Tool

                This Question Answering system uses extractive QA models to find answers within provided context.
                The models identify answer spans in the context text that best match the given question.

                ### Available Models:

                1. **DistilBERT SQuAD** - Lightweight model, fastest inference
                2. **RoBERTa SQuAD** - Balanced performance, based on SQuAD 2.0
                3. **BERT SQuAD** - Large model, best accuracy

                ### How It Works:

                - Extractive QA identifies answer spans within the provided context
                - The model returns the most likely answer and its confidence score
                - Confidence score indicates how certain the model is about the answer

                ### Tips:

                - Provide clear, factual context for best results
                - Keep context under 2000 characters for faster processing
                - Try different models to compare performance
                - Use batch mode to process multiple Q&A pairs efficiently

                ### Limitations:

                - Can only extract answers from provided context
                - Works best with factual, well-structured text
                - May struggle with complex or ambiguous questions
                """)

        # Connect the components
        answer_btn.click(
            fn=answer_question,
            inputs=[context_input, question_input, model_dropdown],
            outputs=[output_json, output_highlight, output_confidence],
        )

        batch_btn.click(fn=batch_answer, inputs=[batch_input, batch_model], outputs=[batch_output])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False, show_error=True)
