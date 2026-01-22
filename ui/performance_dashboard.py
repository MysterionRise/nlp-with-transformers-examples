"""
Model Performance Dashboard - Interactive UI for comparing model performance

Features:
- Compare multiple models side-by-side
- Evaluate outputs with multiple metrics (BLEU, ROUGE, METEOR, BERTScore)
- Visualize performance across different metrics
- Export results and charts
- Manage model cache
"""

import json
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import get_model_registry
from utils import get_cache_info, get_logger
from utils.error_handler import handle_errors

logger = get_logger(__name__)

# Import evaluation metrics
try:
    import nltk
    from bert_score import score as bert_score
    from nltk.translate import meteor_score
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Download required NLTK data
    for resource in ["punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    EVAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evaluation metrics not available: {e}")
    EVAL_AVAILABLE = False


@handle_errors(default_return={}, user_message="Failed to evaluate scores")
def evaluate_text(reference: str, candidate: str) -> Dict:
    """
    Evaluate candidate text against reference using multiple metrics

    Args:
        reference: Reference/ground truth text
        candidate: Generated/candidate text to evaluate

    Returns:
        Dictionary of evaluation scores
    """
    if not EVAL_AVAILABLE:
        return {"error": "Evaluation libraries not installed"}

    if not reference or not candidate:
        return {"error": "Both reference and candidate text are required"}

    try:
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)

        # BLEU Score
        bleu = sentence_bleu([reference_tokens], candidate_tokens)

        # ROUGE Scores
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)

        # METEOR Score
        meteor = meteor_score.single_meteor_score(reference_tokens, candidate_tokens)

        # Cosine Similarity
        vectorizer = TfidfVectorizer().fit_transform([reference, candidate])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]

        # BERTScore
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)

        return {
            "BLEU": round(bleu, 4),
            "ROUGE-1": round(rouge_scores["rouge1"].fmeasure, 4),
            "ROUGE-2": round(rouge_scores["rouge2"].fmeasure, 4),
            "ROUGE-L": round(rouge_scores["rougeL"].fmeasure, 4),
            "METEOR": round(meteor, 4),
            "Cosine Similarity": round(cosine_sim, 4),
            "BERTScore P": round(P.mean().item(), 4),
            "BERTScore R": round(R.mean().item(), 4),
            "BERTScore F1": round(F1.mean().item(), 4),
        }
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}


def create_comparison_chart(results: List[Dict]) -> go.Figure:
    """
    Create radar chart comparing multiple model results

    Args:
        results: List of evaluation results with model names

    Returns:
        Plotly figure
    """
    if not results:
        return go.Figure()

    # Define metrics to visualize
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore F1"]

    fig = go.Figure()

    for result in results:
        model_name = result.get("model", "Unknown")
        values = [result.get(metric, 0) for metric in metrics]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill="toself",
                name=model_name,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Comparison (Radar Chart)",
        height=500,
    )

    return fig


def create_metric_bars(results: List[Dict]) -> go.Figure:
    """
    Create bar chart for metric comparison

    Args:
        results: List of evaluation results

    Returns:
        Plotly figure
    """
    if not results:
        return go.Figure()

    metrics = ["BLEU", "ROUGE-1", "ROUGE-L", "METEOR", "BERTScore F1"]
    models = [r.get("model", f"Model {i + 1}") for i, r in enumerate(results)]

    fig = go.Figure()

    for metric in metrics:
        values = [r.get(metric, 0) for r in results]
        fig.add_trace(go.Bar(name=metric, x=models, y=values))

    fig.update_layout(
        barmode="group",
        title="Metrics Comparison Across Models",
        xaxis_title="Models",
        yaxis_title="Score",
        height=400,
        yaxis=dict(range=[0, 1]),
    )

    return fig


def batch_evaluate(
    reference: str, candidates_text: str, delimiter: str = "\n---\n"
) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
    """
    Evaluate multiple candidates against a reference

    Args:
        reference: Reference text
        candidates_text: Text containing multiple candidates
        delimiter: Delimiter to split candidates

    Returns:
        Tuple of (DataFrame, radar chart, bar chart)
    """
    if not reference or not candidates_text:
        return pd.DataFrame(), go.Figure(), go.Figure()

    candidates = [c.strip() for c in candidates_text.split(delimiter) if c.strip()]

    results = []
    for i, candidate in enumerate(candidates):
        scores = evaluate_text(reference, candidate)
        if "error" not in scores:
            scores["model"] = f"Model {i + 1}"
            scores["candidate"] = candidate[:100] + "..." if len(candidate) > 100 else candidate
            results.append(scores)

    if not results:
        return pd.DataFrame(), go.Figure(), go.Figure()

    df = pd.DataFrame(results)

    # Reorder columns
    cols = [
        "model",
        "candidate",
        "BLEU",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "METEOR",
        "BERTScore F1",
    ]
    df = df[[col for col in cols if col in df.columns]]

    radar_chart = create_comparison_chart(results)
    bar_chart = create_metric_bars(results)

    return df, radar_chart, bar_chart


def get_cache_stats() -> str:
    """Get formatted cache statistics"""
    try:
        stats = get_cache_info()
        return f"""### Cache Statistics

**Cached Models:** {stats["cached_models"]} / {stats["max_size"]}
**Device:** {stats["device"]}

**Loaded Models:**
{chr(10).join(f"- {model}" for model in stats.get("models", [])) if stats.get("models") else "No models loaded"}
"""
    except Exception as e:
        return f"Error getting cache stats: {str(e)}"


def list_available_models() -> str:
    """List all available models by category"""
    try:
        registry = get_model_registry()
        categories = registry.list_categories()

        output = ["### Available Models\n"]
        for category in categories:
            models = registry.get_category(category)
            output.append(f"\n**{category.replace('_', ' ').title()}:**")
            for key, config in models.items():
                output.append(f"- {config.name} (`{config.model_id}`)")

        return "\n".join(output)
    except Exception as e:
        return f"Error listing models: {str(e)}"


def export_results(df: pd.DataFrame) -> str:
    """Export results to JSON"""
    if df.empty:
        return "{}"
    return df.to_json(orient="records", indent=2)


# Example texts
EXAMPLE_REFERENCE = (
    "The investment strategy involves diversifying assets across various sectors to "
    "mitigate risks and maximize returns. This approach ensures a balanced portfolio "
    "that can withstand market volatility while providing steady growth opportunities."
)

EXAMPLE_CANDIDATE_1 = (
    "The strategy for investments involves diversifying assets across multiple sectors "
    "to minimize risks and optimize returns. This method ensures a balanced portfolio "
    "that can handle market fluctuations while offering consistent growth opportunities."
)

EXAMPLE_CANDIDATE_2 = (
    "An investment approach focuses on spreading assets across different sectors to "
    "reduce risk and increase profits. This strategy creates a stable portfolio "
    "resistant to market changes and provides continuous growth potential."
)

EXAMPLE_CANDIDATE_3 = (
    "The financial plan includes asset diversification across various industries to "
    "lower risks and enhance returns. This ensures portfolio balance and resilience "
    "against market volatility with sustainable growth prospects."
)


def create_ui():
    """Create the Performance Dashboard UI"""

    with gr.Blocks(title="Model Performance Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📊 Model Performance Dashboard

        Compare and evaluate model outputs using multiple metrics including BLEU, ROUGE, METEOR, and BERTScore.
        """)

        with gr.Tabs():
            # Tab 1: Single Comparison
            with gr.Tab("Single Comparison"):
                gr.Markdown("### Compare Two Texts")

                with gr.Row():
                    with gr.Column():
                        ref_input = gr.Textbox(
                            label="Reference Text",
                            placeholder="Enter the reference/ground truth text...",
                            lines=5,
                            value=EXAMPLE_REFERENCE,
                        )
                    with gr.Column():
                        cand_input = gr.Textbox(
                            label="Candidate Text",
                            placeholder="Enter the candidate/generated text...",
                            lines=5,
                            value=EXAMPLE_CANDIDATE_1,
                        )

                eval_btn = gr.Button("Evaluate", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        scores_json = gr.JSON(label="Evaluation Scores")
                    with gr.Column():
                        scores_md = gr.Markdown(label="Formatted Scores")

                def format_scores(scores):
                    if "error" in scores:
                        return f"**Error:** {scores['error']}"

                    md = "### Evaluation Results\n\n"
                    md += "| Metric | Score |\n|--------|-------|\n"
                    for metric, score in scores.items():
                        md += f"| {metric} | {score:.4f} |\n"
                    return md

                eval_btn.click(
                    fn=evaluate_text,
                    inputs=[ref_input, cand_input],
                    outputs=[scores_json],
                ).then(fn=format_scores, inputs=[scores_json], outputs=[scores_md])

            # Tab 2: Batch Comparison
            with gr.Tab("Batch Comparison"):
                gr.Markdown("""
                ### Compare Multiple Candidates

                Enter multiple candidate texts separated by `---` to compare them against a reference.
                """)

                batch_ref = gr.Textbox(
                    label="Reference Text",
                    lines=4,
                    value=EXAMPLE_REFERENCE,
                    placeholder="Enter reference text...",
                )

                batch_cands = gr.Textbox(
                    label="Candidate Texts (separated by ---)",
                    lines=10,
                    value=f"{EXAMPLE_CANDIDATE_1}\n---\n{EXAMPLE_CANDIDATE_2}\n---\n{EXAMPLE_CANDIDATE_3}",
                    placeholder="Enter multiple candidates separated by ---",
                )

                batch_delimiter = gr.Textbox(label="Delimiter", value="\n---\n", max_lines=1)

                batch_btn = gr.Button("Compare All", variant="primary", size="lg")

                results_table = gr.Dataframe(label="Results Table", wrap=True)

                with gr.Row():
                    radar_chart = gr.Plot(label="Radar Chart")
                    bar_chart = gr.Plot(label="Bar Chart")

                batch_btn.click(
                    fn=batch_evaluate,
                    inputs=[batch_ref, batch_cands, batch_delimiter],
                    outputs=[results_table, radar_chart, bar_chart],
                )

            # Tab 3: Model Registry
            with gr.Tab("Model Registry"):
                gr.Markdown("### Available Models & Cache Status")

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm")

                with gr.Row():
                    with gr.Column():
                        cache_stats = gr.Markdown(value=get_cache_stats())
                    with gr.Column():
                        available_models = gr.Markdown(value=list_available_models())

                refresh_btn.click(fn=get_cache_stats, outputs=[cache_stats]).then(
                    fn=list_available_models, outputs=[available_models]
                )

            # Tab 4: Export
            with gr.Tab("Export"):
                gr.Markdown("### Export Evaluation Results")

                export_input = gr.Dataframe(label="Results to Export (from Batch Comparison)")
                export_btn = gr.Button("Generate JSON Export", variant="primary")
                export_output = gr.Code(label="Exported JSON", language="json")

                export_btn.click(fn=export_results, inputs=[export_input], outputs=[export_output])

        gr.Markdown("""
        ---
        ### Metrics Explanation

        - **BLEU**: Measures n-gram overlap (0-1, higher is better)
        - **ROUGE-1/2/L**: Measures recall of unigrams, bigrams, and longest common subsequence
        - **METEOR**: Considers synonyms and stemming (0-1, higher is better)
        - **Cosine Similarity**: TF-IDF based similarity (0-1, higher is better)
        - **BERTScore**: Contextual embeddings similarity (precision, recall, F1)
        """)

    return demo


if __name__ == "__main__":
    logger.info("Starting Model Performance Dashboard")
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7864, share=False)
