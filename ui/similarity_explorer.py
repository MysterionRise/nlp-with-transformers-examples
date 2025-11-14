"""
Sentence Similarity Explorer - Interactive UI for semantic similarity analysis

Features:
- Pairwise sentence comparison
- Batch similarity analysis
- Embedding visualization (2D/3D with t-SNE, UMAP, PCA)
- Similarity heatmap
- Semantic search (find most similar sentences)
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None


def load_model():
    """Load the sentence transformer model (lazy loading)"""
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        logger.info("Model loaded successfully")


def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to get sentence embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of sentences

    Args:
        sentences: List of input sentences

    Returns:
        Normalized embeddings as numpy array
    """
    load_model()

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Apply mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.numpy()


def compare_two_sentences(text1: str, text2: str) -> Tuple[float, str, Dict]:
    """
    Compare two sentences and return similarity score with visualization

    Args:
        text1: First sentence
        text2: Second sentence

    Returns:
        Tuple of (similarity score, interpretation text, results dict)
    """
    if not text1 or not text2:
        return 0.0, "Please provide both sentences", {}

    try:
        # Get embeddings
        embeddings = get_embeddings([text1, text2])

        # Calculate cosine similarity
        similarity = float(np.dot(embeddings[0], embeddings[1]))

        # Interpretation
        if similarity > 0.9:
            interpretation = "🟢 Very High Similarity - Nearly identical meaning"
        elif similarity > 0.7:
            interpretation = "🟡 High Similarity - Very related topics"
        elif similarity > 0.5:
            interpretation = "🟠 Moderate Similarity - Somewhat related"
        elif similarity > 0.3:
            interpretation = "🔵 Low Similarity - Loosely related"
        else:
            interpretation = "🔴 Very Low Similarity - Different topics"

        # Create gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=similarity,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Semantic Similarity Score"},
                delta={"reference": 0.5},
                gauge={
                    "axis": {"range": [None, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.3], "color": "lightgray"},
                        {"range": [0.3, 0.5], "color": "lightblue"},
                        {"range": [0.5, 0.7], "color": "lightyellow"},
                        {"range": [0.7, 0.9], "color": "lightgreen"},
                        {"range": [0.9, 1.0], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 0.8,
                    },
                },
            )
        )

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        results = {
            "Similarity Score": f"{similarity:.4f}",
            "Percentage": f"{similarity * 100:.2f}%",
            "Interpretation": interpretation,
        }

        return similarity, interpretation, fig

    except Exception as e:
        logger.error(f"Error comparing sentences: {e}")
        return 0.0, f"Error: {str(e)}", {}


def batch_similarity(text: str, query: str) -> Tuple[str, object]:
    """
    Compare a query sentence against multiple sentences

    Args:
        text: Multiple sentences (one per line)
        query: Query sentence to compare

    Returns:
        Tuple of (formatted results, similarity bar chart)
    """
    if not text or not query:
        return "Please provide both query and sentences", None

    try:
        # Parse sentences
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        if not sentences:
            return "No valid sentences found", None

        # Get embeddings for all sentences + query
        all_texts = sentences + [query]
        embeddings = get_embeddings(all_texts)

        # Query embedding is the last one
        query_emb = embeddings[-1]
        sentence_embs = embeddings[:-1]

        # Calculate similarities
        similarities = [float(np.dot(query_emb, emb)) for emb in sentence_embs]

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Format results
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            sent = sentences[idx]
            sim = similarities[idx]
            results.append(
                f"{rank}. [Score: {sim:.4f}] {sent[:100]}{'...' if len(sent) > 100 else ''}\n"
            )

        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    y=[f"Sentence {i+1}" for i in sorted_indices[:10]],  # Top 10
                    x=[similarities[i] for i in sorted_indices[:10]],
                    orientation="h",
                    marker=dict(
                        color=[similarities[i] for i in sorted_indices[:10]],
                        colorscale="Viridis",
                        showscale=True,
                    ),
                    text=[f"{similarities[i]:.3f}" for i in sorted_indices[:10]],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Top 10 Most Similar Sentences",
            xaxis_title="Similarity Score",
            yaxis_title="Sentences",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return "".join(results), fig

    except Exception as e:
        logger.error(f"Error in batch similarity: {e}")
        return f"Error: {str(e)}", None


def visualize_embeddings(text: str, method: str = "t-SNE") -> object:
    """
    Visualize sentence embeddings in 2D space

    Args:
        text: Multiple sentences (one per line)
        method: Dimensionality reduction method (t-SNE, PCA, UMAP)

    Returns:
        Plotly figure
    """
    if not text:
        return None

    try:
        # Parse sentences
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        if len(sentences) < 2:
            return None

        # Get embeddings
        embeddings = get_embeddings(sentences)

        # Apply dimensionality reduction
        if method == "t-SNE":
            reducer = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(sentences) - 1)
            )
        elif method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        else:  # UMAP would require umap-learn
            reducer = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(sentences) - 1)
            )

        coords = reducer.fit_transform(embeddings)

        # Create labels (truncated sentences)
        labels = [s[:50] + "..." if len(s) > 50 else s for s in sentences]

        # Create scatter plot
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=np.arange(len(sentences)),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Sentence ID"),
                    ),
                    text=[f"{i+1}" for i in range(len(sentences))],
                    textposition="top center",
                    hovertext=labels,
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=f"Sentence Embeddings Visualization ({method})",
            xaxis_title=f"{method} Dimension 1",
            yaxis_title=f"{method} Dimension 2",
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    except Exception as e:
        logger.error(f"Error visualizing embeddings: {e}")
        return None


def create_similarity_matrix(text: str) -> object:
    """
    Create a similarity heatmap for all sentence pairs

    Args:
        text: Multiple sentences (one per line)

    Returns:
        Plotly heatmap figure
    """
    if not text:
        return None

    try:
        # Parse sentences
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        if len(sentences) < 2:
            return None

        # Get embeddings
        embeddings = get_embeddings(sentences)

        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Create labels
        labels = [f"S{i+1}" for i in range(len(sentences))]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=similarity_matrix,
                x=labels,
                y=labels,
                colorscale="RdYlGn",
                text=np.round(similarity_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Similarity"),
            )
        )

        fig.update_layout(
            title="Sentence Similarity Matrix",
            xaxis_title="Sentences",
            yaxis_title="Sentences",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating similarity matrix: {e}")
        return None


# Example sentences
EXAMPLE_PAIRS = [
    ["The cat sat on the mat.", "A feline rested on the rug."],
    ["Python is a programming language.", "Java is used for software development."],
    ["I love eating pizza.", "Pizza is my favorite food."],
    ["The weather is sunny today.", "It's raining heavily outside."],
    ["Machine learning is fascinating.", "AI and ML are interesting topics."],
]

EXAMPLE_BATCH = """The cat sat on the mat.
Dogs are loyal animals.
Python is a programming language.
The weather is sunny today.
I enjoy reading books.
Coffee tastes great in the morning."""


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(
        title="Sentence Similarity Explorer", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Sentence Similarity Explorer

            Explore semantic similarity between sentences using transformer-based embeddings.
            Discover how AI understands meaning beyond just matching words!
            """
        )

        with gr.Tab("Pairwise Comparison"):
            gr.Markdown(
                "### Compare two sentences to see how semantically similar they are"
            )

            with gr.Row():
                with gr.Column():
                    sent1_input = gr.Textbox(
                        label="First Sentence",
                        placeholder="Enter the first sentence...",
                        lines=3,
                    )
                    sent2_input = gr.Textbox(
                        label="Second Sentence",
                        placeholder="Enter the second sentence...",
                        lines=3,
                    )
                    compare_btn = gr.Button("Compare Sentences", variant="primary")

                with gr.Column():
                    similarity_output = gr.Textbox(label="Interpretation")
                    similarity_plot = gr.Plot(label="Similarity Gauge")

            gr.Examples(
                examples=EXAMPLE_PAIRS,
                inputs=[sent1_input, sent2_input],
                label="Try these examples",
            )

        with gr.Tab("Semantic Search"):
            gr.Markdown("### Find the most similar sentences to your query")

            with gr.Row():
                with gr.Column():
                    batch_text = gr.Textbox(
                        label="Sentences (one per line)",
                        placeholder="Sentence 1\nSentence 2\nSentence 3\n...",
                        lines=10,
                        value=EXAMPLE_BATCH,
                    )
                    query_input = gr.Textbox(
                        label="Query Sentence",
                        placeholder="Enter your query...",
                        lines=2,
                    )
                    search_btn = gr.Button("Search", variant="primary")

                with gr.Column():
                    search_results = gr.Textbox(label="Ranked Results", lines=15)
                    search_plot = gr.Plot(label="Top Matches")

        with gr.Tab("Embedding Visualization"):
            gr.Markdown("### Visualize sentence embeddings in 2D space")

            with gr.Row():
                with gr.Column(scale=1):
                    viz_text = gr.Textbox(
                        label="Sentences (one per line)",
                        placeholder="Sentence 1\nSentence 2\nSentence 3\n...",
                        lines=15,
                        value=EXAMPLE_BATCH,
                    )
                    viz_method = gr.Radio(
                        choices=["t-SNE", "PCA"],
                        value="t-SNE",
                        label="Visualization Method",
                    )
                    viz_btn = gr.Button("Visualize", variant="primary")

                with gr.Column(scale=2):
                    viz_plot = gr.Plot(label="2D Embedding Space")

        with gr.Tab("Similarity Matrix"):
            gr.Markdown("### See similarity scores between all sentence pairs")

            with gr.Row():
                with gr.Column(scale=1):
                    matrix_text = gr.Textbox(
                        label="Sentences (one per line)",
                        placeholder="Sentence 1\nSentence 2\nSentence 3\n...",
                        lines=15,
                        value=EXAMPLE_BATCH,
                    )
                    matrix_btn = gr.Button("Generate Matrix", variant="primary")

                with gr.Column(scale=2):
                    matrix_plot = gr.Plot(label="Similarity Heatmap")

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Tool

                This tool uses the **sentence-transformers/all-MiniLM-L6-v2** model to generate
                semantic embeddings for sentences and compute their similarity.

                ### How It Works:

                1. **Embedding Generation**: Each sentence is converted into a 384-dimensional vector
                   that captures its semantic meaning.

                2. **Cosine Similarity**: We compute the cosine similarity between embeddings,
                   which ranges from -1 (opposite) to 1 (identical).

                3. **Dimensionality Reduction**: For visualization, we reduce embeddings to 2D
                   using t-SNE or PCA.

                ### Use Cases:

                - **Duplicate Detection**: Find similar or duplicate content
                - **Semantic Search**: Find documents similar to a query
                - **Clustering**: Group similar sentences together
                - **Paraphrase Detection**: Identify sentences with similar meaning
                - **Content Recommendation**: Recommend similar articles or products

                ### Similarity Score Guide:

                - **0.9 - 1.0**: Nearly identical meaning (paraphrases)
                - **0.7 - 0.9**: Highly related topics
                - **0.5 - 0.7**: Moderately related
                - **0.3 - 0.5**: Loosely related
                - **0.0 - 0.3**: Different topics

                ### Tips:

                - Longer, more detailed sentences often produce better embeddings
                - The model understands context and semantics, not just keywords
                - Try comparing sentences with similar meanings but different words
                """
            )

        # Connect components
        compare_btn.click(
            fn=lambda s1, s2: compare_two_sentences(s1, s2)[1:],
            inputs=[sent1_input, sent2_input],
            outputs=[similarity_output, similarity_plot],
        )

        search_btn.click(
            fn=batch_similarity,
            inputs=[batch_text, query_input],
            outputs=[search_results, search_plot],
        )

        viz_btn.click(
            fn=visualize_embeddings, inputs=[viz_text, viz_method], outputs=[viz_plot]
        )

        matrix_btn.click(
            fn=create_similarity_matrix, inputs=[matrix_text], outputs=[matrix_plot]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
