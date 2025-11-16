"""
Named Entity Recognition Visualizer - Interactive UI for NER

Features:
- Real-time entity extraction and highlighting
- Support for both spaCy and GLiNER models
- Entity type filtering
- Entity statistics and counts
- Export entities to CSV/JSON
- Multiple example texts
"""

import json
import logging
from collections import Counter
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import spacy
from gliner import GLiNER
from spacy import displacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache
nlp_model = None
gliner_model = None


def load_spacy_model():
    """Load Spacy NER model (lazy loading)"""
    global nlp_model
    if nlp_model is None:
        logger.info("Loading Spacy NER model: en_core_web_trf")
        try:
            nlp_model = spacy.load("en_core_web_trf")
            logger.info("Model loaded successfully")
        except OSError:
            logger.warning("en_core_web_trf not found, trying en_core_web_sm")
            try:
                nlp_model = spacy.load("en_core_web_sm")
                logger.info("Loaded en_core_web_sm model")
            except OSError:
                logger.error("No Spacy model found. Please install with: python -m spacy download en_core_web_sm")
                raise
    return nlp_model


def load_gliner_model(model_name="urchade/gliner_multi-v2.1"):
    """Load GLiNER model (lazy loading)"""
    global gliner_model
    if gliner_model is None:
        logger.info(f"Loading GLiNER model: {model_name}")
        try:
            gliner_model = GLiNER.from_pretrained(model_name)
            logger.info("GLiNER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise
    return gliner_model


def extract_entities_spacy(text: str, entity_types: List[str] = None) -> Tuple[str, List[Dict], Counter]:
    """
    Extract named entities using spaCy

    Args:
        text: Input text
        entity_types: List of entity types to display (None = all)

    Returns:
        Tuple of (HTML visualization, entities list, entity counts)
    """
    nlp = load_spacy_model()
    doc = nlp(text)

    # Filter entities if specific types requested
    if entity_types and len(entity_types) > 0:
        filtered_ents = [ent for ent in doc.ents if ent.label_ in entity_types]
        doc.ents = filtered_ents

    # Generate HTML visualization
    html = displacy.render(doc, style="ent", page=False)

    # Wrap HTML with styling
    styled_html = f"""
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
        {html}
    </div>
    """

    # Extract entity information
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "Text": ent.text,
                "Type": ent.label_,
                "Start": ent.start_char,
                "End": ent.end_char,
            }
        )

    # Count entity types
    entity_counts = Counter([ent["Type"] for ent in entities]) if entities else Counter()

    return styled_html, entities, entity_counts


def extract_entities_gliner(
    text: str, labels: List[str] = None, threshold: float = 0.5
) -> Tuple[str, List[Dict], Counter]:
    """
    Extract named entities using GLiNER

    Args:
        text: Input text
        labels: List of entity labels to extract
        threshold: Confidence threshold for entity extraction

    Returns:
        Tuple of (HTML visualization, entities list, entity counts)
    """
    if labels is None or len(labels) == 0:
        # Default entity types
        labels = [
            "person",
            "organization",
            "location",
            "date",
            "event",
            "product",
            "language",
            "country",
            "city",
            "money",
        ]

    model = load_gliner_model()
    predicted_entities = model.predict_entities(text, labels, threshold=threshold)

    # Create HTML visualization similar to spaCy's displacy
    html_parts = []
    last_end = 0

    for entity in sorted(predicted_entities, key=lambda x: x["start"]):
        start = entity["start"]
        end = entity["end"]
        label = entity["label"].upper()

        # Add text before entity
        if start > last_end:
            html_parts.append(text[last_end:start])

        # Add highlighted entity
        html_parts.append(
            f'<mark style="background-color: #ddd; padding: 0.25em 0.4em; margin: 0 0.25em; '
            f'line-height: 1; border-radius: 0.35em;">'
            f"{text[start:end]}"
            f'<span style="font-size: 0.8em; font-weight: bold; line-height: 1; '
            f"border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; "
            f'margin-left: 0.5rem">{label}</span>'
            f"</mark>"
        )

        last_end = end

    # Add remaining text
    if last_end < len(text):
        html_parts.append(text[last_end:])

    html_content = "".join(html_parts)
    styled_html = f"""
    <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;
                line-height: 2.5; direction: ltr">
        {html_content}
    </div>
    """

    # Extract entity information
    entities = []
    for ent in predicted_entities:
        entities.append(
            {
                "Text": ent["text"],
                "Type": ent["label"].upper(),
                "Start": ent["start"],
                "End": ent["end"],
                "Score": round(ent["score"], 3),
            }
        )

    # Count entity types
    entity_counts = Counter([ent["Type"] for ent in entities]) if entities else Counter()

    return styled_html, entities, entity_counts


def extract_entities(
    text: str,
    model_type: str = "spacy",
    entity_types: List[str] = None,
    custom_labels: str = "",
    threshold: float = 0.5,
) -> Tuple[str, str, object, str]:
    """
    Extract named entities from text using selected model

    Args:
        text: Input text
        model_type: Model to use ('spacy' or 'gliner')
        entity_types: List of entity types to display (for spaCy)
        custom_labels: Comma-separated custom labels (for GLiNER)
        threshold: Confidence threshold (for GLiNER)

    Returns:
        Tuple of (HTML visualization, entities table, statistics plot, JSON export)
    """
    if not text or not text.strip():
        return "Please provide some text to analyze", "", None, ""

    try:
        if model_type == "spacy":
            styled_html, entities, entity_counts = extract_entities_spacy(text, entity_types)
        else:  # gliner
            # Parse custom labels if provided
            if custom_labels and custom_labels.strip():
                labels = [label.strip() for label in custom_labels.split(",")]
            else:
                labels = None
            styled_html, entities, entity_counts = extract_entities_gliner(text, labels, threshold)

        # Create entities DataFrame
        if entities:
            df = pd.DataFrame(entities)
            entities_table = df.to_markdown(index=False)

            # Create bar chart
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(entity_counts.keys()),
                        y=list(entity_counts.values()),
                        marker=dict(
                            color=list(entity_counts.values()),
                            colorscale="Viridis",
                            showscale=False,
                        ),
                        text=list(entity_counts.values()),
                        textposition="auto",
                    )
                ]
            )

            fig.update_layout(
                title="Entity Type Distribution",
                xaxis_title="Entity Type",
                yaxis_title="Count",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )

            # Create JSON export
            json_export = json.dumps(entities, indent=2)

        else:
            entities_table = "No entities found in the text."
            fig = None
            json_export = "[]"

        return styled_html, entities_table, fig, json_export

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return f"Error: {str(e)}", "", None, ""


def get_entity_types():
    """Get list of available entity types from Spacy"""
    try:
        nlp = load_spacy_model()
        # Standard Spacy entity types
        entity_types = [
            "PERSON",  # People, including fictional
            "NORP",  # Nationalities or religious or political groups
            "FAC",  # Buildings, airports, highways, bridges, etc.
            "ORG",  # Companies, agencies, institutions, etc.
            "GPE",  # Countries, cities, states
            "LOC",  # Non-GPE locations, mountain ranges, bodies of water
            "PRODUCT",  # Objects, vehicles, foods, etc. (not services)
            "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
            "WORK_OF_ART",  # Titles of books, songs, etc.
            "LAW",  # Named documents made into laws
            "LANGUAGE",  # Any named language
            "DATE",  # Absolute or relative dates or periods
            "TIME",  # Times smaller than a day
            "PERCENT",  # Percentage, including "%"
            "MONEY",  # Monetary values, including unit
            "QUANTITY",  # Measurements, as of weight or distance
            "ORDINAL",  # "first", "second", etc.
            "CARDINAL",  # Numerals that do not fall under another type
        ]
        return entity_types
    except:
        return []


# Example texts
EXAMPLES = [
    [
        "West Germany (German: Westdeutschland) is the colloquial English term used to indicate "
        "the Federal Republic of Germany (FRG) between its formation on 23 May 1949 and the German "
        "reunification on 3 October 1990. The FRG's provisional capital was the city of Bonn.",
        "spacy",
        [],
        "",
        0.5,
    ],
    [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in "
        "Cupertino, California. The company's first product was the Apple I computer.",
        "gliner",
        [],
        "person, organization, location, date, product",
        0.5,
    ],
    [
        "The Eiffel Tower in Paris, France was built between 1887 and 1889 by Gustave Eiffel. "
        "It stands 330 meters tall and receives approximately 7 million visitors annually.",
        "spacy",
        [],
        "",
        0.5,
    ],
    [
        "On July 20, 1969, Neil Armstrong became the first human to walk on the Moon during NASA's "
        "Apollo 11 mission. He was accompanied by Buzz Aldrin, while Michael Collins orbited above.",
        "gliner",
        [],
        "person, organization, date, event, location",
        0.5,
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    entity_types = get_entity_types()

    with gr.Blocks(title="NER Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏷️ Named Entity Recognition Visualizer

            Extract and visualize named entities from text using state-of-the-art models.
            Choose between **spaCy** (traditional NER) or **GLiNER** (zero-shot NER).
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Paste or type your text here...",
                    lines=10,
                )

                model_selector = gr.Radio(
                    choices=["spacy", "gliner"],
                    value="spacy",
                    label="Select NER Model",
                    info="spaCy: Traditional NER | GLiNER: Zero-shot NER with custom labels",
                )

                with gr.Row():
                    with gr.Column():
                        entity_filter = gr.CheckboxGroup(
                            choices=entity_types,
                            label="Filter Entity Types (spaCy only)",
                            value=[],
                            visible=True,
                        )

                    with gr.Column():
                        custom_labels_input = gr.Textbox(
                            label="Custom Entity Labels (GLiNER only)",
                            placeholder="person, organization, location, date, event",
                            value="",
                            visible=False,
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold (GLiNER only)",
                            visible=False,
                        )

                analyze_btn = gr.Button("Extract Entities", variant="primary")

            with gr.Column(scale=2):
                stats_plot = gr.Plot(label="Entity Statistics")

        with gr.Row():
            html_output = gr.HTML(label="Visualized Entities")

        with gr.Tabs():
            with gr.Tab("Entity Table"):
                entities_table = gr.Markdown(label="Extracted Entities")

            with gr.Tab("JSON Export"):
                json_output = gr.Code(label="JSON Format (copy to export)", language="json", lines=15)

        with gr.Accordion("Examples - Click to load", open=False):
            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, model_selector, entity_filter, custom_labels_input, threshold_slider],
                label="Try these examples",
            )

        with gr.Accordion("Model Comparison", open=False):
            gr.Markdown(
                """
                ### spaCy vs GLiNER

                **spaCy:**
                - Traditional statistical NER model
                - Pre-trained on standard entity types (18 types)
                - Fast and accurate for common entities
                - Great for: News articles, general text, standard entities

                **GLiNER:**
                - Zero-shot NER using transformers
                - Custom entity labels - define any entity type!
                - More flexible but requires more compute
                - Great for: Domain-specific entities, custom categories, exploratory analysis

                ### Tips:
                - Start with spaCy for standard entity types
                - Use GLiNER when you need custom entity categories
                - Adjust GLiNER threshold if you get too many/few results
                - GLiNER labels should be lowercase and descriptive
                """
            )

        with gr.Accordion("Entity Type Guide", open=False):
            gr.Markdown(
                """
                ### spaCy Entity Types:

                - **PERSON**: People, including fictional characters
                - **NORP**: Nationalities, religious or political groups
                - **FAC**: Facilities (buildings, airports, highways, bridges)
                - **ORG**: Organizations (companies, agencies, institutions)
                - **GPE**: Geopolitical entities (countries, cities, states)
                - **LOC**: Non-GPE locations (mountain ranges, bodies of water)
                - **PRODUCT**: Objects, vehicles, foods (not services)
                - **EVENT**: Named events (hurricanes, battles, wars, sports events)
                - **WORK_OF_ART**: Titles of books, songs, movies, etc.
                - **LAW**: Named documents made into laws
                - **LANGUAGE**: Any named language
                - **DATE**: Absolute or relative dates or periods
                - **TIME**: Times smaller than a day
                - **PERCENT**: Percentage, including "%"
                - **MONEY**: Monetary values, including unit
                - **QUANTITY**: Measurements (weight, distance, etc.)
                - **ORDINAL**: "first", "second", etc.
                - **CARDINAL**: Numerals that don't fall under another type

                ### GLiNER Custom Labels:
                You can use any labels you want! Examples:
                - person, company, location, date
                - disease, symptom, medication, dosage
                - programming language, framework, library
                - cuisine, ingredient, restaurant, chef
                """
            )

        with gr.Accordion("About", open=False):
            gr.Markdown(
                """
                ## About This Tool

                This NER visualizer supports two state-of-the-art approaches:

                1. **Spacy** - Transformer-based traditional NER
                2. **GLiNER** - Zero-shot NER with custom entity labels

                ### Features:
                - Real-time entity extraction
                - Interactive entity highlighting
                - Model selection (spaCy/GLiNER)
                - Custom entity labels (GLiNER)
                - Entity type filtering (spaCy)
                - Statistics and visualization
                - JSON export for downstream processing

                ### Use Cases:
                - Information extraction from documents
                - Content analysis and categorization
                - Data mining and knowledge graph construction
                - Privacy detection (identifying personal information)
                - Domain-specific entity extraction
                """
            )

        # Model selector change handlers
        def update_visibility(model_type):
            if model_type == "spacy":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:  # gliner
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

        model_selector.change(
            fn=update_visibility,
            inputs=[model_selector],
            outputs=[entity_filter, custom_labels_input, threshold_slider],
        )

        # Connect components
        analyze_btn.click(
            fn=extract_entities,
            inputs=[text_input, model_selector, entity_filter, custom_labels_input, threshold_slider],
            outputs=[html_output, entities_table, stats_plot, json_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, show_error=True)
