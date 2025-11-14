"""
Named Entity Recognition Visualizer - Interactive UI for NER

Features:
- Real-time entity extraction and highlighting
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
from spacy import displacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model cache
nlp_model = None


def load_model():
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


def extract_entities(text: str, entity_types: List[str] = None) -> Tuple[str, str, object, str]:
    """
    Extract named entities from text

    Args:
        text: Input text
        entity_types: List of entity types to display (None = all)

    Returns:
        Tuple of (HTML visualization, entities table, statistics plot, JSON export)
    """
    if not text or not text.strip():
        return "Please provide some text to analyze", "", None, ""

    try:
        # Load model and process text
        nlp = load_model()
        doc = nlp(text)

        # Filter entities if specific types requested
        if entity_types and len(entity_types) > 0:
            # Filter document entities
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

        # Create entities DataFrame
        if entities:
            df = pd.DataFrame(entities)
            entities_table = df.to_markdown(index=False)

            # Count entity types
            entity_counts = Counter([ent["Type"] for ent in entities])

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
        nlp = load_model()
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
        [],
    ],
    [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in "
        "Cupertino, California. The company's first product was the Apple I computer.",
        [],
    ],
    [
        "The Eiffel Tower in Paris, France was built between 1887 and 1889 by Gustave Eiffel. "
        "It stands 330 meters tall and receives approximately 7 million visitors annually.",
        [],
    ],
    [
        "On July 20, 1969, Neil Armstrong became the first human to walk on the Moon during NASA's "
        "Apollo 11 mission. He was accompanied by Buzz Aldrin, while Michael Collins orbited above.",
        [],
    ],
    [
        "Amazon.com, Inc. is an American multinational technology company based in Seattle, Washington. "
        "It was founded by Jeff Bezos on July 5, 1994, and has become one of the world's most valuable companies.",
        [],
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    entity_types = get_entity_types()

    with gr.Blocks(title="NER Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏷️ Named Entity Recognition Visualizer

            Extract and visualize named entities from text using state-of-the-art Spacy models.
            Identify people, organizations, locations, dates, and more!
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Paste or type your text here...",
                    lines=10,
                )

                with gr.Row():
                    entity_filter = gr.CheckboxGroup(
                        choices=entity_types,
                        label="Filter Entity Types (leave empty for all)",
                        value=[],
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
                inputs=[text_input, entity_filter],
                label="Try these examples",
            )

        with gr.Accordion("Entity Type Guide", open=False):
            gr.Markdown(
                """
                ### Entity Types Explained:

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

                ### Tips:
                - Use entity type filters to focus on specific categories
                - Export to JSON for further processing
                - Longer, well-written texts typically yield better results
                """
            )

        with gr.Accordion("About", open=False):
            gr.Markdown(
                """
                ## About This Tool

                This NER visualizer uses **Spacy** with transformer-based models to identify
                and classify named entities in text.

                ### Features:
                - Real-time entity extraction
                - Interactive entity highlighting
                - Entity type filtering
                - Statistics and visualization
                - JSON export for downstream processing

                ### Use Cases:
                - Information extraction from documents
                - Content analysis and categorization
                - Data mining and knowledge graph construction
                - Privacy detection (identifying personal information)
                - News article analysis
                """
            )

        # Connect components
        analyze_btn.click(
            fn=extract_entities,
            inputs=[text_input, entity_filter],
            outputs=[html_output, entities_table, stats_plot, json_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, show_error=True)
