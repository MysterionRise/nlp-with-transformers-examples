"""
Text Summarization Studio - Interactive UI for text summarization

Features:
- Multiple summarization models comparison
- Adjustable generation parameters
- Side-by-side summary comparison
- Summary statistics (compression ratio, readability)
- Example articles for testing
"""

import logging
import time
from typing import Dict, List, Tuple

import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available summarization models
SUMMARIZATION_MODELS = {
    "BART Large CNN": "facebook/bart-large-cnn",
    "T5 Large": "google/flan-t5-large",
    "T5 Base": "t5-base",
    "Pegasus XSum": "google/pegasus-xsum",
    "DistilBART CNN": "sshleifer/distilbart-cnn-12-6",
}

# Model cache
model_cache = {}


def load_model(model_name: str):
    """Load or retrieve cached summarization model"""
    if model_name not in model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            model_cache[model_name] = pipeline(
                "summarization",
                model=SUMMARIZATION_MODELS[model_name],
                tokenizer=SUMMARIZATION_MODELS[model_name],
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    return model_cache[model_name]


def calculate_stats(original: str, summary: str) -> Dict[str, str]:
    """Calculate summary statistics"""
    orig_words = len(original.split())
    summ_words = len(summary.split())
    orig_chars = len(original)
    summ_chars = len(summary)

    compression_ratio = (1 - summ_chars / orig_chars) * 100 if orig_chars > 0 else 0

    return {
        "Original Length": f"{orig_words} words, {orig_chars} characters",
        "Summary Length": f"{summ_words} words, {summ_chars} characters",
        "Compression Ratio": f"{compression_ratio:.1f}%",
        "Reduction": f"{orig_words - summ_words} words removed",
    }


def summarize_text(
    text: str,
    model_name: str,
    min_length: int,
    max_length: int,
    do_sample: bool = False,
) -> Tuple[str, Dict, float]:
    """
    Summarize text using specified model

    Args:
        text: Input text to summarize
        model_name: Name of the model to use
        min_length: Minimum summary length
        max_length: Maximum summary length
        do_sample: Whether to use sampling

    Returns:
        Tuple of (summary, statistics, generation time)
    """
    if not text or not text.strip():
        return "Please provide some text to summarize", {}, 0.0

    if len(text.split()) < 30:
        return (
            "Text too short. Please provide at least 30 words for meaningful summarization.",
            {},
            0.0,
        )

    try:
        # Load model
        summarizer = load_model(model_name)

        # Generate summary
        start_time = time.time()

        # Truncate input if too long
        max_input_length = 1024
        words = text.split()
        if len(words) > max_input_length:
            text = " ".join(words[:max_input_length])
            logger.warning(f"Input truncated to {max_input_length} words")

        summary = summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=do_sample,
            truncation=True,
        )[
            0
        ]["summary_text"]

        gen_time = time.time() - start_time

        # Calculate statistics
        stats = calculate_stats(text, summary)
        stats["Generation Time"] = f"{gen_time:.2f} seconds"

        return summary, stats, gen_time

    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return f"Error: {str(e)}", {}, 0.0


def compare_models(text: str, models: List[str], min_length: int, max_length: int) -> List[Tuple[str, str, Dict]]:
    """
    Compare multiple models on the same text

    Args:
        text: Input text
        models: List of model names
        min_length: Minimum summary length
        max_length: Maximum summary length

    Returns:
        List of (model_name, summary, stats) tuples
    """
    if not text or not text.strip():
        return [("Error", "Please provide some text to summarize", {})]

    if not models or len(models) == 0:
        return [("Error", "Please select at least one model", {})]

    results = []
    for model_name in models:
        summary, stats, _ = summarize_text(text, model_name, min_length, max_length)
        results.append((model_name, summary, stats))

    return results


def format_comparison_results(results: List[Tuple[str, str, Dict]]) -> str:
    """Format comparison results as markdown"""
    if not results:
        return "No results to display"

    markdown = "# Summary Comparison\n\n"

    for model_name, summary, stats in results:
        markdown += f"## {model_name}\n\n"
        markdown += f"**Summary:**\n{summary}\n\n"
        markdown += "**Statistics:**\n"
        for key, value in stats.items():
            markdown += f"- {key}: {value}\n"
        markdown += "\n---\n\n"

    return markdown


# Example articles
EXAMPLE_ARTICLE_1 = (
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
    "rainforest in the Amazon biome that covers most of the Amazon basin of South "
    "America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which "
    "5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. The majority of "
    "the forest is contained within Brazil, with 60% of the rainforest, followed by "
    "Peru with 13%, Colombia with 10%, and with minor amounts in Bolivia, Ecuador, "
    "French Guiana, Guyana, Suriname, and Venezuela.\n\n"
    "The Amazon represents over half of the planet's remaining rainforests, and "
    "comprises the largest and most biodiverse tract of tropical rainforest in the "
    "world, with an estimated 390 billion individual trees divided into 16,000 species. "
    "The region is home to about 2.5 million insect species, tens of thousands of "
    "plants, and some 2,000 birds and mammals. To date, at least 40,000 plant species, "
    "2,200 fishes, 1,294 birds, 427 mammals, 428 amphibians, and 378 reptiles have been "
    "scientifically classified in the region.\n\n"
    "The rainforest likely formed during the Eocene era, following the extinction event "
    "which wiped out the dinosaurs. It appeared following a global reduction of tropical "
    "temperatures when the Atlantic Ocean had widened sufficiently to provide a warm, "
    "moist climate to the Amazon basin. The rainforest has been in existence for at "
    "least 55 million years, and most of the region remained free of savanna-type biomes "
    "during glacial periods, allowing for the survival and evolution of a broad diversity "
    "of species."
)

EXAMPLE_ARTICLE_2 = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed "
    "to natural intelligence displayed by animals including humans. AI research has been "
    "defined as the field of study of intelligent agents, which refers to any system that "
    "perceives its environment and takes actions that maximize its chance of achieving "
    'its goals. The term "artificial intelligence" had previously been used to describe '
    'machines that mimic and display "human" cognitive skills that are associated with '
    'the human mind, such as "learning" and "problem-solving". This definition has '
    "since been rejected by major AI researchers who now describe AI in terms of "
    "rationality and acting rationally, which does not limit how intelligence can be "
    "articulated.\n\n"
    "AI applications include advanced web search engines, recommendation systems, "
    "understanding human speech, self-driving cars, generative or creative tools, "
    "automated decision-making and competing at the highest level in strategic game "
    "systems. As machines become increasingly capable, tasks considered to require "
    '"intelligence" are often removed from the definition of AI, a phenomenon known '
    "as the AI effect. For instance, optical character recognition is frequently excluded "
    "from things considered to be AI, having become a routine technology.\n\n"
    "Artificial intelligence was founded as an academic discipline in 1956, and in the "
    "years since it has experienced several waves of optimism, followed by disappointment "
    "and the loss of funding, followed by new approaches, success and renewed funding. "
    "AI research has tried and discarded many different approaches, including simulating "
    "the brain, modeling human problem solving, formal logic, large databases of knowledge "
    "and imitating animal behavior. In the first decades of the 21st century, highly "
    "mathematical and statistical machine learning has dominated the field, and this "
    "technique has proved highly successful, helping to solve many challenging problems "
    "throughout industry and academia."
)

EXAMPLE_ARTICLE_3 = (
    "Climate change refers to long-term shifts in temperatures and weather patterns. "
    "Such shifts can be natural, due to changes in the sun's activity or large volcanic "
    "eruptions. But since the 1800s, human activities have been the main driver of "
    "climate change, primarily due to the burning of fossil fuels like coal, oil, and "
    "gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket "
    "wrapped around the Earth, trapping the sun's heat and raising temperatures.\n\n"
    "The main greenhouse gases that are causing climate change include carbon dioxide and "
    "methane. These come from using gasoline for driving a car or coal for heating a "
    "building, for example. Clearing land and cutting down forests can also release "
    "carbon dioxide. Agriculture, oil and gas operations are major sources of methane "
    "emissions. Energy, industry, transport, buildings, agriculture and land use are "
    "among the main sectors causing greenhouse gases.\n\n"
    "The consequences of climate change now include intense droughts, water scarcity, "
    "severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms "
    "and declining biodiversity. People are experiencing climate change in diverse ways. "
    "It affects our health, ability to grow food, housing, safety and work. Some of us "
    "are already more vulnerable to climate impacts, such as people living in small "
    "island nations and other developing countries. Conditions like sea-level rise and "
    "saltwater intrusion have advanced to the point where whole communities have had to "
    "relocate, and protracted droughts are putting people at risk of famine. In the "
    "future, the number of climate refugees is expected to rise."
)


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Text Summarization Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # 📝 Text Summarization Studio

            Generate concise summaries from long-form text using state-of-the-art transformer models.
            Compare different models and fine-tune parameters for optimal results!
            """)

        with gr.Tab("Single Model"):
            gr.Markdown("### Summarize text with a single model")

            with gr.Row():
                with gr.Column(scale=2):
                    single_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your article or text here...",
                        lines=15,
                        value=EXAMPLE_ARTICLE_1,
                    )

                    with gr.Row():
                        single_model = gr.Dropdown(
                            choices=list(SUMMARIZATION_MODELS.keys()),
                            value="BART Large CNN",
                            label="Select Model",
                        )

                    with gr.Row():
                        min_length = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Minimum Length (words)",
                        )
                        max_length = gr.Slider(
                            minimum=50,
                            maximum=300,
                            value=130,
                            step=10,
                            label="Maximum Length (words)",
                        )

                    single_btn = gr.Button("Generate Summary", variant="primary")

                with gr.Column(scale=2):
                    summary_output = gr.Textbox(label="Generated Summary", lines=10)
                    stats_output = gr.JSON(label="Statistics")

        with gr.Tab("Model Comparison"):
            gr.Markdown("### Compare summaries from multiple models side-by-side")

            with gr.Row():
                with gr.Column(scale=1):
                    compare_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your article or text here...",
                        lines=15,
                        value=EXAMPLE_ARTICLE_2,
                    )

                    compare_models_select = gr.CheckboxGroup(
                        choices=list(SUMMARIZATION_MODELS.keys()),
                        value=["BART Large CNN", "T5 Large"],
                        label="Select Models to Compare",
                    )

                    with gr.Row():
                        compare_min = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Min Length",
                        )
                        compare_max = gr.Slider(
                            minimum=50,
                            maximum=300,
                            value=130,
                            step=10,
                            label="Max Length",
                        )

                    compare_btn = gr.Button("Compare Models", variant="primary")

                with gr.Column(scale=2):
                    comparison_output = gr.Markdown(label="Comparison Results")

        with gr.Tab("Example Articles"):
            gr.Markdown("### Try these example articles")

            with gr.Tabs():
                with gr.Tab("Amazon Rainforest"):
                    gr.Textbox(value=EXAMPLE_ARTICLE_1, lines=20, interactive=False)
                    copy_btn1 = gr.Button("Use This Article")

                with gr.Tab("Artificial Intelligence"):
                    gr.Textbox(value=EXAMPLE_ARTICLE_2, lines=20, interactive=False)
                    copy_btn2 = gr.Button("Use This Article")

                with gr.Tab("Climate Change"):
                    gr.Textbox(value=EXAMPLE_ARTICLE_3, lines=20, interactive=False)
                    copy_btn3 = gr.Button("Use This Article")

        with gr.Tab("About"):
            gr.Markdown("""
                ## About Text Summarization Studio

                This tool provides state-of-the-art text summarization using transformer-based models.

                ### Available Models:

                1. **BART Large CNN** - Trained on CNN/DailyMail dataset, excellent for news articles
                2. **T5 Large** - Versatile model from Google, good for various text types
                3. **T5 Base** - Faster, lighter version of T5
                4. **Pegasus XSum** - Optimized for extreme summarization
                5. **DistilBART CNN** - Distilled version of BART, faster inference

                ### Features:

                - **Adjustable Parameters**: Control summary length with min/max length sliders
                - **Model Comparison**: Compare multiple models side-by-side
                - **Statistics**: Track compression ratio, word count, generation time
                - **Example Articles**: Pre-loaded articles for quick testing

                ### Tips:

                - **Minimum Input**: Provide at least 30 words for meaningful summarization
                - **Optimal Length**: Articles with 200-1000 words work best
                - **Model Selection**:
                  - Use BART for news articles
                  - Use T5 for scientific or technical text
                  - Use Pegasus for very short summaries
                - **Length Parameters**:
                  - Min length: Ensures summaries aren't too brief
                  - Max length: Controls maximum summary size
                  - Typical ratio: 15-30% of original length

                ### Use Cases:

                - News article summarization
                - Research paper abstracts
                - Document summarization
                - Content curation
                - Information extraction
                - Quick reading/scanning
                """)

        # Connect components - Single Model
        single_btn.click(
            fn=lambda t, m, min_l, max_l: summarize_text(t, m, min_l, max_l)[:2],
            inputs=[single_text, single_model, min_length, max_length],
            outputs=[summary_output, stats_output],
        )

        # Connect components - Model Comparison
        compare_btn.click(
            fn=lambda t, m, min_l, max_l: format_comparison_results(compare_models(t, m, min_l, max_l)),
            inputs=[compare_text, compare_models_select, compare_min, compare_max],
            outputs=[comparison_output],
        )

        # Copy example articles
        copy_btn1.click(fn=lambda: EXAMPLE_ARTICLE_1, outputs=[single_text])
        copy_btn2.click(fn=lambda: EXAMPLE_ARTICLE_2, outputs=[single_text])
        copy_btn3.click(fn=lambda: EXAMPLE_ARTICLE_3, outputs=[single_text])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False, show_error=True)
