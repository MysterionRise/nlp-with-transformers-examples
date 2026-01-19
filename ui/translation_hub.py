"""
Translation Hub - Interactive UI for multi-language translation

Features:
- Multi-language translation support
- Multiple model options (mBART, Helsinki OPUS)
- Real-time translation
- Batch processing support
- Language detection
"""

import logging
from typing import Dict, List, Tuple

import gradio as gr
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language codes and names for mBART
MBARTLANGUAGES = {
    "English": "en_XX",
    "Spanish": "es_XX",
    "French": "fr_XX",
    "German": "de_DE",
    "Italian": "it_IT",
    "Portuguese": "pt_XX",
    "Russian": "ru_RU",
    "Chinese": "zh_CN",
    "Japanese": "ja_XX",
    "Korean": "ko_KR",
    "Arabic": "ar_AR",
    "Hindi": "hi_IN",
}

# Language pairs for Helsinki OPUS
HELSINKI_LANGUAGE_PAIRS = [
    "en-es",
    "en-fr",
    "en-de",
    "es-en",
    "fr-en",
    "de-en",
    "en-pt",
    "pt-en",
    "en-it",
    "it-en",
]

# Translation models
TRANSLATION_MODELS = {
    "mBART-50 (50+ languages)": "mbart",
    "Helsinki OPUS (specific pairs)": "helsinki",
}

# Cache for loaded models
model_cache = {}


def load_mbart_model():
    """Load mBART translation model"""
    if "mbart" not in model_cache:
        logger.info("Loading mBART model...")
        try:
            model_cache["mbart"] = pipeline(
                "translation",
                model="facebook/mbart-large-50-many-to-many-mmt",
            )
            logger.info("mBART model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading mBART: {e}")
            raise
    return model_cache["mbart"]


def load_helsinki_model(lang_pair: str):
    """Load Helsinki OPUS translation model for specific language pair"""
    if lang_pair not in model_cache:
        logger.info(f"Loading Helsinki OPUS model for {lang_pair}...")
        try:
            model_id = f"Helsinki-NLP/opus-mt-{lang_pair}"
            model_cache[lang_pair] = pipeline(
                "translation",
                model=model_id,
            )
            logger.info(f"Helsinki model {lang_pair} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Helsinki model for {lang_pair}: {e}")
            raise
    return model_cache[lang_pair]


def translate_mbart(text: str, source_lang: str, target_lang: str) -> Dict:
    """
    Translate using mBART model

    Args:
        text: Text to translate
        source_lang: Source language name
        target_lang: Target language name

    Returns:
        Dictionary with translation results
    """
    if not text or not text.strip():
        return {"error": "Please provide text to translate"}

    try:
        translator = load_mbart_model()

        # Get language codes
        src_code = MBARTLANGUAGES.get(source_lang)
        tgt_code = MBARTLANGUAGES.get(target_lang)

        if not src_code or not tgt_code:
            return {"error": f"Language pair not supported"}

        # Set source language
        translator.tokenizer.src_lang = src_code

        # Translate
        result = translator(text, target_lang=tgt_code, max_length=400)

        return {
            "Original": text[:500] + ("..." if len(text) > 500 else ""),
            "Translation": result[0]["translation_text"],
            "Source Language": source_lang,
            "Target Language": target_lang,
            "Model": "mBART-50",
            "Input Length": len(text),
        }

    except Exception as e:
        logger.error(f"Error translating with mBART: {e}")
        return {"error": f"Error: {str(e)}"}


def translate_helsinki(text: str, lang_pair: str) -> Dict:
    """
    Translate using Helsinki OPUS model

    Args:
        text: Text to translate
        lang_pair: Language pair (e.g., 'en-es')

    Returns:
        Dictionary with translation results
    """
    if not text or not text.strip():
        return {"error": "Please provide text to translate"}

    try:
        translator = load_helsinki_model(lang_pair)

        # Translate
        result = translator(text, max_length=400)

        src_lang, tgt_lang = lang_pair.split("-")

        return {
            "Original": text[:500] + ("..." if len(text) > 500 else ""),
            "Translation": result[0]["translation_text"],
            "Source Language": src_lang.upper(),
            "Target Language": tgt_lang.upper(),
            "Model": "Helsinki OPUS",
            "Language Pair": lang_pair,
            "Input Length": len(text),
        }

    except Exception as e:
        logger.error(f"Error translating with Helsinki: {e}")
        return {"error": f"Error: {str(e)}"}


def batch_translate(
    batch_text: str,
    model_choice: str,
    source_lang: str = None,
    target_lang: str = None,
    lang_pair: str = None,
) -> str:
    """
    Translate multiple texts

    Args:
        batch_text: Multiple texts separated by newlines
        model_choice: Which model to use
        source_lang: Source language for mBART
        target_lang: Target language for mBART
        lang_pair: Language pair for Helsinki

    Returns:
        Formatted results as string
    """
    if not batch_text or not batch_text.strip():
        return "Please provide texts to translate"

    try:
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]

        if not texts:
            return "No valid texts found"

        results = []

        if model_choice == "mbart":
            if not source_lang or not target_lang:
                return "Please select source and target languages"

            translator = load_mbart_model()
            src_code = MBARTLANGUAGES.get(source_lang)
            tgt_code = MBARTLANGUAGES.get(target_lang)
            translator.tokenizer.src_lang = src_code

            for i, text in enumerate(texts, 1):
                try:
                    result = translator(text, target_lang=tgt_code, max_length=400)
                    results.append(
                        f"{i}. Original: {text[:60]}{'...' if len(text) > 60 else ''}\n"
                        f"   Translation: {result[0]['translation_text'][:80]}{'...' if len(result[0]['translation_text']) > 80 else ''}\n"
                    )
                except Exception as e:
                    results.append(f"{i}. Error: {str(e)}\n")

        else:  # helsinki
            if not lang_pair:
                return "Please select a language pair"

            translator = load_helsinki_model(lang_pair)

            for i, text in enumerate(texts, 1):
                try:
                    result = translator(text, max_length=400)
                    results.append(
                        f"{i}. Original: {text[:60]}{'...' if len(text) > 60 else ''}\n"
                        f"   Translation: {result[0]['translation_text'][:80]}{'...' if len(result[0]['translation_text']) > 80 else ''}\n"
                    )
                except Exception as e:
                    results.append(f"{i}. Error: {str(e)}\n")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch translation: {e}")
        return f"Error: {str(e)}"


# Example texts for testing
EXAMPLES = [
    [
        "Hello, how are you today? I hope you're having a great day!",
        "English",
        "Spanish",
    ],
    [
        "The quick brown fox jumps over the lazy dog.",
        "English",
        "French",
    ],
    [
        "Machine learning is a subset of artificial intelligence.",
        "English",
        "German",
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Translation Hub", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌍 Translation Hub

            Translate text between multiple languages using state-of-the-art neural translation models.
            Choose between multilingual mBART for 50+ languages or Helsinki OPUS for specific language pairs.
            """
        )

        with gr.Tab("Translate"):
            model_choice = gr.Radio(
                choices=list(TRANSLATION_MODELS.keys()),
                value="mBART-50 (50+ languages)",
                label="Select Translation Model",
            )

            # mBART interface
            with gr.Group(visible=True) as mbart_group:
                gr.Markdown("### mBART-50: Multilingual Translation (50+ languages)")

                with gr.Row():
                    with gr.Column():
                        mbart_text = gr.Textbox(
                            label="Text to translate",
                            placeholder="Enter text here...",
                            lines=5,
                        )
                    with gr.Column():
                        mbart_source = gr.Dropdown(
                            choices=list(MBARTLANGUAGES.keys()),
                            value="English",
                            label="Source Language",
                        )
                        mbart_target = gr.Dropdown(
                            choices=list(MBARTLANGUAGES.keys()),
                            value="Spanish",
                            label="Target Language",
                        )
                        mbart_btn = gr.Button("Translate", variant="primary")

                mbart_output = gr.JSON(label="Translation Result")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[mbart_text, mbart_source, mbart_target],
                    label="Try these examples",
                )

            # Helsinki interface
            with gr.Group(visible=False) as helsinki_group:
                gr.Markdown("### Helsinki OPUS: Specialized Language Pairs")

                with gr.Row():
                    with gr.Column():
                        helsinki_text = gr.Textbox(
                            label="Text to translate",
                            placeholder="Enter text here...",
                            lines=5,
                        )
                    with gr.Column():
                        helsinki_pair = gr.Dropdown(
                            choices=HELSINKI_LANGUAGE_PAIRS,
                            value="en-es",
                            label="Language Pair",
                        )
                        helsinki_btn = gr.Button("Translate", variant="primary")

                helsinki_output = gr.JSON(label="Translation Result")

            # Handle model switching
            def update_visibility(model):
                if "mBART" in model:
                    return gr.Group(visible=True), gr.Group(visible=False)
                else:
                    return gr.Group(visible=False), gr.Group(visible=True)

            model_choice.change(update_visibility, model_choice, [mbart_group, helsinki_group])

        with gr.Tab("Batch Translation"):
            gr.Markdown(
                """
                ### Translate multiple texts at once
                Enter multiple texts (one per line) to translate all of them.
                """
            )

            with gr.Row():
                with gr.Column():
                    batch_texts = gr.Textbox(
                        label="Texts to translate (one per line)",
                        placeholder="Text 1\nText 2\nText 3\n...",
                        lines=10,
                    )

                with gr.Column():
                    batch_model_choice = gr.Radio(
                        choices=list(TRANSLATION_MODELS.keys()),
                        value="mBART-50 (50+ languages)",
                        label="Select Model",
                    )

                    batch_mbart_source = gr.Dropdown(
                        choices=list(MBARTLANGUAGES.keys()),
                        value="English",
                        label="Source Language",
                        visible=True,
                    )
                    batch_mbart_target = gr.Dropdown(
                        choices=list(MBARTLANGUAGES.keys()),
                        value="Spanish",
                        label="Target Language",
                        visible=True,
                    )

                    batch_helsinki_pair = gr.Dropdown(
                        choices=HELSINKI_LANGUAGE_PAIRS,
                        value="en-es",
                        label="Language Pair",
                        visible=False,
                    )

                    batch_btn = gr.Button("Translate All", variant="primary")

            batch_output = gr.Textbox(label="Batch Results", lines=15, max_lines=20)

            # Handle batch model switching
            def update_batch_visibility(model):
                if "mBART" in model:
                    return (
                        gr.Dropdown(visible=True),
                        gr.Dropdown(visible=True),
                        gr.Dropdown(visible=False),
                    )
                else:
                    return (
                        gr.Dropdown(visible=False),
                        gr.Dropdown(visible=False),
                        gr.Dropdown(visible=True),
                    )

            batch_model_choice.change(
                update_batch_visibility,
                batch_model_choice,
                [batch_mbart_source, batch_mbart_target, batch_helsinki_pair],
            )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Tool

                This translation hub provides access to state-of-the-art neural machine translation models.

                ### Available Models:

                **1. mBART-50 (Multilingual BART)**
                - Supports 50+ languages
                - Can translate between any supported language pair
                - More flexible but may sacrifice some quality
                - Good for rare language pairs

                **2. Helsinki OPUS (Open Parallel Corpus)**
                - Specialized models trained for specific language pairs
                - Typically higher quality for supported pairs
                - Limited to predefined language pairs
                - Better for common language pairs (EN-ES, EN-FR, etc.)

                ### Supported Languages:

                **mBART-50:**
                - English, Spanish, French, German, Italian, Portuguese
                - Russian, Chinese, Japanese, Korean, Arabic, Hindi
                - And 38+ more languages

                **Helsinki OPUS:**
                - en-es, en-fr, en-de, es-en, fr-en, de-en
                - en-pt, pt-en, en-it, it-en

                ### How to Use:

                1. **Choose a model** based on your language pair and needs
                2. **Enter text** in the source language
                3. **Select languages** (source and target)
                4. **Click Translate** to get the translation
                5. **Use Batch mode** for translating multiple texts

                ### Tips:

                - For best results, use mBART for distant language pairs
                - Use Helsinki for common language pairs if available
                - Keep texts under 500 characters for faster processing
                - Batch mode is useful for translating many documents

                ### Limitations:

                - Neural translation can make semantic mistakes
                - Proper nouns and technical terms may not translate well
                - Very long texts may lose context
                - Quality varies by language pair

                ### Common Use Cases:

                - Business communication across languages
                - Translating documentation
                - Website localization
                - Multilingual content analysis
                """
            )

        # Connect mBART components
        mbart_btn.click(
            fn=translate_mbart,
            inputs=[mbart_text, mbart_source, mbart_target],
            outputs=[mbart_output],
        )

        # Connect Helsinki components
        helsinki_btn.click(
            fn=translate_helsinki,
            inputs=[helsinki_text, helsinki_pair],
            outputs=[helsinki_output],
        )

        # Connect batch components
        batch_btn.click(
            fn=batch_translate,
            inputs=[batch_texts, batch_model_choice, batch_mbart_source, batch_mbart_target, batch_helsinki_pair],
            outputs=[batch_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7868, share=False, show_error=True)
