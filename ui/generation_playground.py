"""
Text Generation Playground - Interactive UI for text generation

Features:
- Real-time text generation from prompts
- Multiple model comparison
- Parameter control (length, temperature, top-p)
- Batch processing support
- Example prompts for quick testing
"""

import logging
from typing import Dict, List

import gradio as gr
import plotly.graph_objects as go
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available text generation models
GENERATION_MODELS = {
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
    "DistilGPT-2": "distilgpt2",
}

# Cache for loaded models
model_cache = {}


def load_model(model_name: str):
    """Load or retrieve cached text generation model"""
    if model_name not in model_cache:
        logger.info(f"Loading model: {model_name}")
        try:
            model_cache[model_name] = pipeline(
                "text-generation",
                model=GENERATION_MODELS[model_name],
                tokenizer=GENERATION_MODELS[model_name],
            )
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    return model_cache[model_name]


def generate_text(
    prompt: str,
    model_name: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    seed: int = None,
) -> Dict[str, any]:
    """
    Generate text from a prompt

    Args:
        prompt: Input prompt
        model_name: Name of the model to use
        max_length: Maximum length of generated text
        temperature: Sampling temperature (controls randomness)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_return_sequences: Number of sequences to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with generated texts and metadata
    """
    if not prompt or not prompt.strip():
        return {"error": "Please provide a prompt"}

    try:
        # Load model
        generator = load_model(model_name)

        # Set seed for reproducibility if provided
        if seed is not None:
            import torch

            torch.manual_seed(seed)

        # Generate text
        outputs = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=min(num_return_sequences, 5),  # Limit to 5 for performance
            do_sample=True,
            pad_token_id=50256,  # GPT2 EOS token
        )

        # Format results
        generated_texts = []
        for i, output in enumerate(outputs, 1):
            generated_texts.append(output["generated_text"])

        formatted_results = {
            "Prompt": prompt,
            "Model": model_name,
            "Parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "sequences": len(outputs),
            },
            "Samples": generated_texts,
        }

        return formatted_results

    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return {"error": f"Error: {str(e)}"}


def compare_models(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = None,
) -> Dict[str, str]:
    """
    Compare text generation across all models

    Args:
        prompt: Input prompt
        max_length: Maximum length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        seed: Random seed

    Returns:
        Dictionary with comparisons
    """
    if not prompt or not prompt.strip():
        return {"error": "Please provide a prompt"}

    try:
        results = {}

        for model_name in GENERATION_MODELS.keys():
            try:
                generator = load_model(model_name)

                if seed is not None:
                    import torch

                    torch.manual_seed(seed)

                output = generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=50,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=50256,
                )

                results[model_name] = output[0]["generated_text"]
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = f"Error: {str(e)}"

        return results

    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        return {"error": f"Error: {str(e)}"}


def batch_generate(
    batch_prompts: str,
    model_name: str,
    max_length: int = 100,
    temperature: float = 0.7,
) -> str:
    """
    Generate text for multiple prompts

    Args:
        batch_prompts: Multiple prompts separated by newlines
        model_name: Model to use
        max_length: Maximum length
        temperature: Sampling temperature

    Returns:
        Formatted results as string
    """
    if not batch_prompts or not batch_prompts.strip():
        return "Please provide prompts"

    try:
        generator = load_model(model_name)

        # Split prompts
        prompts = [p.strip() for p in batch_prompts.split("\n") if p.strip()]

        if not prompts:
            return "No valid prompts found"

        results = []
        for i, prompt in enumerate(prompts, 1):
            try:
                output = generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=50256,
                )

                generated = output[0]["generated_text"]
                results.append(
                    f"{i}. Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}\n"
                    f"   Generated: {generated[:100]}{'...' if len(generated) > 100 else ''}\n"
                )
            except Exception as e:
                results.append(f"{i}. Error: {str(e)}\n")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        return f"Error: {str(e)}"


# Example prompts for testing
EXAMPLES = [
    [
        "The future of artificial intelligence",
        "GPT-2",
        100,
        0.7,
    ],
    [
        "Once upon a time, in a galaxy far away",
        "GPT-2",
        100,
        0.8,
    ],
    [
        "Scientists discovered a new species",
        "GPT-2 Medium",
        100,
        0.6,
    ],
    [
        "The weather today is",
        "DistilGPT-2",
        100,
        0.5,
    ],
]


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Text Generation Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # 📝 Text Generation Playground

            Generate creative text completions using state-of-the-art language models.
            Experiment with different prompts, models, and parameters!
            """)

        with gr.Tab("Single Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="Start typing and see the model complete your text...",
                        lines=5,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(GENERATION_MODELS.keys()),
                        value="GPT-2",
                        label="Select Model",
                    )

                with gr.Column(scale=1):
                    max_length_slider = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Max Length",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more focused, Higher = more random",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus)",
                    )
                    top_k_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-k",
                    )
                    seed_input = gr.Number(
                        label="Seed (for reproducibility)",
                        value=42,
                        precision=0,
                    )
                    generate_btn = gr.Button("Generate", variant="primary")

            output_json = gr.JSON(label="Generation Results")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt_input, model_dropdown, max_length_slider, temperature_slider],
                label="Try these examples",
            )

        with gr.Tab("Model Comparison"):
            gr.Markdown("""
                ### Compare all models on the same prompt
                Generate text with different models to see how they differ in style and quality.
                """)

            with gr.Row():
                with gr.Column(scale=2):
                    compare_prompt = gr.Textbox(
                        label="Enter prompt",
                        placeholder="Your prompt here...",
                        lines=5,
                    )

                with gr.Column(scale=1):
                    compare_max_length = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Max Length",
                    )
                    compare_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    compare_top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p",
                    )
                    compare_seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                    )
                    compare_btn = gr.Button("Compare Models", variant="primary")

            compare_output = gr.JSON(label="Model Comparison")

        with gr.Tab("Batch Generation"):
            gr.Markdown("""
                ### Generate text for multiple prompts
                Enter multiple prompts (one per line) and generate continuations for all of them.
                """)

            with gr.Row():
                with gr.Column():
                    batch_prompts = gr.Textbox(
                        label="Enter prompts (one per line)",
                        placeholder="Prompt 1\nPrompt 2\nPrompt 3\n...",
                        lines=10,
                    )

                with gr.Column():
                    batch_model = gr.Dropdown(
                        choices=list(GENERATION_MODELS.keys()),
                        value="GPT-2",
                        label="Select Model",
                    )
                    batch_max_length = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Max Length",
                    )
                    batch_temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    batch_btn = gr.Button("Generate Batch", variant="primary")

            batch_output = gr.Textbox(label="Batch Results", lines=15, max_lines=20)

        with gr.Tab("About"):
            gr.Markdown("""
                ## About This Tool

                This text generation playground uses GPT-2 based models to generate creative text completions.

                ### Available Models:

                1. **GPT-2** - Fast baseline model
                2. **GPT-2 Medium** - Better quality, slower inference
                3. **DistilGPT-2** - Distilled version, fastest

                ### Parameters:

                **Temperature:** Controls randomness
                - 0.1 = Very focused, repetitive
                - 0.7 = Balanced (default)
                - 2.0 = Very random, chaotic

                **Top-p (Nucleus):** Only consider tokens with cumulative probability ≤ p
                - Lower values = more focused
                - Higher values = more diverse

                **Top-k:** Only consider top-k most likely next tokens
                - Controls diversity

                **Seed:** Use same seed for reproducible results

                ### Tips:

                - Start with descriptive prompts for better results
                - Adjust temperature based on desired creativity
                - Use lower temperature for technical writing
                - Use higher temperature for creative writing
                - Compare models to see quality differences
                - Batch processing is useful for testing multiple prompts

                ### Limitations:

                - Generated text quality depends on prompt quality
                - Models can produce nonsensical text
                - May be biased based on training data
                - Cannot follow complex instructions
                """)

        # Connect the components
        generate_btn.click(
            fn=generate_text,
            inputs=[
                prompt_input,
                model_dropdown,
                max_length_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
            ],
            outputs=[output_json],
        )

        compare_btn.click(
            fn=compare_models,
            inputs=[
                compare_prompt,
                compare_max_length,
                compare_temperature,
                compare_top_p,
                compare_seed,
            ],
            outputs=[compare_output],
        )

        batch_btn.click(
            fn=batch_generate,
            inputs=[batch_prompts, batch_model, batch_max_length, batch_temperature],
            outputs=[batch_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False, show_error=True)
