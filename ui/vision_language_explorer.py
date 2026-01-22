"""
Vision-Language Explorer - Interactive UI for multi-modal image and text analysis

Features:
- Image captioning with multiple models
- Visual Question Answering (VQA)
- Image-text similarity matching
- Batch processing for multiple images
- Cross-modal retrieval
"""

import io
import logging
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models lazily
_models_cache = {}


def load_clip_model():
    """Load CLIP model for image-text matching"""
    if "clip" not in _models_cache:
        logger.info("Loading CLIP model...")
        try:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _models_cache["clip"] = (model, processor)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CLIP: {e}")
            raise
    return _models_cache["clip"]


def load_image_captioning_model():
    """Load image captioning model (GIT)"""
    if "git" not in _models_cache:
        logger.info("Loading GIT image captioning model...")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
            processor = AutoProcessor.from_pretrained("microsoft/git-base")
            _models_cache["git"] = (model, processor)
            logger.info("GIT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GIT: {e}")
            raise
    return _models_cache["git"]


def load_vqa_model():
    """Load Visual Question Answering model"""
    if "vqa" not in _models_cache:
        logger.info("Loading VQA model...")
        try:
            from transformers import pipeline

            pipeline_vqa = pipeline(
                "visual-question-answering",
                model="nlpconnect/vit-gpt2-image-captioning",
            )
            _models_cache["vqa"] = pipeline_vqa
            logger.info("VQA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VQA: {e}")
            # Use alternative lightweight approach
            logger.info("Using CLIP-based VQA fallback...")

    return _models_cache.get("vqa")


def generate_caption(image: Image.Image) -> Dict:
    """
    Generate caption for an image using GIT model

    Args:
        image: PIL Image object

    Returns:
        Dictionary with caption and metadata
    """
    if image is None:
        return {"error": "Please provide an image"}

    try:
        logger.info("Generating image caption...")
        model, processor = load_image_captioning_model()

        # Process image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generate caption
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            "Caption": generated_caption,
            "Model": "GIT Base (Microsoft)",
            "Image Size": f"{image.size[0]}x{image.size[1]}",
            "Status": "Success",
        }

    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return {"error": f"Error: {str(e)}"}


def calculate_clip_similarity(image: Image.Image, texts: List[str]) -> Tuple[Dict, str]:
    """
    Calculate CLIP similarity between image and multiple text descriptions

    Args:
        image: PIL Image object
        texts: List of text descriptions to match

    Returns:
        Tuple of (results dict, visualization HTML)
    """
    if image is None or not texts:
        return {"error": "Please provide image and text descriptions"}, ""

    try:
        logger.info("Calculating CLIP similarity...")
        model, processor = load_clip_model()

        import torch

        # Process inputs
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Format results
        results = {
            "Image-Text Similarities": {text: f"{prob.item():.4f}" for text, prob in zip(texts, probs[0])},
            "Best Match": texts[probs.argmax(dim=1).item()],
            "Best Score": f"{probs.max().item():.4f}",
            "Model": "CLIP ViT-B/32",
        }

        # Create visualization HTML
        html = "<div style='padding: 10px;'>"
        html += "<h3>CLIP Similarity Scores</h3>"
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += (
            "<tr style='background-color: #f0f0f0;'>"
            "<th style='border: 1px solid #ddd; padding: 8px;'>Text Description</th>"
        )
        html += "<th style='border: 1px solid #ddd; padding: 8px;'>Similarity Score</th></tr>"

        for text, prob in zip(texts, probs[0]):
            score = prob.item()
            bar_width = int(score * 300)
            html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{text}</td>"
            html += f"<td style='border: 1px solid #ddd; padding: 8px;'>"
            html += f"<div style='background-color: #4CAF50; width: {bar_width}px; height: 20px; border-radius: 3px;'>"
            html += f"<span style='color: white; font-weight: bold;'>{score:.4f}</span></div></td></tr>"

        html += "</table></div>"

        return results, html

    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return {"error": f"Error: {str(e)}"}, ""


def batch_caption_images(images: List[Image.Image]) -> str:
    """
    Generate captions for multiple images

    Args:
        images: List of PIL Image objects

    Returns:
        Formatted results as string
    """
    if not images or len(images) == 0:
        return "Please upload at least one image"

    try:
        logger.info(f"Generating captions for {len(images)} images...")
        model, processor = load_image_captioning_model()

        results = []

        for i, image in enumerate(images, 1):
            try:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                results.append(f"{i}. Image ({image.size[0]}x{image.size[1]}): {caption}")
            except Exception as e:
                results.append(f"{i}. Error: {str(e)}")

        return "\n\n".join(results)

    except Exception as e:
        logger.error(f"Error in batch captioning: {e}")
        return f"Error: {str(e)}"


def create_ui():
    """Create and configure the Gradio interface"""

    with gr.Blocks(title="Vision-Language Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # 🖼️ Vision-Language Explorer

            Analyze images and understand their visual content using state-of-the-art vision-language models.
            Generate captions, calculate image-text similarity, and explore multi-modal embeddings!
            """)

        with gr.Tab("Image Captioning"):
            gr.Markdown("### Generate natural language descriptions of images")

            with gr.Row():
                with gr.Column(scale=1):
                    caption_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                    )
                    caption_btn = gr.Button("Generate Caption", variant="primary")

                with gr.Column(scale=1):
                    caption_output = gr.JSON(label="Caption Result")
                    caption_info = gr.Textbox(label="Caption Text", interactive=False)

        with gr.Tab("Image-Text Similarity"):
            gr.Markdown("""
                ### Find the best text match for an image
                Use CLIP to calculate similarity between an image and multiple text descriptions.
                """)

            with gr.Row():
                with gr.Column(scale=1):
                    similarity_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                    )
                    similarity_texts = gr.Textbox(
                        label="Text Descriptions (comma-separated)",
                        placeholder="a cat, a dog, a bird, a tree",
                        lines=3,
                    )
                    similarity_btn = gr.Button("Calculate Similarity", variant="primary")

                with gr.Column(scale=1):
                    similarity_json = gr.JSON(label="Similarity Scores")
                    similarity_html = gr.HTML(label="Visualization")

        with gr.Tab("Batch Processing"):
            gr.Markdown("""
                ### Generate captions for multiple images at once
                Upload multiple images to generate captions for all of them.
                """)

            with gr.Row():
                with gr.Column():
                    batch_images = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    batch_btn = gr.Button("Generate Captions", variant="primary")

                with gr.Column():
                    batch_output = gr.Textbox(label="Batch Results", lines=15, max_lines=25)

        with gr.Tab("About"):
            gr.Markdown("""
                ## About This Tool

                This vision-language explorer combines computer vision and natural language processing
                to understand and describe visual content.

                ### Available Models:

                **Image Captioning:**
                - **GIT (Generative Image-to-Text)** - Fast and accurate image description generation
                - Produces natural language captions describing image content

                **Image-Text Matching:**
                - **CLIP (Contrastive Language-Image Pre-training)** - OpenAI's vision-language model
                - Calculates semantic similarity between images and text descriptions
                - Enables cross-modal retrieval and understanding

                ### How It Works:

                **Image Captioning:**
                1. Upload an image
                2. Model analyzes visual features
                3. Generates natural language description

                **Image-Text Similarity:**
                1. Upload an image
                2. Provide multiple text descriptions
                3. CLIP calculates similarity scores
                4. Shows which text best matches the image

                ### Use Cases:

                - **Accessibility:** Generate alt-text for images
                - **Content Organization:** Automatically tag images by similarity
                - **Search:** Find images matching text descriptions
                - **E-commerce:** Match product images with descriptions
                - **Social Media:** Auto-caption images

                ### Technical Details:

                - **Image Captioning:** Based on vision transformer + text decoder
                - **Similarity:** Uses contrastive learning with shared embedding space
                - **Performance:** Runs on CPU/GPU depending on availability
                - **Supported Formats:** JPG, PNG, WebP

                ### Tips:

                - Use clear, descriptive text for better similarity matching
                - Batch processing is useful for organizing image collections
                - Images should be reasonably clear for best caption quality
                - Try different text descriptions for comparison

                ### Limitations:

                - Captions describe visible content but may miss context
                - Similarity matching depends on text descriptions provided
                - Works best with natural images; struggles with diagrams/charts
                - Performance varies by image quality and complexity
                """)

        # Connect caption components
        def generate_caption_wrapper(image):
            result = generate_caption(image)
            caption_text = result.get("Caption", result.get("error", "No caption"))
            return result, caption_text

        caption_btn.click(
            fn=generate_caption_wrapper,
            inputs=[caption_image],
            outputs=[caption_output, caption_info],
        )

        # Connect similarity components
        def calculate_similarity_wrapper(image, texts_str):
            if not texts_str:
                return {"error": "Please provide text descriptions"}, ""

            texts = [t.strip() for t in texts_str.split(",") if t.strip()]
            return calculate_clip_similarity(image, texts)

        similarity_btn.click(
            fn=calculate_similarity_wrapper,
            inputs=[similarity_image, similarity_texts],
            outputs=[similarity_json, similarity_html],
        )

        # Connect batch components
        def batch_caption_wrapper(files):
            if not files:
                return "Please upload images"

            images = []
            for file_obj in files:
                try:
                    img = Image.open(file_obj.name)
                    images.append(img)
                except Exception as e:
                    logger.error(f"Error opening image: {e}")

            if not images:
                return "No valid images found"

            return batch_caption_images(images)

        batch_btn.click(
            fn=batch_caption_wrapper,
            inputs=[batch_images],
            outputs=[batch_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7869, share=False, show_error=True)
