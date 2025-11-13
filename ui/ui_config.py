"""
Shared UI configuration and theming for all NLP demo UIs

Provides consistent styling, themes, and helper functions across all UIs.
"""

import gradio as gr
from typing import Dict, Any

# Custom CSS for enhanced styling
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Header styling */
.markdown h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Card styling */
.gr-box {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Tab styling */
.tab-nav button {
    font-weight: 500;
    transition: all 0.2s ease;
}

.tab-nav button.selected {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Input/Output boxes */
.gr-input, .gr-text-area {
    border-radius: 8px;
    border: 2px solid #e5e7eb;
    transition: border-color 0.2s ease;
}

.gr-input:focus, .gr-text-area:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Code blocks */
.gr-code {
    border-radius: 8px;
    background: #1f2937;
}

/* Examples */
.gr-examples {
    border-radius: 8px;
    background: #f9fafb;
}

/* Progress bars */
.progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Accordions */
.gr-accordion {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

/* Status messages */
.gr-info {
    background: #dbeafe;
    border-left: 4px solid #3b82f6;
}

.gr-warning {
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
}

.gr-error {
    background: #fee2e2;
    border-left: 4px solid #ef4444;
}
"""

# Enhanced theme configuration
def create_theme() -> gr.Theme:
    """Create a custom Gradio theme with professional styling"""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif"
        ],
        font_mono=[
            gr.themes.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace"
        ],
    ).set(
        # Button styling
        button_primary_background_fill="linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #5568d3 0%, #6b3f94 100%)",
        button_primary_text_color="white",
        button_primary_border_color="transparent",
        # Input styling
        input_border_color="#e5e7eb",
        input_border_color_focus="#667eea",
        input_border_width="2px",
        input_radius="8px",
        # Block styling
        block_border_width="1px",
        block_border_color="#e5e7eb",
        block_radius="12px",
        block_shadow="0 1px 3px rgba(0, 0, 0, 0.1)",
        # Panel styling
        panel_border_color="#e5e7eb",
        panel_border_width="1px",
    )


# Plotly chart configuration for consistent styling
PLOTLY_CONFIG: Dict[str, Any] = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'nlp_chart',
        'height': 600,
        'width': 900,
        'scale': 2
    }
}

# Common plotly layout settings
PLOTLY_LAYOUT = dict(
    font=dict(
        family="Inter, sans-serif",
        size=12,
        color="#1f2937"
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=40, r=40, t=60, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Inter, sans-serif"
    ),
    colorway=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
)


def create_footer() -> str:
    """Create a consistent footer for all UIs"""
    return """
    <div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid #e5e7eb;">
        <p style="color: #6b7280; font-size: 14px;">
            Built with ❤️ using <a href="https://huggingface.co/transformers" target="_blank" style="color: #667eea; text-decoration: none;">Hugging Face Transformers</a>
            and <a href="https://gradio.app" target="_blank" style="color: #667eea; text-decoration: none;">Gradio</a>
        </p>
        <p style="color: #9ca3af; font-size: 12px; margin-top: 8px;">
            NLP with Transformers Examples |
            <a href="https://github.com/MysterionRise/nlp-with-transformers-examples" target="_blank" style="color: #9ca3af; text-decoration: none;">GitHub</a>
        </p>
    </div>
    """


def create_header(title: str, description: str, emoji: str = "🤖") -> str:
    """Create a consistent header for all UIs"""
    return f"""
    <div style="text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 2.5rem; margin: 0; font-weight: 700;">
            {emoji} {title}
        </h1>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin-top: 10px; margin-bottom: 0;">
            {description}
        </p>
    </div>
    """


def apply_plotly_theme(fig):
    """Apply consistent theme to Plotly figures"""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# Error message templates
ERROR_TEMPLATES = {
    "empty_input": "⚠️ Please provide some text to analyze.",
    "model_load_error": "❌ Failed to load model. Please try again or select a different model.",
    "processing_error": "❌ An error occurred during processing. Please check your input and try again.",
    "timeout_error": "⏱️ Processing timed out. Please try with shorter text.",
}

# Success message templates
SUCCESS_TEMPLATES = {
    "processing_complete": "✅ Processing complete!",
    "model_loaded": "✅ Model loaded successfully!",
    "export_success": "✅ Export successful!",
}

# Info message templates
INFO_TEMPLATES = {
    "loading_model": "⏳ Loading model... This may take a moment on first run.",
    "processing": "⏳ Processing your request...",
    "large_input": "ℹ️ Large input detected. This may take longer to process.",
}
