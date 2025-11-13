"""
Unit tests for UI configuration and theming
"""

import pytest
from ui.ui_config import (
    create_theme,
    create_footer,
    create_header,
    CUSTOM_CSS,
    PLOTLY_CONFIG,
    PLOTLY_LAYOUT,
    ERROR_TEMPLATES,
    SUCCESS_TEMPLATES,
    INFO_TEMPLATES
)


class TestUIConfig:
    """Test UI configuration module"""

    def test_create_theme(self):
        """Test theme creation"""
        theme = create_theme()
        assert theme is not None
        # Theme should be a Gradio theme object
        assert hasattr(theme, 'set')

    def test_create_footer(self):
        """Test footer creation"""
        footer = create_footer()
        assert isinstance(footer, str)
        assert "Built with" in footer
        assert "Hugging Face Transformers" in footer
        assert "Gradio" in footer
        assert "href=" in footer  # Contains links

    def test_create_header(self):
        """Test header creation"""
        title = "Test Title"
        description = "Test Description"
        emoji = "🎨"

        header = create_header(title, description, emoji)
        assert isinstance(header, str)
        assert title in header
        assert description in header
        assert emoji in header
        assert "gradient" in header.lower()  # Has gradient styling

    def test_custom_css_not_empty(self):
        """Test custom CSS is defined"""
        assert isinstance(CUSTOM_CSS, str)
        assert len(CUSTOM_CSS) > 0
        assert ".gradio-container" in CUSTOM_CSS

    def test_plotly_config(self):
        """Test Plotly configuration"""
        assert isinstance(PLOTLY_CONFIG, dict)
        assert "displayModeBar" in PLOTLY_CONFIG
        assert "displaylogo" in PLOTLY_CONFIG
        assert PLOTLY_CONFIG["displaylogo"] is False

    def test_plotly_layout(self):
        """Test Plotly layout settings"""
        assert isinstance(PLOTLY_LAYOUT, dict)
        assert "font" in PLOTLY_LAYOUT
        assert "plot_bgcolor" in PLOTLY_LAYOUT
        assert "paper_bgcolor" in PLOTLY_LAYOUT

    def test_error_templates(self):
        """Test error message templates"""
        assert isinstance(ERROR_TEMPLATES, dict)
        assert "empty_input" in ERROR_TEMPLATES
        assert "model_load_error" in ERROR_TEMPLATES
        assert all(isinstance(msg, str) for msg in ERROR_TEMPLATES.values())

    def test_success_templates(self):
        """Test success message templates"""
        assert isinstance(SUCCESS_TEMPLATES, dict)
        assert "processing_complete" in SUCCESS_TEMPLATES
        assert all(isinstance(msg, str) for msg in SUCCESS_TEMPLATES.values())

    def test_info_templates(self):
        """Test info message templates"""
        assert isinstance(INFO_TEMPLATES, dict)
        assert "loading_model" in INFO_TEMPLATES
        assert all(isinstance(msg, str) for msg in INFO_TEMPLATES.values())
