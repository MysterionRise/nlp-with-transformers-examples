#!/usr/bin/env python3
"""
UI Launcher - Launch all NLP demo UIs

This script provides an easy way to launch all available interactive UIs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Available UIs with their configurations
UIS = {
    "sentiment": {
        "name": "Sentiment Analysis Playground",
        "file": "ui/sentiment_playground.py",
        "port": 7860,
        "description": "Analyze sentiment with multiple models",
    },
    "similarity": {
        "name": "Sentence Similarity Explorer",
        "file": "ui/similarity_explorer.py",
        "port": 7861,
        "description": "Compare sentence embeddings and semantic similarity",
    },
    "ner": {
        "name": "NER Visualizer",
        "file": "ui/ner_visualizer.py",
        "port": 7862,
        "description": "Extract and visualize named entities",
    },
    "summarization": {
        "name": "Text Summarization Studio",
        "file": "ui/summarization_studio.py",
        "port": 7863,
        "description": "Generate and compare text summaries",
    },
    "performance": {
        "name": "Model Performance Dashboard",
        "file": "ui/performance_dashboard.py",
        "port": 7864,
        "description": "Compare and evaluate model performance",
    },
    "qa": {
        "name": "Question Answering System",
        "file": "ui/qa_system.py",
        "port": 7865,
        "description": "Extract answers from context using QA models",
    },
    "generation": {
        "name": "Text Generation Playground",
        "file": "ui/generation_playground.py",
        "port": 7866,
        "description": "Generate creative text completions",
    },
    "zero_shot": {
        "name": "Zero-Shot Classifier",
        "file": "ui/zero_shot_classifier.py",
        "port": 7867,
        "description": "Classify text into custom categories without training data",
    },
    "translation": {
        "name": "Translation Hub",
        "file": "ui/translation_hub.py",
        "port": 7868,
        "description": "Translate between 50+ languages",
    },
    "vision": {
        "name": "Vision-Language Explorer",
        "file": "ui/vision_language_explorer.py",
        "port": 7869,
        "description": "Analyze images with vision-language models",
    },
}


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("  NLP Transformers Examples - Interactive UI Launcher")
    print("=" * 70)
    print()


def list_uis():
    """List all available UIs"""
    print("Available UIs:")
    print()
    for key, info in UIS.items():
        print(f"  {key:15} - {info['name']}")
        print(f"  {' ' * 15}   {info['description']}")
        print(f"  {' ' * 15}   Port: {info['port']}")
        print()


def launch_ui(ui_key: str):
    """Launch a specific UI"""
    if ui_key not in UIS:
        print(f"Error: Unknown UI '{ui_key}'")
        print()
        list_uis()
        return False

    ui_info = UIS[ui_key]
    ui_file = Path(ui_info["file"])

    if not ui_file.exists():
        print(f"Error: UI file not found: {ui_file}")
        return False

    print(f"Launching {ui_info['name']}...")
    print(f"Port: {ui_info['port']}")
    print(f"URL: http://localhost:{ui_info['port']}")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 70)
    print()

    try:
        subprocess.run([sys.executable, str(ui_file)], check=True)
        return True
    except KeyboardInterrupt:
        print("\nShutting down...")
        return True
    except Exception as e:
        print(f"Error launching UI: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch interactive UIs for NLP demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_ui.py sentiment              # Launch sentiment analysis UI
  python launch_ui.py qa                     # Launch question answering UI
  python launch_ui.py --list                 # List all available UIs
  python launch_ui.py                        # Show menu to select UI

Available UIs:
  sentiment      - Sentiment Analysis Playground
  similarity     - Sentence Similarity Explorer
  ner            - Named Entity Recognition Visualizer
  summarization  - Text Summarization Studio
  performance    - Model Performance Dashboard
  qa             - Question Answering System
  generation     - Text Generation Playground
  zero_shot      - Zero-Shot Classifier
  translation    - Translation Hub
  vision         - Vision-Language Explorer
        """,
    )

    parser.add_argument(
        "ui",
        nargs="?",
        choices=list(UIS.keys()),
        help="UI to launch (sentiment, similarity, ner, summarization, performance, "
        "qa, generation, zero_shot, translation, vision)",
    )

    parser.add_argument("--list", "-l", action="store_true", help="List all available UIs")

    args = parser.parse_args()

    print_banner()

    if args.list:
        list_uis()
        return 0

    if args.ui:
        success = launch_ui(args.ui)
        return 0 if success else 1

    # Interactive menu
    list_uis()
    print("Enter the name of the UI you want to launch (or 'q' to quit):")
    print()

    while True:
        try:
            choice = input("UI> ").strip().lower()

            if choice in ["q", "quit", "exit"]:
                print("Goodbye!")
                return 0

            if choice in UIS:
                launch_ui(choice)
                break
            else:
                print(f"Unknown UI: {choice}")
                print("Available options:", ", ".join(UIS.keys()))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except EOFError:
            print("\nGoodbye!")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
