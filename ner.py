import argparse

import spacy
from gliner import GLiNER
from spacy import displacy

# Sample text for NER demonstration
raw_text = (
    "West Germany (German: Westdeutschland) is the colloquial English "
    "term used to indicate the Federal Republic of Germany (FRG; German: Bundesrepublik Deutschland "
    "[ˈbʊndəsʁepuˌbliːk ˈdɔʏtʃlant] (listen), BRD) between its formation on 23 May 1949 "
    "and the German reunification through the accession of East Germany on 3 October 1990. "
    "During the Cold War, the western portion of Germany and the associated territory of West Berlin "
    "were parts of the Western Bloc. West Germany was formed as a political entity during the Allied "
    "occupation of Germany after World War II, established from 12 states formed in the three Allied zones "
    "of occupation held by the United States, the United Kingdom, and France. "
    "The FRG's provisional capital was the city of Bonn, and the Cold War era country is retrospectively "
    "designated as the Bonn Republic (Bonner Republik).[4]"
)


def run_spacy_ner(text, model_name="en_core_web_trf"):
    """
    Run NER using spaCy model.

    Args:
        text: Input text for NER
        model_name: spaCy model name (default: en_core_web_trf)

    Returns:
        spaCy Doc object with entities
    """
    print(f"Loading spaCy model: {model_name}")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Trying en_core_web_sm...")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    print("\n" + "=" * 80)
    print("spaCy NER Results")
    print("=" * 80)
    for ent in doc.ents:
        print(f"{ent.text:30} | {ent.label_:15} | Start: {ent.start_char:4} | End: {ent.end_char:4}")

    print("\n" + "=" * 80)
    print("HTML Visualization")
    print("=" * 80)
    print(displacy.render(doc, style="ent"))

    return doc


def run_gliner_ner(text, model_name="urchade/gliner_multi-v2.1", labels=None):
    """
    Run NER using GLiNER model.

    Args:
        text: Input text for NER
        model_name: GLiNER model name from HuggingFace
        labels: List of entity labels to extract. If None, uses default set.

    Returns:
        List of entities
    """
    if labels is None:
        # Default entity types similar to spaCy
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
            "law",
            "work of art",
        ]

    print(f"Loading GLiNER model: {model_name}")
    model = GLiNER.from_pretrained(model_name)

    print(f"Entity labels: {', '.join(labels)}")

    # Predict entities
    entities = model.predict_entities(text, labels, threshold=0.5)

    print("\n" + "=" * 80)
    print("GLiNER Results")
    print("=" * 80)
    for entity in entities:
        print(
            f"{entity['text']:30} | {entity['label']:15} | "
            f"Start: {entity['start']:4} | End: {entity['end']:4} | "
            f"Score: {entity['score']:.3f}"
        )

    return entities


def main():
    parser = argparse.ArgumentParser(description="Named Entity Recognition with spaCy or GLiNER")
    parser.add_argument(
        "--model",
        type=str,
        choices=["spacy", "gliner", "both"],
        default="both",
        help="Choose NER model: 'spacy', 'gliner', or 'both' (default: both)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Custom text for NER. If not provided, uses default example.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_trf",
        help="spaCy model name (default: en_core_web_trf)",
    )
    parser.add_argument(
        "--gliner-model",
        type=str,
        default="urchade/gliner_multi-v2.1",
        help="GLiNER model name from HuggingFace (default: urchade/gliner_multi-v2.1)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Entity labels for GLiNER (space-separated). If not provided, uses default set.",
    )

    args = parser.parse_args()

    # Use custom text if provided, otherwise use default
    text = args.text if args.text else raw_text

    if args.model in ["spacy", "both"]:
        run_spacy_ner(text, args.spacy_model)

    if args.model in ["gliner", "both"]:
        if args.model == "both":
            print("\n\n")
        run_gliner_ner(text, args.gliner_model, args.labels)


if __name__ == "__main__":
    main()
