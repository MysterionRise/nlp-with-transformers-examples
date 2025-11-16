"""
Example: Using GLiNER for Named Entity Recognition

This example demonstrates how to use GLiNER for zero-shot NER
with custom entity labels.
"""

from gliner import GLiNER

# Load GLiNER model
print("Loading GLiNER model...")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# Example 1: Standard entities
print("\n" + "=" * 80)
print("Example 1: Standard Entity Extraction")
print("=" * 80)

text1 = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976
in Cupertino, California. The company's first product was the Apple I computer.
"""

labels1 = ["person", "organization", "location", "date", "product"]
entities1 = model.predict_entities(text1, labels1, threshold=0.5)

for entity in entities1:
    print(f"{entity['text']:30} | {entity['label']:15} | Score: {entity['score']:.3f}")


# Example 2: Domain-specific entities (Medical)
print("\n" + "=" * 80)
print("Example 2: Medical Domain Entities")
print("=" * 80)

text2 = """
The patient was diagnosed with type 2 diabetes and prescribed metformin 500mg twice daily.
They also reported symptoms of fatigue and increased thirst.
"""

labels2 = ["disease", "medication", "dosage", "symptom"]
entities2 = model.predict_entities(text2, labels2, threshold=0.4)

for entity in entities2:
    print(f"{entity['text']:30} | {entity['label']:15} | Score: {entity['score']:.3f}")


# Example 3: Technical/Programming entities
print("\n" + "=" * 80)
print("Example 3: Technical/Programming Entities")
print("=" * 80)

text3 = """
The application uses Python with Django framework, PostgreSQL database, and Redis for caching.
It's deployed on AWS using Docker containers.
"""

labels3 = ["programming language", "framework", "database", "technology", "cloud provider"]
entities3 = model.predict_entities(text3, labels3, threshold=0.5)

for entity in entities3:
    print(f"{entity['text']:30} | {entity['label']:15} | Score: {entity['score']:.3f}")


# Example 4: Adjusting threshold
print("\n" + "=" * 80)
print("Example 4: Threshold Comparison")
print("=" * 80)

text4 = "Microsoft CEO Satya Nadella announced new AI features in Seattle."
labels4 = ["person", "organization", "location", "technology"]

print("\nWith threshold=0.3:")
entities_low = model.predict_entities(text4, labels4, threshold=0.3)
for entity in entities_low:
    print(f"{entity['text']:30} | {entity['label']:15} | Score: {entity['score']:.3f}")

print("\nWith threshold=0.6:")
entities_high = model.predict_entities(text4, labels4, threshold=0.6)
for entity in entities_high:
    print(f"{entity['text']:30} | {entity['label']:15} | Score: {entity['score']:.3f}")
