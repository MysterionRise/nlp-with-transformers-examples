import json

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the JSON data from a file
with open("data/1.json", "r") as f:
    data = json.load(f)

# Extract the texts
texts = [item[1] for item in data]  # item[1] corresponds to "text"

# Join the texts into a single string
joined_text = " ".join(texts)

# Join the texts into a single string
joined_text = " ".join(texts)

# Summarize the joined text
summary = summarizer(joined_text, max_length=150, min_length=30, do_sample=False)

print("Summary:")
print(summary[0]["summary_text"])
