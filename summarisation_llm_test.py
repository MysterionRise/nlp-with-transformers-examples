import json
import logging

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# List of model names to use
model_names = [
    "facebook/bart-large-cnn",
    "google/flan-t5-xxl",
    "google/flan-t5-xl",
    "google/flan-t5-large",
    "google/pegasus-xsum",
]

start, finish = 1, 8  # Define the start and finish
json_files = list(map(lambda i: f"data/{i}.json", range(start, finish + 1)))

results = {}

for model_idx, model_name in tqdm(enumerate(model_names), total=len(model_names), desc="Models"):
    logging.warning(f"Processing model {model_idx + 1} of {len(model_names)}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    for file_idx, json_file in tqdm(enumerate(json_files), total=len(json_files), desc="Files"):
        logging.warning(f"Processing file {file_idx + 1} of {len(json_files)}: {json_file}")
        with open(json_file, "r") as f:
            data = json.load(f)

        texts = [item[1] for item in data["raw_answers"]]  # item[1] corresponds to "text"
        joined_text = " ".join(texts)

        tokens_input = tokenizer.encode("summarize: " + joined_text, return_tensors="pt", truncation=True)
        ids = model.generate(tokens_input, min_length=30, max_length=300)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)

        # Store the result
        if model_name not in results:
            results[model_name] = {}
        results[model_name][data["query"]] = summary

# Save results to a JSON file
with open("results.json", "w") as f:
    json.dump(results, f)
