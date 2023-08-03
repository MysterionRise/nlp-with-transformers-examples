import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# List of model names to use
model_names = ["facebook/bart-large-cnn", "google/flan-t5-large", "google/pegasus-xsum"]

# List of JSON files to process
json_files = ["data/1.json", "data/2.json", "data/3.json"]

results = {}

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        texts = [item[1] for item in data]  # item[1] corresponds to "text"
        joined_text = " ".join(texts)

        tokens_input = tokenizer.encode(
            "summarize: " + joined_text, return_tensors="pt", truncation=True
        )
        ids = model.generate(tokens_input, min_length=30, max_length=300)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)

        # Store the result
        if model_name not in results:
            results[model_name] = {}
        results[model_name][json_file] = summary

# Save results to a JSON file
with open("results.json", "w") as f:
    json.dump(results, f)
