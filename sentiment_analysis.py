import csv

from transformers import pipeline


# not doing anything for now
def preprocess(content):
    return content


if __name__ == "__main__":
    filename = "data/all_reviews.csv"
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # Create pipeline once before processing reviews (more efficient)
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    # open the file for reading in text mode
    with open(filename, "r", newline="") as f:
        # create a CSV reader object
        reader = csv.reader(f)

        # iterate over each row in the file
        for row in reader:
            content = row[3]
            score = row[4]
            content = preprocess(content)
            print(content)
            print(score)
            print(sentiment_task(content))
