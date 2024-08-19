# !pip install nltk rouge-score scikit-learn torch transformers bert-score

import nltk
from bert_score import score
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")


def evaluate_scores(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)

    # BLEU Score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)

    # METEOR Score
    meteor = meteor_score.single_meteor_score(reference_tokens, candidate_tokens)

    # Cosine Similarity
    vectorizer = TfidfVectorizer().fit_transform([reference, candidate])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]

    # BERTScore
    P, R, F1 = score([candidate], [reference], lang="en", verbose=True)

    return {
        "BLEU": bleu_score,
        "ROUGE": rouge_scores,
        "METEOR": meteor,
        "Cosine Similarity": cosine_sim,
        "BERTScore Precision": P.mean().item(),
        "BERTScore Recall": R.mean().item(),
        "BERTScore F1": F1.mean().item(),
    }


reference = (
    "The investment strategy involves diversifying assets across "
    "various sectors to mitigate risks and maximize returns."
)
candidate = (
    "The strategy for investments involves diversifying assets "
    "across multiple sectors to minimize risks and optimize returns."
)

scores = evaluate_scores(reference, candidate)
print("Scores:", scores)
