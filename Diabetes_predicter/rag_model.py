"""Local, retrieval-based educational assistant for GlucoScope."""

from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).with_name("rag_data.txt")


def _load_documents():
    text = DATA_PATH.read_text(encoding="utf-8")
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


DOCUMENTS = _load_documents()
_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
_MATRIX = _VECTORIZER.fit_transform(DOCUMENTS)


def retrieve(question, limit=3):
    """Return relevant educational passages; unrelated questions return none."""
    query = _VECTORIZER.transform([question])
    scores = cosine_similarity(query, _MATRIX).ravel()
    best = scores.argsort()[::-1]
    return [(DOCUMENTS[i], float(scores[i])) for i in best[:limit] if scores[i] >= 0.05]


def answer(question, context=None):
    """Compose a grounded, transparent reply without pretending to diagnose."""
    matches = retrieve(question)
    if not matches:
        return ("I can help with general diabetes screening and healthy-habit questions, "
                "but I don’t have a reliable reference for that. For personal medical advice, "
                "please ask a qualified healthcare professional.")

    passages = [passage for passage, _ in matches]
    response = " ".join(passages)
    if context and any(term in question.lower() for term in
                       ("my result", "my estimate", "my score", "my risk",
                        "my percentage", "my percent", "this result", "this estimate",
                        "what does this mean", "what does my result mean",
                        "what does my estimate mean", "what does my percentage mean",
                        "what does it say")):
        response = (f"Your GlucoScope estimate was {context['percent']:.1f}% "
                    f"({context['band']} band). It is an educational model estimate, not a diagnosis. "
                    + response)
    return response + " This is general educational information, not a diagnosis or a substitute for care from a clinician."
