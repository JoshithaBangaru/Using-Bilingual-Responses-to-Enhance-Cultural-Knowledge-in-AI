"""Semantic disagreement between the local-language and English answers."""
import numpy as np

import config

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.SIMILARITY_MODEL_NAME)
    return _model


def semantic_similarity(text_a: str, text_b: str) -> float:
    # E5 recommends the "query:" prefix for symmetric similarity tasks.
    embeddings = get_embedding_model().encode(
        [f"query: {text_a}", f"query: {text_b}"], normalize_embeddings=True
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def check_disagreement(local_answer: str, english_answer: str, threshold: float = None):
    """Return (cosine similarity, is_disagreed)."""
    threshold = config.DISAGREEMENT_THRESHOLD if threshold is None else threshold
    similarity = semantic_similarity(local_answer, english_answer)
    return similarity, similarity < threshold
