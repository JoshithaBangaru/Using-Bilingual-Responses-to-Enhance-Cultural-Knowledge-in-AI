"""Cultural KB: builds the FAISS index (run as a script) and serves scoped retrieval."""
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from textutils import extract_text

_model = None
_index = None
_metadata = []


def load_retriever_resources():
    """Load the embedding model and index ONCE (they stay in memory afterwards)."""
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    if _index is None:
        if not (os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH)):
            raise FileNotFoundError(
                "KB index not found. Run `python scrape_cultural_kb.py` and then "
                "`python kb_retriever.py` to build it."
            )
        _index = faiss.read_index(config.INDEX_PATH)
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        if _index.ntotal != len(_metadata):
            raise RuntimeError(
                f"Index has {_index.ntotal} vectors but metadata has {len(_metadata)} entries; "
                "rebuild with `python kb_retriever.py`."
            )


def retrieve_cultural_evidence(query: str, country: str = None, top_k: int = 5):
    """Two-stage scoped retrieval: dense recall first, then an exact country filter."""
    load_retriever_resources()
    query_vec = _model.encode([f"query: {query}"], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    fetch_k = top_k * 5 if country else top_k
    while True:
        fetch_k = min(fetch_k, _index.ntotal)
        scores, ids = _index.search(query_vec, fetch_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = _metadata[idx]
            if country and str(item.get("country", "")).lower() != country.lower():
                continue
            entry = dict(item)
            entry["score"] = float(score)
            results.append(entry)
            if len(results) >= top_k:
                break
        # widen the candidate pool if the country filter removed too many
        if len(results) >= top_k or not country or fetch_k >= _index.ntotal:
            return results
        fetch_k *= 4


def search_kb(query: str, country: str = None, top_k: int = 5):
    """Unified wrapper used by the API and the pipeline."""
    return retrieve_cultural_evidence(query=query, country=country, top_k=top_k)


def build_index():
    if not os.path.exists(config.DATA_PATH):
        raise SystemExit(f"{config.DATA_PATH} not found - run scrape_cultural_kb.py first.")
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    documents = [f"passage: {extract_text(item)}" for item in raw_data]
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, config.INDEX_PATH)
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    counts = {}
    for item in raw_data:
        counts[item.get("country", "?")] = counts.get(item.get("country", "?"), 0) + 1
    print(f"Indexed {len(raw_data)} facts (d={embeddings.shape[1]}): {counts}")


if __name__ == "__main__":
    build_index()
