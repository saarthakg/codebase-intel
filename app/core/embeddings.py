import os
from typing import Optional

import numpy as np
from tqdm import tqdm

BATCH_SIZE = 64
_local_model = None  # lazy-loaded singleton


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def _embed_local(texts: list[str]) -> np.ndarray:
    model = _get_local_model()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", unit="batch", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs.astype(np.float32))
    return np.vstack(all_embeddings)


def _embed_openai(texts: list[str]) -> np.ndarray:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (OpenAI)", unit="batch", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embs = np.array([d.embedding for d in response.data], dtype=np.float32)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 numpy array of embeddings."""
    if not texts:
        raise ValueError("texts must be non-empty")
    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    if backend == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(query: str) -> np.ndarray:
    """Return (1, D) float32 numpy array."""
    return embed_texts([query])


def get_embedding_dim() -> int:
    """Return the dimension of embeddings for the current backend."""
    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    if backend == "openai":
        return 1536  # text-embedding-3-small
    return 384  # all-MiniLM-L6-v2
