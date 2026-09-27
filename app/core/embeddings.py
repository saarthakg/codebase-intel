import os
from typing import Optional

import numpy as np
from tqdm import tqdm

BATCH_SIZE = 64
_local_model = None  # lazy-loaded singleton

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    return _local_model


def _embed_local(texts: list[str]) -> np.ndarray:
    model = _get_local_model()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", unit="batch", leave=False,
                  disable=len(texts) <= BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs.astype(np.float32))
    return np.vstack(all_embeddings)


def _embed_openai(texts: list[str]) -> np.ndarray:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (OpenAI)", unit="batch", leave=False,
                  disable=len(texts) <= BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=OPENAI_MODEL_NAME, input=batch)
        embs = np.array([d.embedding for d in response.data], dtype=np.float32)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def _resolve_backend(backend: Optional[str]) -> str:
    return (backend or os.environ.get("EMBEDDING_BACKEND", "local")).lower()


def embed_texts(texts: list[str], backend: Optional[str] = None) -> np.ndarray:
    """Return (N, D) float32 numpy array of embeddings.

    `backend` overrides the EMBEDDING_BACKEND env var. Pass it explicitly at
    query time using the backend a repo was actually indexed with (see
    FAISSStore.embedding_backend) — otherwise a query embedded with a
    different backend than the index will silently produce garbage results
    (or a hard dimension-mismatch error) if EMBEDDING_BACKEND has since changed.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    resolved = _resolve_backend(backend)
    if resolved == "openai":
        return _embed_openai(texts)
    return _embed_local(texts)


def embed_query(query: str, backend: Optional[str] = None) -> np.ndarray:
    """Return (1, D) float32 numpy array."""
    return embed_texts([query], backend=backend)


def get_embedding_dim(backend: Optional[str] = None) -> int:
    """Return the dimension of embeddings for the given (or configured) backend."""
    resolved = _resolve_backend(backend)
    if resolved == "openai":
        return 1536  # text-embedding-3-small
    return 384  # all-MiniLM-L6-v2


def get_embedding_model_name(backend: Optional[str] = None) -> str:
    resolved = _resolve_backend(backend)
    return OPENAI_MODEL_NAME if resolved == "openai" else LOCAL_MODEL_NAME
