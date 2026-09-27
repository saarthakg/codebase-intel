import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

BATCH_SIZE = 64

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"
# Kept for backwards compatibility with code that imported it.
LOCAL_MODEL_NAME = DEFAULT_LOCAL_MODEL

# Cap on tokens per input. Chunks are ~1600 chars (~400-500 tokens); models
# that accept 8K tokens would otherwise pad/attend over far more than needed.
MAX_SEQ_TOKENS = 512


@dataclass(frozen=True)
class LocalModelSpec:
    dim: int
    # Some retrieval models are trained with an instruction on the query side
    # only; documents are embedded as-is.
    query_prefix: str = ""
    trust_remote_code: bool = False


# Known local models. Any other sentence-transformers model name also works
# (dimension is read from the model), just without a query prefix.
LOCAL_MODELS: dict[str, LocalModelSpec] = {
    "all-MiniLM-L6-v2": LocalModelSpec(dim=384),
    "BAAI/bge-small-en-v1.5": LocalModelSpec(
        dim=384, query_prefix="Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-base-en-v1.5": LocalModelSpec(
        dim=768, query_prefix="Represent this sentence for searching relevant passages: "
    ),
    "jinaai/jina-embeddings-v2-base-code": LocalModelSpec(dim=768, trust_remote_code=True),
}

_local_models: dict[str, object] = {}  # lazy-loaded, one instance per model name


def _get_local_model(name: str):
    if name not in _local_models:
        from sentence_transformers import SentenceTransformer
        spec = LOCAL_MODELS.get(name, LocalModelSpec(dim=0))
        model = SentenceTransformer(name, trust_remote_code=spec.trust_remote_code)
        model.max_seq_length = min(model.max_seq_length or MAX_SEQ_TOKENS, MAX_SEQ_TOKENS)
        _local_models[name] = model
    return _local_models[name]


def _embed_local(texts: list[str], model_name: str) -> np.ndarray:
    model = _get_local_model(model_name)
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


def get_embedding_model_name(backend: Optional[str] = None) -> str:
    """The model to use for a *new* ingest: EMBEDDING_MODEL for the local
    backend (default all-MiniLM-L6-v2), text-embedding-3-small for OpenAI."""
    if _resolve_backend(backend) == "openai":
        return OPENAI_MODEL_NAME
    return os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_LOCAL_MODEL


def _resolve_model(backend: str, model: Optional[str]) -> str:
    if backend == "openai":
        return OPENAI_MODEL_NAME
    return model or get_embedding_model_name(backend)


def embed_texts(
    texts: list[str],
    backend: Optional[str] = None,
    model: Optional[str] = None,
    is_query: bool = False,
) -> np.ndarray:
    """Return (N, D) float32 numpy array of embeddings.

    `backend`/`model` override the EMBEDDING_BACKEND/EMBEDDING_MODEL env vars.
    Pass them explicitly at query time using what the repo was actually indexed
    with (FAISSStore.embedding_backend / .embedding_model) — otherwise a query
    embedded with a different model than the index will silently produce
    garbage results (or a hard dimension-mismatch error) if the env has since
    changed. `is_query` applies the model's query instruction, if it has one.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    resolved = _resolve_backend(backend)
    if resolved == "openai":
        return _embed_openai(texts)
    model_name = _resolve_model(resolved, model)
    prefix = LOCAL_MODELS.get(model_name, LocalModelSpec(dim=0)).query_prefix if is_query else ""
    if prefix:
        texts = [prefix + t for t in texts]
    return _embed_local(texts, model_name)


def embed_query(query: str, backend: Optional[str] = None, model: Optional[str] = None) -> np.ndarray:
    """Return (1, D) float32 numpy array."""
    return embed_texts([query], backend=backend, model=model, is_query=True)


def get_embedding_dim(backend: Optional[str] = None, model: Optional[str] = None) -> int:
    """Return the dimension of embeddings for the given (or configured) backend/model."""
    resolved = _resolve_backend(backend)
    if resolved == "openai":
        return 1536  # text-embedding-3-small
    model_name = _resolve_model(resolved, model)
    if model_name in LOCAL_MODELS:
        return LOCAL_MODELS[model_name].dim
    return _get_local_model(model_name).get_sentence_embedding_dimension()


def index_embedding_settings(faiss_store) -> tuple[Optional[str], Optional[str]]:
    """(backend, model) to embed queries with for an existing index.

    Indexes record the backend/model they were built with. Old local indexes
    that predate model recording were all built with all-MiniLM-L6-v2, so fall
    back to that rather than to whatever EMBEDDING_MODEL is set to today.
    """
    backend = getattr(faiss_store, "embedding_backend", None)
    model = getattr(faiss_store, "embedding_model", None)
    if model is None and (backend or "local") == "local":
        model = DEFAULT_LOCAL_MODEL
    return backend, model
