from unittest.mock import patch

import numpy as np

from app.core import embeddings
from app.core.embeddings import (
    DEFAULT_LOCAL_MODEL, embed_query, embed_texts, get_embedding_model_name, index_embedding_settings,
)
from app.storage.faiss_store import FAISSStore


def _capture_local():
    calls = []

    def fake(texts, model_name):
        calls.append((list(texts), model_name))
        return np.ones((len(texts), 4), dtype=np.float32)

    return calls, fake


def test_model_comes_from_env_for_new_ingests(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert get_embedding_model_name("local") == DEFAULT_LOCAL_MODEL
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    assert get_embedding_model_name("local") == "BAAI/bge-small-en-v1.5"
    assert get_embedding_model_name("openai") == "text-embedding-3-small"


def test_query_prefix_applied_to_queries_only():
    calls, fake = _capture_local()
    with patch.object(embeddings, "_embed_local", side_effect=fake):
        embed_query("where is auth", backend="local", model="BAAI/bge-small-en-v1.5")
        embed_texts(["def auth(): ..."], backend="local", model="BAAI/bge-small-en-v1.5")
        embed_query("where is auth", backend="local", model="all-MiniLM-L6-v2")
    assert calls[0][0][0].startswith("Represent this sentence")
    assert calls[1][0] == ["def auth(): ..."]
    assert calls[2][0] == ["where is auth"]


def test_queries_use_the_model_the_index_was_built_with(monkeypatch):
    """Changing EMBEDDING_MODEL must not change how an existing repo is queried."""
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    store = FAISSStore(dim=384, embedding_backend="local", embedding_model="BAAI/bge-small-en-v1.5")
    assert index_embedding_settings(store) == ("local", "BAAI/bge-small-en-v1.5")

    legacy = FAISSStore(dim=384, embedding_backend="local", embedding_model=None)
    assert index_embedding_settings(legacy) == ("local", DEFAULT_LOCAL_MODEL)


def test_search_embeds_with_index_model(tmp_path, monkeypatch):
    from app.core.search import search_chunks
    from app.storage.metadata_store import MetadataStore

    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    store = FAISSStore(dim=4, embedding_backend="local", embedding_model="BAAI/bge-small-en-v1.5")
    store.add(np.ones((1, 4), dtype=np.float32), ["c1"])
    calls, fake = _capture_local()
    with patch.object(embeddings, "_embed_local", side_effect=fake):
        search_chunks("q", "r", 1, store, MetadataStore(str(tmp_path / "m.db")), mode="semantic")
    assert calls[0][1] == "BAAI/bge-small-en-v1.5"
