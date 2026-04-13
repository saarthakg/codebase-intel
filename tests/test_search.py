import numpy as np
import pytest
from app.storage.faiss_store import FAISSStore


DIM = 8


def unit_vector(i: int, dim: int = DIM) -> np.ndarray:
    """Create a unit vector with a 1 at position i."""
    v = np.zeros((1, dim), dtype=np.float32)
    v[0, i % dim] += 1.0
    return v


def test_add_and_search_returns_exact_match():
    store = FAISSStore(dim=DIM)
    emb = unit_vector(0)
    store.add(emb, ["chunk-0"])
    results = store.search(unit_vector(0), top_k=1)
    assert len(results) == 1
    assert results[0][0] == "chunk-0"


def test_self_search_score_is_one():
    store = FAISSStore(dim=DIM)
    emb = unit_vector(3)
    store.add(emb, ["chunk-3"])
    results = store.search(unit_vector(3), top_k=1)
    assert abs(results[0][1] - 1.0) < 1e-5


def test_search_returns_correct_ranking():
    store = FAISSStore(dim=DIM)
    # Add 3 orthogonal vectors
    for i in range(3):
        store.add(unit_vector(i), [f"chunk-{i}"])
    # Query closest to chunk-1
    results = store.search(unit_vector(1), top_k=3)
    assert results[0][0] == "chunk-1"


def test_top_k_limiting():
    store = FAISSStore(dim=DIM)
    for i in range(5):
        store.add(unit_vector(i), [f"chunk-{i}"])
    results = store.search(unit_vector(0), top_k=2)
    assert len(results) == 2


def test_empty_store_returns_empty():
    store = FAISSStore(dim=DIM)
    results = store.search(unit_vector(0), top_k=5)
    assert results == []


def test_save_and_load(tmp_path):
    store = FAISSStore(dim=DIM)
    store.add(unit_vector(0), ["chunk-0"])
    store.add(unit_vector(1), ["chunk-1"])
    path = str(tmp_path / "test.index")
    store.save(path)

    store2 = FAISSStore(dim=DIM)
    store2.load(path)
    results = store2.search(unit_vector(0), top_k=1)
    assert results[0][0] == "chunk-0"


def test_normalization_consistency():
    """Non-unit vectors should still find the right match after normalization."""
    store = FAISSStore(dim=DIM)
    # Add a vector scaled by 5
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, 2] = 5.0
    store.add(v, ["chunk-2"])
    # Query with a differently-scaled version
    q = np.zeros((1, DIM), dtype=np.float32)
    q[0, 2] = 100.0
    results = store.search(q, top_k=1)
    assert results[0][0] == "chunk-2"
    assert abs(results[0][1] - 1.0) < 1e-5
