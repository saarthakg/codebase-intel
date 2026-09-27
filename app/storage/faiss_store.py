"""Exact cosine-similarity vector index, stored as a numpy array.

Named FAISSStore for historical reasons: it used a faiss IndexFlatIP. That was
replaced with numpy (same exact search, same results) because faiss-cpu and
torch each bundle their own OpenMP runtime on macOS, and once both were
initialized in one process it aborted: a server whose first request ran a
vector search from stored vectors (/impact) crashed on the next /search.
At repo scale (thousands to ~100K chunks) a normalized matrix-vector product
is as fast as faiss's flat index.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np


class IndexFormatError(FileNotFoundError):
    """The index on disk was written in an older format and must be re-ingested."""


class FAISSStore:
    def __init__(
        self,
        dim: int,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)  # L2-normalized rows
        self.id_map: list[str] = []          # row → chunk_id
        # Which embedding backend/model built this index. Persisted so query-time
        # code can embed with the *same* backend the index was built with, rather
        # than trusting whatever EMBEDDING_BACKEND happens to be set to right now.
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self._positions: dict[str, int] = {}

    @property
    def ntotal(self) -> int:
        return len(self.id_map)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so inner product equals cosine similarity."""
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        return vectors / norms

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Normalize embeddings, add to index, record chunk_ids."""
        if len(embeddings) == 0:
            return
        if embeddings.shape[-1] != self.dim:
            raise ValueError(f"Embeddings have dim {embeddings.shape[-1]}, index has dim {self.dim}")
        self.vectors = np.vstack([self.vectors, self._normalize(embeddings)])
        self.id_map.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) pairs ordered by score descending."""
        if self.ntotal == 0:
            return []
        if query_embedding.shape[-1] != self.dim:
            raise ValueError(
                f"Query embedding has dim {query_embedding.shape[-1]} but this index "
                f"was built with dim {self.dim} (embedding_backend="
                f"{self.embedding_backend!r}). Embed the query with the same backend "
                f"the repo was ingested with."
            )
        k = min(top_k, self.ntotal)
        query = self._normalize(np.asarray(query_embedding).reshape(1, -1))[0]
        scores = self.vectors @ query
        top = np.argpartition(-scores, k - 1)[:k] if k < self.ntotal else np.arange(self.ntotal)
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.id_map[i], float(scores[i])) for i in top]

    def vectors_for(self, chunk_ids: list[str]) -> np.ndarray:
        """Stored (normalized) vectors for these chunk ids, (N, dim); unknown ids skipped."""
        if len(self._positions) != len(self.id_map):
            self._positions = {cid: i for i, cid in enumerate(self.id_map)}
        rows = [self._positions[c] for c in chunk_ids if c in self._positions]
        return self.vectors[rows] if rows else np.zeros((0, self.dim), dtype=np.float32)

    def save(self, path: str) -> None:
        """Save vectors (numpy .npy format, at `path`) + id_map/metadata JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, self.vectors, allow_pickle=False)
        idmap_path = path.replace(".index", ".idmap.json")
        with open(idmap_path, "w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "id_map": self.id_map,
                    "embedding_backend": self.embedding_backend,
                    "embedding_model": self.embedding_model,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load vectors + id_map from disk."""
        try:
            with open(path, "rb") as f:
                vectors = np.load(f, allow_pickle=False)
        except ValueError:
            raise IndexFormatError(
                f"The index at {path} was written by an older version (faiss format). "
                f"Re-run ingest for this repo; unchanged code isn't re-embedded, so it's quick."
            ) from None
        idmap_path = path.replace(".index", ".idmap.json")
        with open(idmap_path) as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.vectors = vectors.astype(np.float32).reshape(-1, self.dim)
        self.id_map = data["id_map"]
        self.embedding_backend = data.get("embedding_backend")
        self.embedding_model = data.get("embedding_model")
        self._positions = {}
