import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized
        self.id_map: list[str] = []          # position → chunk_id

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so inner product equals cosine similarity."""
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        return vectors / norms

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Normalize embeddings, add to index, record chunk_ids."""
        if len(embeddings) == 0:
            return
        normalized = self._normalize(embeddings)
        self.index.add(normalized)
        self.id_map.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) pairs ordered by score descending."""
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        normalized = self._normalize(query_embedding)
        scores, indices = self.index.search(normalized, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.id_map[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        """Save index + id_map to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path)
        idmap_path = path.replace(".index", ".idmap.json")
        with open(idmap_path, "w") as f:
            json.dump({"dim": self.dim, "id_map": self.id_map}, f)

    def load(self, path: str) -> None:
        """Load index + id_map from disk."""
        self.index = faiss.read_index(path)
        idmap_path = path.replace(".index", ".idmap.json")
        with open(idmap_path) as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.id_map = data["id_map"]
