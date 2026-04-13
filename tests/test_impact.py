import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.graph import DependencyGraph
from app.core.impact import analyze_impact
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore
from app.models.schemas import ChunkMetadata


def make_graph_abc() -> DependencyGraph:
    """A → B → C"""
    g = DependencyGraph()
    for f in ["A.py", "B.py", "C.py"]:
        g.add_file(f)
    g.add_import_edge("A.py", "B.py")
    g.add_import_edge("B.py", "C.py")
    return g


def make_mock_metadata(symbol_results=None) -> MagicMock:
    store = MagicMock(spec=MetadataStore)
    store.find_symbol.return_value = symbol_results or []
    store.get_chunk.return_value = None
    return store


def make_mock_faiss(chunk_file_map: dict[str, str]) -> tuple[MagicMock, MagicMock]:
    """Returns (faiss_store_mock, metadata_store_mock with chunk lookup)."""
    faiss = MagicMock(spec=FAISSStore)
    hits = [(cid, 0.5) for cid in chunk_file_map]
    faiss.search.return_value = hits

    meta = MagicMock(spec=MetadataStore)
    meta.find_symbol.return_value = []

    def get_chunk(cid):
        if cid in chunk_file_map:
            return ChunkMetadata(
                chunk_id=cid,
                file_path=chunk_file_map[cid],
                language="python",
                start_line=1,
                end_line=10,
                symbols=[],
                imports=[],
                content="",
            )
        return None

    meta.get_chunk.side_effect = get_chunk
    return faiss, meta


class MockEmbeddings:
    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros((1, 8), dtype=np.float32)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_high_confidence_direct_import():
    """Files that directly import target get confidence 0.95."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    high_files = [f.file_path for f in resp.high_confidence]
    assert "B.py" in high_files
    assert resp.high_confidence[0].confidence == 0.95


def test_medium_confidence_transitive_import():
    """A is 2 hops from C, so confidence = 0.75 → high_confidence bucket."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert "A.py" in all_files
    a = all_files["A.py"]
    assert a.confidence == 0.75  # depth=2 → 0.75 → high_confidence bucket


def test_symbol_target_resolves_through_defining_file():
    """Symbol target: looks up defining file, then traverses graph."""
    g = make_graph_abc()
    meta = make_mock_metadata(
        symbol_results=[{"file_path": "C.py", "kind": "function", "start_line": 5}]
    )
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("some_func", "repo1", g, faiss, meta, MockEmbeddings())
    high_files = [f.file_path for f in resp.high_confidence]
    assert "B.py" in high_files


def test_symbol_reference_medium_confidence():
    """Files that reference a symbol (but don't define it) get confidence 0.70."""
    g = DependencyGraph()
    g.add_file("A.py")
    g.add_file("user.py")
    # No import edges — only symbol references
    meta = MagicMock(spec=MetadataStore)
    meta.find_symbol.return_value = [
        {"file_path": "user.py", "kind": "reference", "start_line": 3}
    ]
    meta.get_chunk.return_value = None
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("some_symbol", "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert "user.py" in all_files
    assert all_files["user.py"].confidence == 0.70


def test_semantic_similarity_low_confidence():
    """FAISS results that aren't in graph/symbol signals go into 'related'."""
    g = DependencyGraph()
    g.add_file("main.py")
    faiss, meta = make_mock_faiss({"chunk-xyz": "utils.py"})

    resp = analyze_impact("main.py", "repo1", g, faiss, meta, MockEmbeddings())
    related_files = [f.file_path for f in resp.related]
    assert "utils.py" in related_files
    assert resp.related[0].confidence == 0.35


def test_deduplication_keeps_highest_confidence():
    """If a file appears in both graph signal and semantic, keep higher confidence."""
    g = make_graph_abc()
    faiss, meta = make_mock_faiss({"chunk-b": "B.py"})

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    # B.py is in graph (0.95); also in semantic (0.35) — should keep 0.95
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert all_files["B.py"].confidence == 0.95


def test_buckets_thresholds():
    """Verify exact bucket boundaries: >=0.7 high, >=0.4 medium, <0.4 related."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    for f in resp.high_confidence:
        assert f.confidence >= 0.7
    for f in resp.medium_confidence:
        assert 0.4 <= f.confidence < 0.7
    for f in resp.related:
        assert f.confidence < 0.4


def test_unknown_target_returns_empty():
    """If target doesn't exist in graph or symbols, all buckets are empty."""
    g = DependencyGraph()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("ghost.py", "repo1", g, faiss, meta, MockEmbeddings())
    assert resp.high_confidence == []
    assert resp.medium_confidence == []
    # related may have semantic hits (empty in this mock)
    assert resp.related == []
