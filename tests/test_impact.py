import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.graph import DependencyGraph
from app.core.impact import analyze_impact, analyze_impact_batch
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
    def embed_query(self, text: str, backend: str | None = None, model: str | None = None) -> np.ndarray:
        return np.zeros((1, 8), dtype=np.float32)

    def index_embedding_settings(self, faiss_store):
        return None, None


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_direct_import_alone_is_medium_confidence():
    """An import alone is weak evidence a file changes with the target (0.40)."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    medium = {f.file_path: f for f in resp.medium_confidence}
    assert medium["B.py"].confidence == 0.40
    assert medium["B.py"].reason == "direct import"


def test_transitive_import_counts_like_a_direct_one():
    """A is 2 hops from C: same confidence as a direct import; the hop count
    only breaks ties."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert "A.py" in all_files
    a = all_files["A.py"]
    assert a.confidence == 0.40 and a.depth == 2
    ranked = [f.file_path for f in resp.medium_confidence]
    assert ranked.index("B.py") < ranked.index("A.py")


def test_symbol_target_resolves_through_defining_file():
    """Symbol target: looks up defining file, then traverses graph."""
    g = make_graph_abc()
    meta = make_mock_metadata(
        symbol_results=[{"file_path": "C.py", "kind": "function", "start_line": 5}]
    )
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("some_func", "repo1", g, faiss, meta, MockEmbeddings())
    assert "B.py" in [f.file_path for f in resp.medium_confidence]


def test_symbol_reference_medium_confidence():
    """Files that use a symbol (but don't define it) get confidence 0.70.

    Usages come from the reference index (find_references). This test used to
    feed "reference" rows through find_symbol, which mirrored a bug: Signal 2
    queried the definitions table, so real usages were never found.
    """
    g = DependencyGraph()
    g.add_file("lib.py")
    g.add_file("user.py")
    # No import edges — only symbol references
    meta = MagicMock(spec=MetadataStore)
    meta.find_symbol.return_value = [
        {"file_path": "lib.py", "kind": "function", "start_line": 1, "qualified_name": "some_symbol"}
    ]
    meta.find_references.return_value = [
        {"file_path": "user.py", "line": 3},
        {"file_path": "lib.py", "line": 9},  # recursive call inside the definer: not an impact
    ]
    meta.get_chunk.return_value = None
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("some_symbol", "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert "user.py" in all_files
    assert all_files["user.py"].confidence == 0.70
    assert "lib.py" not in all_files


def test_semantic_similarity_low_confidence():
    """FAISS results that aren't in graph/symbol signals go into 'related'."""
    g = DependencyGraph()
    g.add_file("main.py")
    faiss, meta = make_mock_faiss({"chunk-xyz": "utils.py"})

    resp = analyze_impact("main.py", "repo1", g, faiss, meta, MockEmbeddings())
    related_files = [f.file_path for f in resp.related]
    assert "utils.py" in related_files
    assert resp.related[0].confidence == 0.35


def test_evidence_from_several_signals_combines():
    """A file found by the graph and by history gets their noisy-OR and both
    reasons, and outranks files found by one signal. Semantic neighbours only
    add files no other signal found."""
    from app.core.history import CoChange, Commit
    g = make_graph_abc()
    faiss, meta = make_mock_faiss({"chunk-b": "B.py"})

    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert all_files["B.py"].confidence == 0.40  # semantic hit on B.py is ignored

    history = CoChange.from_commits([Commit(str(i), "d", ["C.py", "A.py"]) for i in range(3)])
    resp = analyze_impact("C.py", "repo1", g, faiss, meta, MockEmbeddings(), cochange=history)
    ranked = resp.high_confidence + resp.medium_confidence + resp.related
    a = ranked[0]
    assert a.file_path == "A.py"
    p = 3 / (3 + 3)  # 3 shared commits, shrunk
    assert a.confidence == pytest.approx(1 - (1 - 0.40) * (1 - (0.4 + 0.5 * p)))
    assert a.reason == "changed together in 3 of 3 commits; transitive import (2 hops)"


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


def test_batch_merges_targets_and_excludes_self():
    """D imports both B and C. Batch impact for [B.py, C.py] should surface D
    once (not twice), with both targets recorded in triggered_by, and neither
    B.py nor C.py should appear as impacted by itself."""
    g = DependencyGraph()
    for f in ["B.py", "C.py", "D.py"]:
        g.add_file(f)
    g.add_import_edge("D.py", "B.py")
    g.add_import_edge("D.py", "C.py")
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact_batch(["B.py", "C.py"], "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert "D.py" in all_files
    assert set(all_files["D.py"].triggered_by) == {"B.py", "C.py"}
    assert "B.py" not in all_files
    assert "C.py" not in all_files


def test_batch_keeps_highest_confidence_across_targets():
    """A.py -> B.py -> C.py. Batch impact for [B.py, C.py]: A.py is a direct
    dependent of B.py and a transitive dependent of C.py; the merged result
    keeps the higher confidence and records both triggers."""
    g = make_graph_abc()
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact_batch(["B.py", "C.py"], "repo1", g, faiss, meta, MockEmbeddings())
    all_files = {f.file_path: f for f in resp.high_confidence + resp.medium_confidence + resp.related}
    assert all_files["A.py"].confidence == 0.40
    assert all_files["A.py"].triggered_by == ["B.py", "C.py"]


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


def test_semantic_signal_queries_with_the_target_files_own_vectors(tmp_path):
    """The semantic query is the target file's content (its chunk vectors), not
    an embedding of the path string — embed_query must not be needed."""
    g = DependencyGraph()
    for f in ["target.py", "similar.py", "unrelated.py"]:
        g.add_file(f)
    store = MetadataStore(str(tmp_path / "m.db"))
    chunks = [
        ChunkMetadata(chunk_id=cid, file_path=fp, language="python", start_line=1, end_line=1,
                      symbols=[], imports=[], content="x\n")
        for cid, fp in [("t", "target.py"), ("s", "similar.py"), ("u", "unrelated.py")]
    ]
    store.add_chunks(chunks, "r")
    faiss = FAISSStore(dim=3)
    faiss.add(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=np.float32), ["t", "s", "u"])

    class NoTextEmbedding:
        def index_embedding_settings(self, s): return None, None
        def embed_query(self, *a, **k): raise AssertionError("should use stored vectors")

    resp = analyze_impact("target.py", "r", g, faiss, store, NoTextEmbedding())
    related = [f.file_path for f in resp.related]
    assert related[0] == "similar.py"
    assert "target.py" not in related


def test_test_named_for_target_ranks_first_and_tests_are_listed():
    g = DependencyGraph()
    for f in ["src/pkg/adapters.py", "src/pkg/models.py", "tests/test_adapters.py",
              "tests/test_other.py", "tests/testserver/server.py"]:
        g.add_file(f)
    g.add_import_edge("src/pkg/models.py", "src/pkg/adapters.py")
    g.add_import_edge("tests/test_other.py", "src/pkg/models.py")
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []

    resp = analyze_impact("src/pkg/adapters.py", "repo1", g, faiss, meta, MockEmbeddings())
    assert resp.high_confidence[0].file_path == "tests/test_adapters.py"
    assert resp.high_confidence[0].reason == "test named for this file"
    assert [t.file_path for t in resp.tests] == ["tests/test_adapters.py", "tests/test_other.py"]


def test_batch_lists_tests_too():
    g = DependencyGraph()
    for f in ["a.py", "tests/test_a.py"]:
        g.add_file(f)
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []
    resp = analyze_impact_batch(["a.py"], "repo1", g, faiss, meta, MockEmbeddings())
    assert [t.file_path for t in resp.tests] == ["tests/test_a.py"]
    assert resp.tests[0].triggered_by == ["a.py"]



def test_equal_confidence_ties_prefer_frequently_changed_files_then_path():
    """Ranking must not depend on graph insertion order; among equal evidence,
    files that change more often (base rate) come first."""
    from app.core.history import CoChange, Commit
    g = DependencyGraph()
    for f in ["t.py", "z_busy.py", "a_quiet.py", "m_quiet.py"]:
        g.add_file(f)
    for f in ["z_busy.py", "m_quiet.py", "a_quiet.py"]:  # insertion order deliberately scrambled
        g.add_import_edge(f, "t.py")
    history = CoChange.from_commits([Commit(str(i), "d", ["z_busy.py", "x.py"]) for i in range(5)])
    meta = make_mock_metadata()
    faiss = MagicMock(spec=FAISSStore)
    faiss.search.return_value = []
    resp = analyze_impact("t.py", "repo1", g, faiss, meta, MockEmbeddings(), cochange=history)
    assert [f.file_path for f in resp.medium_confidence] == ["z_busy.py", "a_quiet.py", "m_quiet.py"]
