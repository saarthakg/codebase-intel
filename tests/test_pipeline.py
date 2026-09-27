import json
from unittest.mock import patch

import numpy as np
import pytest

from app.core import paths
from app.core.pipeline import IngestError, run_ingestion
from app.storage.metadata_store import MetadataStore


def _fake_embed_texts(texts, backend=None, **kwargs):
    return np.random.rand(len(texts), 8).astype(np.float32)


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import b\n\ndef foo():\n    return b.bar()\n")
    (repo / "b.py").write_text("def bar():\n    return 1\n")
    return repo


@pytest.fixture(autouse=True)
def _isolated_data_dirs(tmp_path, monkeypatch):
    """Point the shared paths module at a scratch dir so tests don't touch real data/."""
    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    yield


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_reingest_replaces_not_accumulates(mock_embed, tmp_path):
    repo = _make_repo(tmp_path)
    run_ingestion(str(repo), "myrepo")
    run_ingestion(str(repo), "myrepo")  # re-ingest same repo_id

    store = MetadataStore(str(paths.db_path("myrepo")))
    # Two files, one chunk each → 2 chunks, not 4, after two ingests.
    assert store.count_chunks("myrepo") == 2
    assert store.count_symbols("myrepo") == 2


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_reingest_drops_removed_symbol(mock_embed, tmp_path):
    """A symbol removed from source shouldn't linger in the DB after re-ingest."""
    repo = _make_repo(tmp_path)
    run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_symbol("myrepo", "bar")

    (repo / "b.py").write_text("def baz():\n    return 1\n")  # renamed bar -> baz
    run_ingestion(str(repo), "myrepo")
    store2 = MetadataStore(str(paths.db_path("myrepo")))
    assert store2.find_symbol("myrepo", "bar") == []
    assert store2.find_symbol("myrepo", "baz")


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_writes_meta_json(mock_embed, tmp_path):
    repo = _make_repo(tmp_path)
    summary = run_ingestion(str(repo), "myrepo")
    assert summary["files_indexed"] == 2
    with open(paths.meta_path("myrepo")) as f:
        meta = json.load(f)
    assert meta["repo_id"] == "myrepo"
    assert meta["embedding_backend"]
    assert meta["ingested_at"]


def test_rejects_invalid_repo_id(tmp_path):
    repo = _make_repo(tmp_path)
    with pytest.raises(ValueError):
        run_ingestion(str(repo), "../escape")


def test_rejects_non_directory_repo_path(tmp_path):
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("hi")
    with pytest.raises(IngestError):
        run_ingestion(str(not_a_dir), "myrepo")


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_empty_repo_still_creates_loadable_index(mock_embed, tmp_path):
    """A repo with zero indexable files shouldn't leave get_repo_state() unable
    to find an index — an empty FAISS index should still be written."""
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    (repo / "image.png").write_bytes(b"\x89PNG")  # filtered out, no indexable files
    summary = run_ingestion(str(repo), "myrepo")
    assert summary["chunks_indexed"] == 0
    assert paths.index_path("myrepo").exists()
    assert paths.idmap_path("myrepo").exists()


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_package_init_does_not_get_self_edges(mock_embed, tmp_path):
    """`from . import helper_attr` inside __init__.py resolves to __init__.py itself;
    that must not become a self-edge, and `from . import mod` must link to mod.py."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("VERSION = 1\nfrom . import VERSION\nfrom . import mod\n")
    (repo / "pkg" / "mod.py").write_text("from . import VERSION\n")
    run_ingestion(str(repo), "myrepo")

    from app.core.graph import DependencyGraph
    g = DependencyGraph()
    g.load(str(paths.graph_path("myrepo")))
    edges = set(g.G.edges)
    assert ("pkg/__init__.py", "pkg/__init__.py") not in edges
    assert ("pkg/__init__.py", "pkg/mod.py") in edges
    assert ("pkg/mod.py", "pkg/__init__.py") in edges


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_same_name_methods_all_stored_with_qualified_names(mock_embed, tmp_path):
    """Every `send` must be stored — the old (name, file) key kept only the last one."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(
        "class A:\n    def send(self):\n        pass\n\n"
        "class B:\n    def send(self):\n        return A().send()\n"
    )
    summary = run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    assert {r["qualified_name"] for r in store.find_symbol("myrepo", "send")} == {"A.send", "B.send"}
    assert [r["start_line"] for r in store.find_symbol("myrepo", "B.send")] == [6]
    assert summary["symbols_extracted"] == 4  # A, A.send, B, B.send


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_references_are_stored_and_pruned_to_repo_symbols(mock_embed, tmp_path):
    repo = _make_repo(tmp_path)  # a.py calls b.bar()
    run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_references("myrepo", "bar") == [{"file_path": "a.py", "line": 4}]
    # `return` isn't an identifier and `b` isn't a defined symbol → pruned
    assert store.find_references("myrepo", "b") == []


@patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts)
def test_chunks_list_only_their_own_symbols(mock_embed, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    body = "".join(f"def f{i}():\n" + "    x = 1\n" * 40 + "\n" for i in range(6))
    (repo / "big.py").write_text(body)
    run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    chunks = store.get_chunks_by_file("myrepo", "big.py")
    assert len(chunks) > 1
    for i in range(6):
        # def f{i} sits on line 1 + 42*i; only chunks covering that line list it
        # (previously every chunk carried the whole file's symbol list).
        def_line = 1 + 42 * i
        holders = [c for c in chunks if f"f{i}" in c.symbols]
        assert holders and all(c.start_line <= def_line <= c.end_line for c in holders)


def test_failed_ingest_leaves_previous_index_intact(tmp_path):
    """If embedding fails, the DB rebuild must roll back rather than leave a
    cleared/half-written DB that no longer matches the FAISS index."""
    repo = _make_repo(tmp_path)
    with patch("app.core.pipeline.embed_texts", side_effect=_fake_embed_texts):
        run_ingestion(str(repo), "myrepo")

    (repo / "b.py").write_text("def renamed():\n    return 1\n")
    with patch("app.core.pipeline.embed_texts", side_effect=RuntimeError("model download failed")):
        with pytest.raises(RuntimeError):
            run_ingestion(str(repo), "myrepo")

    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_symbol("myrepo", "bar")          # old state still there
    assert store.find_symbol("myrepo", "renamed") == []
    assert store.count_chunks("myrepo") == 2


def test_legacy_symbols_table_is_migrated(tmp_path):
    """DBs from before qualified names get upgraded in place, keeping their rows."""
    import sqlite3
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE symbols (symbol_name TEXT NOT NULL, repo_id TEXT NOT NULL,
            file_path TEXT NOT NULL, start_line INTEGER, kind TEXT,
            PRIMARY KEY (symbol_name, repo_id, file_path));
        CREATE INDEX idx_symbols_repo_name ON symbols (repo_id, symbol_name);
        INSERT INTO symbols VALUES ('foo', 'r', 'a.py', 3, 'function');
    """)
    conn.commit()
    conn.close()

    store = MetadataStore(str(db))
    rows = store.find_symbol("r", "foo")
    assert rows and rows[0]["qualified_name"] == "foo" and rows[0]["start_line"] == 3
    assert store.find_references("r", "foo") == []  # new table exists, empty until re-ingest


# ── Embedding cache (incremental re-ingest) ───────────────────────────────────

class _CountingEmbedder:
    """Deterministic fake embedder that records every text it embeds."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts, backend=None, **kwargs):
        self.calls.append(list(texts))
        rng = [np.random.default_rng(abs(hash(t)) % (2**32)) for t in texts]
        return np.vstack([r.random(8) for r in rng]).astype(np.float32)

    @property
    def total(self):
        return sum(len(c) for c in self.calls)


def test_reingest_reuses_embeddings_for_unchanged_code(tmp_path):
    repo = _make_repo(tmp_path)
    embed = _CountingEmbedder()
    with patch("app.core.pipeline.embed_texts", side_effect=embed):
        first = run_ingestion(str(repo), "myrepo")
        assert (first["chunks_embedded"], first["chunks_reused"]) == (2, 0)

        second = run_ingestion(str(repo), "myrepo")
        assert (second["chunks_embedded"], second["chunks_reused"]) == (0, 2)
        assert embed.total == 2  # nothing re-embedded

        (repo / "b.py").write_text("def bar():\n    return 2\n")
        third = run_ingestion(str(repo), "myrepo")
    assert (third["chunks_embedded"], third["chunks_reused"]) == (1, 1)
    assert embed.calls[-1] == [t for t in embed.calls[-1] if "return 2" in t]  # only b.py's chunk


def test_cached_vectors_equal_fresh_ones(tmp_path):
    """A re-ingest served from the cache must produce the same index vectors."""
    from app.storage.faiss_store import FAISSStore

    def vectors_by_file():
        store = MetadataStore(str(paths.db_path("myrepo")))
        index = FAISSStore(dim=8)
        index.load(str(paths.index_path("myrepo")))
        return {store.get_chunk(cid).file_path: index.vectors_for([cid])[0] for cid in index.id_map}

    repo = _make_repo(tmp_path)
    with patch("app.core.pipeline.embed_texts", side_effect=_CountingEmbedder()):
        run_ingestion(str(repo), "myrepo")
        fresh = vectors_by_file()
        assert run_ingestion(str(repo), "myrepo")["chunks_reused"] == 2
        reused = vectors_by_file()
    assert sorted(fresh) == sorted(reused) == ["a.py", "b.py"]
    assert all(np.allclose(fresh[f], reused[f]) for f in fresh)


def test_changing_model_does_not_reuse_other_models_vectors(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    embed = _CountingEmbedder()
    with patch("app.core.pipeline.embed_texts", side_effect=embed):
        run_ingestion(str(repo), "myrepo")
        monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
        again = run_ingestion(str(repo), "myrepo")
    assert again["chunks_embedded"] == 2 and again["chunks_reused"] == 0
    store = MetadataStore(str(paths.db_path("myrepo")))
    models = {r[0] for r in store._conn.execute("SELECT DISTINCT model FROM embedding_cache")}
    assert models == {"local:some/other-model"}  # old model's vectors pruned


def test_duplicate_chunks_are_embedded_once(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "x.txt").write_text("same boilerplate\n")
    (repo / "y.txt").write_text("same boilerplate\n")
    embed = _CountingEmbedder()
    with patch("app.core.pipeline.embed_texts", side_effect=embed), \
         patch("app.core.pipeline.embedding_text", side_effect=lambda c, s=None: c.content):
        summary = run_ingestion(str(repo), "myrepo")
    assert embed.total == 1
    assert (summary["chunks_embedded"], summary["chunks_reused"]) == (1, 0)
    assert summary["chunks_indexed"] == 2
