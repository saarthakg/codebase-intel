import json
from unittest.mock import patch

import pytest

from app.core import paths
from app.core.pipeline import IngestError, run_ingestion
from app.storage.metadata_store import MetadataStore


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


def test_reindex_replaces_not_accumulates(tmp_path):
    repo = _make_repo(tmp_path)
    run_ingestion(str(repo), "myrepo")
    run_ingestion(str(repo), "myrepo")  # re-index same repo_id

    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.count_symbols("myrepo") == 2
    assert store.indexed_files("myrepo") == ["a.py", "b.py"]


def test_reindex_drops_removed_symbol(tmp_path):
    """A symbol removed from source shouldn't linger in the DB after re-indexing."""
    repo = _make_repo(tmp_path)
    run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_symbol("myrepo", "bar")

    (repo / "b.py").write_text("def baz():\n    return 1\n")  # renamed bar -> baz
    run_ingestion(str(repo), "myrepo")
    store2 = MetadataStore(str(paths.db_path("myrepo")))
    assert store2.find_symbol("myrepo", "bar") == []
    assert store2.find_symbol("myrepo", "baz")


def test_writes_meta_json(tmp_path):
    repo = _make_repo(tmp_path)
    summary = run_ingestion(str(repo), "myrepo")
    assert summary["files_indexed"] == 2
    with open(paths.meta_path("myrepo")) as f:
        meta = json.load(f)
    assert meta["repo_id"] == "myrepo"
    assert meta["repo_path"] == str(repo.resolve())
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


def test_empty_repo_still_creates_loadable_index(tmp_path):
    from app.state import get_repo_state
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    (repo / "image.png").write_bytes(b"\x89PNG")  # filtered out, no indexable files
    summary = run_ingestion(str(repo), "myrepo")
    assert summary["files_indexed"] == 0
    assert get_repo_state("myrepo").graph.G.number_of_nodes() == 0
    assert paths.known_repo_ids() == ["myrepo"]


def test_package_init_does_not_get_self_edges(tmp_path):
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


def test_same_name_methods_all_stored_with_qualified_names(tmp_path):
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


def test_references_are_stored_and_pruned_to_repo_symbols(tmp_path):
    repo = _make_repo(tmp_path)  # a.py calls b.bar()
    run_ingestion(str(repo), "myrepo")
    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_references("myrepo", "bar") == [{"file_path": "a.py", "line": 4}]
    # `return` isn't an identifier and `b` isn't a defined symbol → pruned
    assert store.find_references("myrepo", "b") == []


def test_failed_index_leaves_previous_index_intact(tmp_path):
    """If indexing fails partway, the DB rebuild must roll back rather than
    leave a cleared or half-written DB."""
    repo = _make_repo(tmp_path)
    run_ingestion(str(repo), "myrepo")

    (repo / "b.py").write_text("def renamed():\n    return 1\n")
    with patch("app.core.pipeline.cochange_for_repo", side_effect=RuntimeError("git crashed")):
        with pytest.raises(RuntimeError):
            run_ingestion(str(repo), "myrepo")

    store = MetadataStore(str(paths.db_path("myrepo")))
    assert store.find_symbol("myrepo", "bar")          # old state still there
    assert store.find_symbol("myrepo", "renamed") == []
    assert store.indexed_files("myrepo") == ["a.py", "b.py"]


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
    assert store.find_references("r", "foo") == []  # new table exists, empty until re-index


def test_search_era_tables_are_dropped(tmp_path):
    """A v2 database (with code chunks, keyword index, caches) upgrades to v3."""
    import sqlite3
    db = tmp_path / "v2.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE answer_cache (cache_key TEXT PRIMARY KEY);
        CREATE TABLE symbols (symbol_name TEXT NOT NULL, qualified_name TEXT NOT NULL,
            repo_id TEXT NOT NULL, file_path TEXT NOT NULL, start_line INTEGER NOT NULL,
            end_line INTEGER, kind TEXT, PRIMARY KEY (repo_id, file_path, qualified_name, start_line));
        PRAGMA user_version = 2;
    """)
    conn.close()
    store = MetadataStore(str(db))
    tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"chunks", "answer_cache", "embedding_cache", "chunks_fts"}.isdisjoint(tables)
    assert "files" in tables and store.schema_version() == 3
