import json
from unittest.mock import patch

import numpy as np
import pytest

from app.core import paths
from app.core.pipeline import IngestError, run_ingestion
from app.storage.metadata_store import MetadataStore


def _fake_embed_texts(texts, backend=None):
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
