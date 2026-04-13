import os
import tempfile
import pytest

from app.storage.metadata_store import MetadataStore
from app.models.schemas import ChunkMetadata
from app.core.ingest import walk_repo, detect_language, load_file


# ── MetadataStore tests ───────────────────────────────────────────────────────

def make_chunk(**kwargs) -> ChunkMetadata:
    defaults = dict(
        chunk_id="test-chunk-1",
        file_path="src/foo.py",
        language="python",
        start_line=1,
        end_line=20,
        symbols=["foo", "Bar"],
        imports=["os", "sys"],
        content="def foo(): pass",
    )
    defaults.update(kwargs)
    return ChunkMetadata(**defaults)


def test_metadata_store_creates_tables(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    # If tables are missing, queries below would raise; passing means tables exist
    assert store.get_chunk("nonexistent") is None
    store.close()


def test_upsert_and_get_chunk(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    chunk = make_chunk()
    store.upsert_chunk(chunk, repo_id="repo1")
    retrieved = store.get_chunk("test-chunk-1")
    assert retrieved is not None
    assert retrieved.chunk_id == "test-chunk-1"
    assert retrieved.file_path == "src/foo.py"
    assert retrieved.symbols == ["foo", "Bar"]
    assert retrieved.imports == ["os", "sys"]
    store.close()


def test_get_chunks_by_file(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    c1 = make_chunk(chunk_id="c1", start_line=1, end_line=10)
    c2 = make_chunk(chunk_id="c2", start_line=8, end_line=20)
    store.upsert_chunk(c1, "repo1")
    store.upsert_chunk(c2, "repo1")
    chunks = store.get_chunks_by_file("repo1", "src/foo.py")
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "c1"
    store.close()


def test_upsert_and_find_symbol(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    store.upsert_symbol("submit_order", "repo1", "orders.py", 42, "function")
    results = store.find_symbol("repo1", "submit_order")
    assert len(results) == 1
    assert results[0]["file_path"] == "orders.py"
    assert results[0]["start_line"] == 42
    assert results[0]["kind"] == "function"
    store.close()


def test_find_symbol_missing(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    results = store.find_symbol("repo1", "nonexistent_fn")
    assert results == []
    store.close()


def test_upsert_edge(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    store.upsert_edge("repo1", "a.py", "b.py", "import")
    edges = store.get_edges_from("repo1", "a.py")
    assert len(edges) == 1
    assert edges[0]["target_file"] == "b.py"
    edges_to = store.get_edges_to("repo1", "b.py")
    assert edges_to[0]["source_file"] == "a.py"
    store.close()


# ── Ingest tests ──────────────────────────────────────────────────────────────

def test_walk_repo_basic(tmp_path):
    # Create some files
    (tmp_path / "main.py").write_text("print('hi')")
    (tmp_path / "utils.ts").write_text("export {}")
    (tmp_path / "README.md").write_text("# Readme")
    paths = walk_repo(str(tmp_path))
    names = [os.path.basename(p) for p in paths]
    assert "main.py" in names
    assert "utils.ts" in names
    assert "README.md" in names


def test_walk_repo_skips_directories(tmp_path):
    for skip_dir in ["node_modules", ".git", "__pycache__", ".venv"]:
        d = tmp_path / skip_dir
        d.mkdir()
        (d / "file.py").write_text("x = 1")
    (tmp_path / "real.py").write_text("real = True")
    paths = walk_repo(str(tmp_path))
    names = [os.path.basename(p) for p in paths]
    assert "real.py" in names
    assert "file.py" not in names


def test_walk_repo_skips_extensions(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "archive.zip").write_bytes(b"PK")
    (tmp_path / "code.py").write_text("x = 1")
    paths = walk_repo(str(tmp_path))
    names = [os.path.basename(p) for p in paths]
    assert "code.py" in names
    assert "image.png" not in names
    assert "archive.zip" not in names


def test_detect_language():
    assert detect_language("foo.py") == "python"
    assert detect_language("bar.ts") == "typescript"
    assert detect_language("baz.tsx") == "typescript"
    assert detect_language("app.js") == "javascript"
    assert detect_language("app.jsx") == "javascript"
    assert detect_language("notes.md") == "markdown"
    assert detect_language("config.yaml") == "unknown"
    assert detect_language("unknown.xyz") == "unknown"


def test_load_file_text(tmp_path):
    f = tmp_path / "hello.py"
    f.write_text("print('hello')")
    content = load_file(str(f))
    assert content == "print('hello')"


def test_load_file_binary_returns_none(tmp_path):
    f = tmp_path / "binary.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    content = load_file(str(f))
    assert content is None


def test_load_file_missing_returns_none():
    content = load_file("/nonexistent/path/file.py")
    assert content is None
