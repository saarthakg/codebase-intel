import os
import tempfile
import pytest

from codebase_intel.storage.metadata_store import MetadataStore
from codebase_intel.core.ingest import walk_repo, detect_language, load_file


# ── MetadataStore tests ───────────────────────────────────────────────────────

def test_metadata_store_creates_tables(tmp_path):
    store = MetadataStore(str(tmp_path / "test.db"))
    # If tables are missing, queries below would raise; passing means tables exist
    assert store.indexed_files("repo1") == []
    store.add_files("repo1", [("src/foo.py", "python")])
    assert store.indexed_files("repo1") == ["src/foo.py"]
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
    assert store.all_edges("repo1") == [("a.py", "b.py")]
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


# ── scan_repo: .gitignore, lockfiles, minified, size ──────────────────────────

import os
import subprocess

from codebase_intel.core.ingest import scan_repo


def _rel(scan, root):
    return sorted(os.path.relpath(f, root) for f in scan.files)


def _git_init(repo):
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)


def test_scan_respects_gitignore_including_nested(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "gen").mkdir(parents=True)
    (repo / "build_out").mkdir()
    (repo / ".gitignore").write_text("build_out/\n*.generated.py\n")
    (repo / "src" / ".gitignore").write_text("gen/\n")
    (repo / "src" / "app.py").write_text("x = 1\n")
    (repo / "src" / "schema.generated.py").write_text("x = 1\n")
    (repo / "src" / "gen" / "stub.py").write_text("x = 1\n")
    (repo / "build_out" / "bundle.js").write_text("x = 1\n")
    (repo / "untracked_but_not_ignored.py").write_text("x = 1\n")
    _git_init(repo)
    scan = scan_repo(str(repo))
    assert scan.used_git
    assert _rel(scan, repo) == ["src/app.py", "untracked_but_not_ignored.py"]


def test_scan_without_git_falls_back_to_walk(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    scan = scan_repo(str(tmp_path))
    assert not scan.used_git
    assert _rel(scan, tmp_path) == ["main.py"]


def test_scan_skips_lockfiles_minified_and_large_files(tmp_path, monkeypatch):
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n')
    (tmp_path / "vendor.min.js").write_text("var a=1;\n")
    (tmp_path / "bundle.js").write_text("var a=1;" * 2000 + "\n")        # one 16 KB line
    (tmp_path / "normal.js").write_text("const a = 1;\n" * 200)
    (tmp_path / "huge.py").write_text("x = 1\n" * 50_000)                 # ~300 KB
    monkeypatch.setenv("INGEST_MAX_FILE_BYTES", "100000")
    scan = scan_repo(str(tmp_path))
    assert _rel(scan, tmp_path) == ["normal.js"]
    assert dict(scan.skipped) == {"lockfile": 1, "minified": 2, "too_large": 1}


def test_scan_of_git_subdirectory_lists_only_that_subtree(tmp_path):
    repo = tmp_path / "mono"
    (repo / "svc_a").mkdir(parents=True)
    (repo / "svc_b").mkdir()
    (repo / "svc_a" / "a.py").write_text("x = 1\n")
    (repo / "svc_b" / "b.py").write_text("x = 1\n")
    _git_init(repo)
    scan = scan_repo(str(repo / "svc_a"))
    assert _rel(scan, repo / "svc_a") == ["a.py"]


def test_schema_is_versioned_and_newer_dbs_are_refused(tmp_path):
    import sqlite3

    import pytest
    from codebase_intel.storage.metadata_store import SchemaVersionError

    db = tmp_path / "v.db"
    store = MetadataStore(str(db))
    assert store.schema_version() == MetadataStore.SCHEMA_VERSION
    store.close()
    MetadataStore(str(db)).close()  # reopening is a no-op

    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {MetadataStore.SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError, match="newer version"):
        MetadataStore(str(db))
