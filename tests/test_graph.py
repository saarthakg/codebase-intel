import pytest
from app.core.graph import DependencyGraph


def make_chain() -> DependencyGraph:
    """A → B → C (A imports B, B imports C)"""
    g = DependencyGraph()
    for f in ["A.py", "B.py", "C.py"]:
        g.add_file(f)
    g.add_import_edge("A.py", "B.py")
    g.add_import_edge("B.py", "C.py")
    return g


def test_dependents_direct():
    g = make_chain()
    result = g.dependents_of("B.py", depth=1)
    files = [r["file"] for r in result]
    assert "A.py" in files
    assert "C.py" not in files


def test_dependents_transitive():
    g = make_chain()
    result = g.dependents_of("C.py", depth=2)
    files = [r["file"] for r in result]
    assert "B.py" in files
    assert "A.py" in files


def test_dependents_depth_limiting():
    g = make_chain()
    result = g.dependents_of("C.py", depth=1)
    files = [r["file"] for r in result]
    assert "B.py" in files
    assert "A.py" not in files  # A is 2 hops away


def test_dependencies_forward():
    g = make_chain()
    result = g.dependencies_of("A.py", depth=2)
    files = [r["file"] for r in result]
    assert "B.py" in files
    assert "C.py" in files


def test_depth_values_correct():
    g = make_chain()
    result = g.dependents_of("C.py", depth=2)
    by_file = {r["file"]: r["depth"] for r in result}
    assert by_file["B.py"] == 1
    assert by_file["A.py"] == 2


def test_deduplication_multiple_paths():
    """D imports B and C; B and C both import A — A should appear once."""
    g = DependencyGraph()
    for f in ["A.py", "B.py", "C.py", "D.py"]:
        g.add_file(f)
    g.add_import_edge("B.py", "A.py")
    g.add_import_edge("C.py", "A.py")
    g.add_import_edge("D.py", "B.py")
    g.add_import_edge("D.py", "C.py")
    result = g.dependents_of("A.py", depth=3)
    files = [r["file"] for r in result]
    # D, B, C should all be present but no duplicates
    assert len(files) == len(set(files))
    assert "B.py" in files
    assert "C.py" in files
    assert "D.py" in files


def test_import_cycle_excludes_start_from_its_own_results():
    """A.py <-> B.py is a real import cycle (common with circular imports).
    dependents_of/dependencies_of must never report a file as one of its own
    dependents/dependencies, even though it's graph-theoretically reachable
    from itself via the cycle."""
    g = DependencyGraph()
    g.add_file("A.py")
    g.add_file("B.py")
    g.add_import_edge("A.py", "B.py")
    g.add_import_edge("B.py", "A.py")

    dependents = [r["file"] for r in g.dependents_of("A.py", depth=5)]
    assert "A.py" not in dependents
    assert "B.py" in dependents

    dependencies = [r["file"] for r in g.dependencies_of("A.py", depth=5)]
    assert "A.py" not in dependencies
    assert "B.py" in dependencies


def test_unknown_file_returns_empty():
    g = make_chain()
    result = g.dependents_of("nonexistent.py", depth=3)
    assert result == []


def test_save_and_load(tmp_path):
    g = make_chain()
    path = str(tmp_path / "graph.json")
    g.save(path)
    g2 = DependencyGraph()
    g2.load(path)
    result = g2.dependents_of("C.py", depth=2)
    files = [r["file"] for r in result]
    assert "B.py" in files and "A.py" in files


def test_edge_count():
    g = make_chain()
    assert g.edge_count == 2


def test_node_count():
    g = make_chain()
    assert g.node_count == 3


# ── Import resolution ─────────────────────────────────────────────────────────

import json
from pathlib import Path

from app.core.graph import (
    find_python_source_roots, load_ts_config, resolve_python_import, resolve_ts_import,
)


def _touch(root: Path, rel: str, text: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _rel(paths: list[str], root: Path) -> list[str]:
    return sorted(Path(p).relative_to(root.resolve()).as_posix() for p in paths)


def _src_layout(tmp_path: Path) -> Path:
    for f in ["src/pkg/__init__.py", "src/pkg/certs.py", "src/pkg/utils.py",
              "src/pkg/sub/__init__.py", "src/pkg/sub/deep.py", "tests/test_x.py"]:
        _touch(tmp_path, f)
    return tmp_path


def test_source_roots_detect_src_layout(tmp_path):
    repo = _src_layout(tmp_path)
    roots = find_python_source_roots(str(repo))
    assert repo.resolve() in roots
    assert (repo / "src").resolve() in roots


def test_absolute_import_resolves_through_src_layout(tmp_path):
    """`import pkg.certs` from tests/ must find src/pkg/certs.py — previously
    every test file in a src/ layout had zero import edges."""
    repo = _src_layout(tmp_path)
    roots = find_python_source_roots(str(repo))
    hits = resolve_python_import("pkg.certs", str(repo / "tests/test_x.py"), str(repo), source_roots=roots)
    assert _rel(hits, repo) == ["src/pkg/certs.py"]


def test_from_dot_import_submodule_resolves_to_submodule(tmp_path):
    """`from . import certs` imports the certs submodule, not the package __init__."""
    repo = _src_layout(tmp_path)
    hits = resolve_python_import(".", str(repo / "src/pkg/utils.py"), str(repo), names=["certs"])
    assert _rel(hits, repo) == ["src/pkg/certs.py"]


def test_from_import_attribute_resolves_to_module(tmp_path):
    repo = _src_layout(tmp_path)
    roots = find_python_source_roots(str(repo))
    hits = resolve_python_import("pkg.utils", str(repo / "tests/test_x.py"), str(repo),
                                 names=["some_function"], source_roots=roots)
    assert _rel(hits, repo) == ["src/pkg/utils.py"]


def test_from_package_import_mixed_submodule_and_attribute(tmp_path):
    repo = _src_layout(tmp_path)
    roots = find_python_source_roots(str(repo))
    hits = resolve_python_import("pkg", str(repo / "tests/test_x.py"), str(repo),
                                 names=["certs", "__version__"], source_roots=roots)
    assert _rel(hits, repo) == ["src/pkg/__init__.py", "src/pkg/certs.py"]


def test_double_dot_relative_import(tmp_path):
    repo = _src_layout(tmp_path)
    hits = resolve_python_import("..utils", str(repo / "src/pkg/sub/deep.py"), str(repo), names=["x"])
    assert _rel(hits, repo) == ["src/pkg/utils.py"]


def test_external_python_import_is_unresolved(tmp_path):
    repo = _src_layout(tmp_path)
    roots = find_python_source_roots(str(repo))
    assert resolve_python_import("numpy.linalg", str(repo / "tests/test_x.py"), str(repo),
                                 source_roots=roots) == []


def test_ts_dotted_filename_is_not_truncated(tmp_path):
    """'./user.service' must resolve to user.service.ts — Path.with_suffix used
    to turn it into './user.ts'."""
    _touch(tmp_path, "src/user.service.ts")
    _touch(tmp_path, "src/user.ts")
    src = _touch(tmp_path, "src/app.ts")
    hit = resolve_ts_import("./user.service", str(src), str(tmp_path))
    assert Path(hit).name == "user.service.ts"


def test_ts_esm_js_extension_maps_to_ts_source(tmp_path):
    _touch(tmp_path, "src/util.ts")
    src = _touch(tmp_path, "src/app.ts")
    hit = resolve_ts_import("./util.js", str(src), str(tmp_path))
    assert Path(hit).name == "util.ts"


def test_ts_index_file_and_external_package(tmp_path):
    _touch(tmp_path, "src/components/index.tsx")
    src = _touch(tmp_path, "src/app.ts")
    assert Path(resolve_ts_import("./components", str(src), str(tmp_path))).name == "index.tsx"
    assert resolve_ts_import("react", str(src), str(tmp_path)) is None


def test_tsconfig_paths_alias_and_base_url(tmp_path):
    _touch(tmp_path, "tsconfig.json", """{
      // comments and trailing commas are legal in tsconfig
      "compilerOptions": {
        "baseUrl": "src",
        "paths": { "@/*": ["*"], "~lib": ["lib/index.ts"], },
      },
    }""")
    _touch(tmp_path, "src/components/Button.tsx")
    _touch(tmp_path, "src/lib/index.ts")
    src = _touch(tmp_path, "src/app.ts")
    cfg = load_ts_config(str(tmp_path))
    assert Path(resolve_ts_import("@/components/Button", str(src), str(tmp_path), cfg)).name == "Button.tsx"
    assert Path(resolve_ts_import("~lib", str(src), str(tmp_path), cfg)).name == "index.ts"
    # baseUrl makes bare paths resolvable too
    assert Path(resolve_ts_import("components/Button", str(src), str(tmp_path), cfg)).name == "Button.tsx"
    assert resolve_ts_import("react", str(src), str(tmp_path), cfg) is None



def test_saved_graph_is_json_not_pickle(tmp_path):
    import json
    g = make_chain()
    path = tmp_path / "graph.json"
    g.save(str(path))
    assert json.loads(path.read_text()) == {
        "nodes": ["A.py", "B.py", "C.py"], "edges": [["A.py", "B.py"], ["B.py", "C.py"]],
    }


def test_legacy_index_graph_is_rebuilt_from_sqlite_not_unpickled(tmp_path, monkeypatch):
    """An index from before the JSON format has a .graph.pkl; it must be
    ignored (never unpickled) and the graph rebuilt from the edges table."""
    import pickle
    from app.core import paths
    from app.state import load_graph
    from app.storage.metadata_store import MetadataStore

    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path)
    store = MetadataStore(str(tmp_path / "old.db"))
    store.upsert_edge("old", "a.py", "b.py", "import")
    store.upsert_symbol("f", "old", "c.py", 1, "function")

    class Boom:
        def __reduce__(self):
            return (exec, ("raise RuntimeError('pickle was loaded')",))

    paths.legacy_graph_path("old").write_bytes(pickle.dumps(Boom()))
    graph = load_graph("old", store)
    assert set(graph.G.edges) == {("a.py", "b.py")}
    assert set(graph.G.nodes) == {"a.py", "b.py", "c.py"}



def test_package_module_does_not_shadow_stdlib(tmp_path):
    """Inside a package, `import typing` / `import json` are the stdlib even if
    the package has its own typing.py and json/ (Flask has both)."""
    for f in ["src/pkg/__init__.py", "src/pkg/typing.py", "src/pkg/json/__init__.py",
              "src/pkg/helpers.py", "scripts/run.py", "scripts/util.py"]:
        _touch(tmp_path, f)
    roots = find_python_source_roots(str(tmp_path))
    helpers = str(tmp_path / "src/pkg/helpers.py")
    assert resolve_python_import("typing", helpers, str(tmp_path), source_roots=roots) == []
    assert resolve_python_import("json", helpers, str(tmp_path), source_roots=roots) == []
    # ...but the package's own modules are still reachable the right way
    assert _rel(resolve_python_import(".typing", helpers, str(tmp_path), names=["x"]), tmp_path) == ["src/pkg/typing.py"]
    assert _rel(resolve_python_import("pkg.json", helpers, str(tmp_path), source_roots=roots), tmp_path) == ["src/pkg/json/__init__.py"]
    # A script outside any package can still import its sibling
    script = str(tmp_path / "scripts/run.py")
    assert _rel(resolve_python_import("util", script, str(tmp_path), source_roots=roots), tmp_path) == ["scripts/util.py"]



def test_class_import_does_not_match_lowercase_module_on_case_insensitive_fs(tmp_path):
    """`from pkg import DataSource` imports a class from pkg/__init__.py. On
    macOS/Windows, DataSource.py "exists" when datasource.py does; that must
    not turn into an edge to a phantom DataSource.py (seen on Django)."""
    _touch(tmp_path, "pkg/__init__.py", "from .datasource import DataSource\n")
    _touch(tmp_path, "pkg/datasource.py", "class DataSource: ...\n")
    _touch(tmp_path, "app.py")
    hits = resolve_python_import("pkg", str(tmp_path / "app.py"), str(tmp_path), names=["DataSource"])
    assert _rel(hits, tmp_path) == ["pkg/__init__.py"]
    # the lowercase submodule import still resolves
    hits = resolve_python_import("pkg", str(tmp_path / "app.py"), str(tmp_path), names=["datasource"])
    assert _rel(hits, tmp_path) == ["pkg/datasource.py"]


def test_ts_import_case_must_match(tmp_path):
    _touch(tmp_path, "src/button.tsx")
    src = _touch(tmp_path, "src/app.ts")
    assert resolve_ts_import("./Button", str(src), str(tmp_path)) is None  # would fail on Linux too
    assert Path(resolve_ts_import("./button", str(src), str(tmp_path))).name == "button.tsx"
