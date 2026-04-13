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


def test_unknown_file_returns_empty():
    g = make_chain()
    result = g.dependents_of("nonexistent.py", depth=3)
    assert result == []


def test_save_and_load(tmp_path):
    g = make_chain()
    path = str(tmp_path / "graph.pkl")
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
