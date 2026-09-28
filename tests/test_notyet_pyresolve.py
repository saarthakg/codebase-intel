"""Python import resolution (notyet.pyresolve)."""
from pathlib import Path

from notyet.pyresolve import find_python_source_roots, resolve_python_import


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

