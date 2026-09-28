"""Resolving Python imports to files in the repo, without importing anything.

Handles src/ layouts and several package roots, relative imports, and
`from pkg import name` where the name is a submodule. Case-exact even on
case-insensitive filesystems.
"""
import functools
import os
from pathlib import Path

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".env", ".next", ".nuxt",
    ".idea", ".vscode", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def find_python_source_roots(repo_root: str) -> list[Path]:
    """Directories that absolute Python imports resolve against.

    Always includes the repo root, plus the parent of every top-level package
    (a dir with __init__.py whose parent has none). That covers `src/` layouts
    (`src/requests/__init__.py` → root `src/`) and monorepos with several
    package roots, without needing to parse pyproject/setup.cfg.
    """
    repo = Path(repo_root).resolve()
    roots = [repo]
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        here = Path(dirpath)
        if "__init__.py" in filenames and here != repo and not (here.parent / "__init__.py").exists():
            if here.parent not in roots:
                roots.append(here.parent)
    return roots


@functools.lru_cache(maxsize=4096)
def _dir_entries(directory: str, mtime_ns: int) -> frozenset[str]:
    # Keyed on the directory's mtime so a long-running server never serves a
    # listing from before files were added or removed.
    try:
        return frozenset(os.listdir(directory))
    except OSError:
        return frozenset()


def _is_file_exact(path: Path) -> bool:
    """is_file() with exact name case, even on case-insensitive filesystems.

    On macOS/Windows `Path("DataSource.py").is_file()` is true when only
    `datasource.py` exists. Django's `from django.contrib.gis.gdal import
    DataSource` (a class) then looked like an import of a submodule, creating
    edges to phantom wrong-case paths instead of to the package.
    """
    if not path.is_file():
        return False
    for node in (path, path.parent):  # the file, and its package directory
        try:
            parent_mtime = node.parent.stat().st_mtime_ns
        except OSError:
            return False
        if node.name not in _dir_entries(str(node.parent), parent_mtime):
            return False
    return True


def _python_module_file(base: Path) -> Path | None:
    """`pkg/mod` → pkg/mod.py or pkg/mod/__init__.py, whichever exists."""
    if base.name and _is_file_exact(candidate := base.with_name(base.name + ".py")):
        return candidate
    if _is_file_exact(candidate := base / "__init__.py"):
        return candidate
    return None


def resolve_python_import(
    imported_module: str,
    source_file: str,
    repo_root: str,
    names: list[str] | None = None,
    source_roots: list[Path] | None = None,
) -> list[str]:
    """Resolve one Python import statement to the in-repo files it loads.

    `imported_module` is the module as written ("os", "pkg.mod", ".", "..pkg").
    `names` are the names after `import` in a from-import. Each name that is
    itself a submodule (`from . import certs`) resolves to that submodule's
    file; names that are attributes resolve to the module file. Returns
    absolute paths; empty if the import is external/unresolvable.
    """
    if not imported_module:
        return []
    names = [n for n in (names or []) if n != "*"]
    level = len(imported_module) - len(imported_module.lstrip("."))
    parts = [p for p in imported_module.lstrip(".").split(".") if p]
    source_dir = Path(source_file).resolve().parent

    if level:
        package = source_dir
        for _ in range(level - 1):
            package = package.parent
        bases = [package.joinpath(*parts)]
    else:
        roots = source_roots if source_roots is not None else [Path(repo_root).resolve()]
        bases = [root.joinpath(*parts) for root in roots]
        # A file's own directory is on sys.path only when that file runs as a
        # script, i.e. it isn't inside a package. Inside a package, `import
        # typing` or `import json` means the stdlib even if the package has
        # its own typing.py or json/ (Flask has both), so don't look there.
        if not (source_dir / "__init__.py").exists():
            bases.append(source_dir.joinpath(*parts))

    for base in bases:
        module_file = _python_module_file(base) if parts else base / "__init__.py"
        if module_file is not None and not _is_file_exact(module_file):
            module_file = None

        resolved: list[Path] = []
        attribute_import = False
        for name in names:
            sub = _python_module_file(base / name)
            if sub is not None:
                resolved.append(sub)
            else:
                attribute_import = True

        if module_file is None and not resolved and not level and len(parts) > 1:
            # `import pkg.mod.attr`-style path where a trailing part isn't a
            # module: fall back to the longest prefix that is one.
            for cut in range(len(parts) - 1, 0, -1):
                prefix = _python_module_file(base.parents[len(parts) - cut - 1])
                if prefix is not None:
                    module_file = prefix
                    break

        if module_file is not None and (attribute_import or not names):
            resolved.append(module_file)
        if resolved:
            return list(dict.fromkeys(str(p.resolve()) for p in resolved))
    return []
