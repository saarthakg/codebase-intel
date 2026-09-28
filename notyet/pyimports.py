"""Which files import which, in the working tree: for choosing the tests that
exercise a change.

Imports are parsed with `ast` and resolved with notyet.pyresolve
(source roots, relative imports, submodule names). Results are cached in
.git/notyet/cache/ per file content (git blob id); the cache is dropped when
the set of Python files changes, since that changes how imports resolve. On
a repo the size of Django the first build takes seconds; later checks only
re-resolve changed files.
"""
import ast
import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Iterable

from notyet.pyresolve import find_python_source_roots, resolve_python_import
from notyet import snapshot, store


def python_blobs(root: str, tree: str) -> dict[str, str]:
    """path → blob id for every .py file in `tree`."""
    out = snapshot.git(root, "ls-tree", "-r", "-z", tree)
    blobs = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        if path.endswith(".py") and meta.split()[1] == "blob":
            blobs[path] = meta.split()[2]
    return blobs


def _imports_of(source: str) -> list[tuple[str, list[str]]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    specs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            specs += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            specs.append(("." * node.level + (node.module or ""), [a.name for a in node.names]))
    return specs


class ImportGraph:
    def __init__(self, root: str, tree: str):
        self.root = root
        self.blobs = python_blobs(root, tree)
        self.edges: dict[str, list[str]] = self._load_or_build()
        self.reverse: dict[str, set[str]] = {}
        for src, targets in self.edges.items():
            for t in targets:
                self.reverse.setdefault(t, set()).add(src)

    def _cache_path(self) -> Path:
        layout = hashlib.sha1("\n".join(sorted(self.blobs)).encode()).hexdigest()[:16]
        d = store.state_dir(self.root) / "cache"
        d.mkdir(exist_ok=True)
        return d / f"imports-{layout}.json"

    def _load_or_build(self) -> dict[str, list[str]]:
        path = self._cache_path()
        cached = json.loads(path.read_text()) if path.exists() else {}
        roots = None
        edges, fresh = {}, {}
        for rel, blob in self.blobs.items():
            hit = cached.get(rel)
            if hit and hit[0] == blob:
                edges[rel] = hit[1]
            else:
                if roots is None:
                    roots = find_python_source_roots(self.root)
                edges[rel] = self._resolve(rel, roots)
            fresh[rel] = [blob, edges[rel]]
        if fresh != cached:
            for old in path.parent.glob("imports-*.json"):  # one layout's cache at a time
                if old != path:
                    old.unlink(missing_ok=True)
            path.write_text(json.dumps(fresh))
        return edges

    def _resolve(self, rel: str, roots) -> list[str]:
        full = os.path.join(self.root, rel)
        try:
            source = Path(full).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        targets = set()
        for module, names in _imports_of(source):
            for hit in resolve_python_import(module, full, self.root, names=names, source_roots=roots):
                target = os.path.relpath(hit, self.root)
                if target != rel and target in self.blobs:
                    targets.add(target)
        return sorted(targets)

    def importers(self, targets: Iterable[str], max_depth: int = 4) -> dict[str, int]:
        """Files that import any of `targets`, directly (depth 1) or through
        other files, with the smallest depth for each."""
        seen: dict[str, int] = {}
        queue = deque((t, 0) for t in targets)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for src in self.reverse.get(node, ()):
                if src not in seen or seen[src] > depth + 1:
                    seen[src] = depth + 1
                    queue.append((src, depth + 1))
        return seen
