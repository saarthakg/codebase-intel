import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from app.storage.metadata_store import MetadataStore


class DependencyGraph:
    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()

    def add_file(self, file_path: str) -> None:
        """Add a node for this file."""
        self.G.add_node(file_path)

    def add_import_edge(self, source_file: str, target_file: str) -> None:
        """source_file imports target_file. Edge: source → target."""
        self.G.add_edge(source_file, target_file)

    def dependents_of(self, file_path: str, depth: int = 3) -> list[dict]:
        """Return files that DEPEND ON file_path (reverse edges), up to `depth` hops.

        Returns list of {"file": str, "depth": int}, deduplicated, sorted by depth.
        """
        return self._bfs(file_path, reverse=True, depth=depth)

    def dependencies_of(self, file_path: str, depth: int = 3) -> list[dict]:
        """Return files that file_path DEPENDS ON (forward edges), up to `depth` hops."""
        return self._bfs(file_path, reverse=False, depth=depth)

    def _bfs(self, start: str, reverse: bool, depth: int) -> list[dict]:
        if start not in self.G:
            return []
        graph = self.G.reverse() if reverse else self.G
        visited: dict[str, int] = {}  # file → shallowest depth seen
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= depth:
                continue
            for neighbor in graph.successors(node):
                # Exclude `start` itself: with an import cycle (real code does
                # this — e.g. two modules importing each other), a path can
                # lead back to the start node, which would otherwise report a
                # file as one of its own dependents/dependencies. That's never
                # useful information, cycle or not.
                if neighbor == start or neighbor in visited:
                    continue
                visited[neighbor] = d + 1
                queue.append((neighbor, d + 1))
        return sorted(
            [{"file": f, "depth": d} for f, d in visited.items()],
            key=lambda x: x["depth"],
        )

    def files_referencing_symbol(
        self, symbol: str, metadata_store: "MetadataStore", repo_id: str
    ) -> list[str]:
        """Return file paths that use this symbol (identifier usages, not definitions)."""
        return sorted({r["file_path"] for r in metadata_store.find_references(repo_id, symbol)})

    def save(self, path: str) -> None:
        """Write the graph as JSON: {"nodes": [...], "edges": [[source, target], ...]}.

        JSON rather than pickle: loading a pickle executes whatever it
        contains, so a tampered or shared index file could run code.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"nodes": sorted(self.G.nodes), "edges": sorted([a, b] for a, b in self.G.edges)}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load a graph written by save()."""
        with open(path) as f:
            data = json.load(f)
        self.G = nx.DiGraph()
        self.G.add_nodes_from(data["nodes"])
        self.G.add_edges_from(tuple(e) for e in data["edges"])

    @classmethod
    def from_metadata(cls, metadata_store: "MetadataStore", repo_id: str) -> "DependencyGraph":
        """Rebuild the graph from the SQLite index (edges table + indexed files).

        Used for indexes written before the graph was stored as JSON, instead
        of unpickling their old .graph.pkl file.
        """
        graph = cls()
        graph.G.add_nodes_from(metadata_store.indexed_files(repo_id))
        for source, target in metadata_store.all_edges(repo_id):
            graph.add_import_edge(source, target)
        return graph

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()


# ── Import resolution ─────────────────────────────────────────────────────────

def find_python_source_roots(repo_root: str) -> list[Path]:
    """Directories that absolute Python imports resolve against.

    Always includes the repo root, plus the parent of every top-level package
    (a dir with __init__.py whose parent has none). That covers `src/` layouts
    (`src/requests/__init__.py` → root `src/`) and monorepos with several
    package roots, without needing to parse pyproject/setup.cfg.
    """
    from app.core.ingest import SKIP_DIRS

    repo = Path(repo_root).resolve()
    roots = [repo]
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        here = Path(dirpath)
        if "__init__.py" in filenames and here != repo and not (here.parent / "__init__.py").exists():
            if here.parent not in roots:
                roots.append(here.parent)
    return roots


def _python_module_file(base: Path) -> Path | None:
    """`pkg/mod` → pkg/mod.py or pkg/mod/__init__.py, whichever exists."""
    if base.name and (candidate := base.with_name(base.name + ".py")).is_file():
        return candidate
    if (candidate := base / "__init__.py").is_file():
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
        if module_file is not None and not module_file.is_file():
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


@dataclass
class TSConfig:
    """The bits of tsconfig.json/jsconfig.json that affect module resolution."""
    base_dir: Path                      # directory `paths` targets are relative to
    base_url: Path | None = None
    paths: dict[str, list[str]] = field(default_factory=dict)


def _strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas, leaving strings intact.

    tsconfig allows both, and a naive regex would eat path patterns like "@/*".
    """
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            i = text.find("\n", i)
            i = n if i == -1 else i
        elif text.startswith("/*", i):
            i = text.find("*/", i + 2)
            i = n if i == -1 else i + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def load_ts_config(repo_root: str) -> TSConfig | None:
    repo = Path(repo_root).resolve()
    for name in ("tsconfig.json", "jsconfig.json"):
        path = repo / name
        if not path.is_file():
            continue
        try:
            data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            return None
        options = data.get("compilerOptions") or {}
        base_url = options.get("baseUrl")
        base_url_path = (repo / base_url).resolve() if base_url else None
        return TSConfig(
            # Since TS 4.1, `paths` without `baseUrl` resolve relative to the tsconfig.
            base_dir=base_url_path or repo,
            base_url=base_url_path,
            paths={k: list(v) for k, v in (options.get("paths") or {}).items()},
        )
    return None


_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".d.ts")


def _ts_file_candidates(base: Path):
    # Append extensions rather than Path.with_suffix: with_suffix would turn
    # "./user.service" into "./user.ts".
    yield base
    for ext in _TS_EXTS:
        yield base.with_name(base.name + ext)
    # ESM-style imports name the compiled file: "./util.js" → util.ts on disk.
    if base.suffix in (".js", ".jsx", ".mjs", ".cjs"):
        stem = base.with_suffix("")
        for ext in (".ts", ".tsx", ".d.ts"):
            yield stem.with_name(stem.name + ext)
    for ext in _TS_EXTS:
        yield base / f"index{ext}"


def resolve_ts_import(
    imported_module: str,
    source_file: str,
    repo_root: str,
    ts_config: TSConfig | None = None,
) -> str | None:
    """Resolve a TypeScript/JS import to an in-repo file path.

    Handles relative imports ('./x', '../x') and, when a tsconfig/jsconfig is
    given, `paths` aliases (e.g. '@/components/*') and `baseUrl`-relative bare
    imports. Returns None for node_modules packages or unresolvable paths.
    """
    bases: list[Path] = []
    if imported_module.startswith("."):
        bases.append(Path(source_file).resolve().parent / imported_module)
    elif ts_config is not None:
        for pattern, targets in ts_config.paths.items():
            if "*" in pattern:
                prefix, _, suffix = pattern.partition("*")
                if imported_module.startswith(prefix) and imported_module.endswith(suffix) \
                        and len(imported_module) >= len(prefix) + len(suffix):
                    star = imported_module[len(prefix):len(imported_module) - len(suffix)]
                    bases += [ts_config.base_dir / t.replace("*", star) for t in targets]
            elif pattern == imported_module:
                bases += [ts_config.base_dir / t for t in targets]
        if ts_config.base_url is not None:
            bases.append(ts_config.base_url / imported_module)
    repo = Path(repo_root).resolve()
    for base in bases:
        for candidate in _ts_file_candidates(base):
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved.is_relative_to(repo):
                    return str(resolved)
    return None
