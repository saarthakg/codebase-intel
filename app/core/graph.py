import pickle
from collections import deque
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
            if node == start:
                for neighbor in graph.successors(node):
                    if neighbor not in visited and d + 1 <= depth:
                        visited[neighbor] = d + 1
                        queue.append((neighbor, d + 1))
            else:
                if d < visited.get(node, d + 1):
                    visited[node] = d
                for neighbor in graph.successors(node):
                    if neighbor not in visited and d + 1 <= depth:
                        visited[neighbor] = d + 1
                        queue.append((neighbor, d + 1))
        return sorted(
            [{"file": f, "depth": d} for f, d in visited.items()],
            key=lambda x: x["depth"],
        )

    def files_referencing_symbol(
        self, symbol: str, metadata_store: "MetadataStore", repo_id: str
    ) -> list[str]:
        """Return file paths that contain this symbol name in their chunks."""
        results = metadata_store.find_symbol(repo_id, symbol)
        return list({r["file_path"] for r in results})

    def save(self, path: str) -> None:
        """Pickle the graph to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.G, f)

    def load(self, path: str) -> None:
        """Load graph from disk."""
        with open(path, "rb") as f:
            self.G = pickle.load(f)

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()


def resolve_python_import(
    imported_module: str, source_file: str, repo_root: str
) -> str | None:
    """Resolve a Python import module name to an actual file path in the repo.

    Returns the resolved absolute path string, or None if external/unresolvable.
    """
    if not imported_module:
        return None

    # Strip relative dots for resolution (we handle is_relative separately)
    module = imported_module.lstrip(".")
    parts = module.replace(".", "/")
    source_dir = Path(source_file).parent
    repo = Path(repo_root)

    candidates = [
        source_dir / f"{parts}.py",
        source_dir / parts / "__init__.py",
        repo / f"{parts}.py",
        repo / parts / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def resolve_ts_import(
    imported_module: str, source_file: str, repo_root: str
) -> str | None:
    """Resolve a TypeScript/JS relative import to an actual file path.

    Only resolves relative imports (starting with './'  or '../').
    Returns None for node_modules or unresolvable paths.
    """
    if not imported_module.startswith("."):
        return None  # external package
    source_dir = Path(source_file).parent
    base = (source_dir / imported_module).resolve()
    for ext in (".ts", ".tsx", ".js", ".jsx"):
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
        candidate2 = base / f"index{ext}"
        if candidate2.exists():
            return str(candidate2)
    return None
