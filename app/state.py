"""Loaded per-repo state (metadata, import graph, co-change), cached per
process and shared by the CLI, the MCP server and the evals."""
from dataclasses import dataclass

from app.core import paths
from app.core.graph import DependencyGraph
from app.core.history import CoChange
from app.core.validation import validate_repo_id
from app.storage.metadata_store import MetadataStore


class RepoNotIndexed(FileNotFoundError):
    """No index for a repo_id: it was never indexed."""


@dataclass
class RepoState:
    metadata_store: MetadataStore
    graph: DependencyGraph
    cochange: CoChange


_loaded_repos: dict[str, RepoState] = {}


def load_graph(repo_id: str, metadata_store: MetadataStore) -> DependencyGraph:
    graph_path = paths.graph_path(repo_id)
    if graph_path.exists():
        graph = DependencyGraph()
        graph.load(str(graph_path))
        return graph
    # Older index (pickled graph, or none): rebuild from SQLite rather than
    # unpickling the old file.
    return DependencyGraph.from_metadata(metadata_store, repo_id)


def get_repo_state(repo_id: str) -> RepoState:
    validate_repo_id(repo_id)
    if repo_id in _loaded_repos:
        return _loaded_repos[repo_id]

    if not paths.graph_path(repo_id).exists():
        raise RepoNotIndexed(f"No index found for repo '{repo_id}'. Index it first.")

    metadata_store = MetadataStore(str(paths.db_path(repo_id)))

    graph = load_graph(repo_id, metadata_store)

    state = RepoState(
        metadata_store=metadata_store,
        graph=graph,
        cochange=metadata_store.load_cochange(repo_id),
    )
    _loaded_repos[repo_id] = state
    return state


def forget_repo(repo_id: str) -> None:
    """Drop a repo's cached state (after re-ingest or delete)."""
    _loaded_repos.pop(repo_id, None)
