from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.api import routes_ask, routes_impact, routes_ingest, routes_repos, routes_search
from app.core import paths
from app.core.graph import DependencyGraph
from app.core.history import CoChange
from app.core.validation import validate_repo_id
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore


@dataclass
class RepoState:
    faiss_store: FAISSStore
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

    index_path = paths.index_path(repo_id)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found for repo '{repo_id}'. Run POST /ingest first."
        )

    faiss_store = FAISSStore(dim=384)  # dim/backend overwritten by load
    faiss_store.load(str(index_path))

    metadata_store = MetadataStore(str(paths.db_path(repo_id)))

    graph = load_graph(repo_id, metadata_store)

    state = RepoState(
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        graph=graph,
        cochange=metadata_store.load_cochange(repo_id),
    )
    _loaded_repos[repo_id] = state
    return state


app = FastAPI(
    title="codebase-intel",
    description="AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis, and grounded repository Q&A.",
    version="1.1.0",
)

app.include_router(routes_ingest.router)
app.include_router(routes_search.router)
app.include_router(routes_impact.router)
app.include_router(routes_ask.router)
app.include_router(routes_repos.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
