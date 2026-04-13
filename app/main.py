from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.api import routes_ingest, routes_search, routes_impact, routes_ask
from app.core.graph import DependencyGraph
from app.storage.faiss_store import FAISSStore
from app.storage.metadata_store import MetadataStore

DATA_INDEXES = Path(__file__).parent.parent / "data" / "indexes"
DATA_METADATA = Path(__file__).parent.parent / "data" / "metadata"


@dataclass
class RepoState:
    faiss_store: FAISSStore
    metadata_store: MetadataStore
    graph: DependencyGraph


_loaded_repos: dict[str, RepoState] = {}


def get_repo_state(repo_id: str) -> RepoState:
    if repo_id in _loaded_repos:
        return _loaded_repos[repo_id]

    index_path = str(DATA_INDEXES / f"{repo_id}.index")
    db_path = str(DATA_METADATA / f"{repo_id}.db")
    graph_path = str(DATA_METADATA / f"{repo_id}.graph.pkl")

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No index found for repo '{repo_id}'. Run POST /ingest first."
        )

    faiss_store = FAISSStore(dim=384)  # dim overwritten by load
    faiss_store.load(index_path)

    metadata_store = MetadataStore(db_path)

    graph = DependencyGraph()
    if Path(graph_path).exists():
        graph.load(graph_path)

    state = RepoState(
        faiss_store=faiss_store,
        metadata_store=metadata_store,
        graph=graph,
    )
    _loaded_repos[repo_id] = state
    return state


app = FastAPI(
    title="codebase-intel",
    description="AI-powered codebase intelligence: semantic search, symbol lookup, dependency-aware impact analysis, and grounded repository Q&A.",
    version="1.0.0",
)

app.include_router(routes_ingest.router)
app.include_router(routes_search.router)
app.include_router(routes_impact.router)
app.include_router(routes_ask.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
