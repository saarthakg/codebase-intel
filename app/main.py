from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api import routes_ask, routes_impact, routes_ingest, routes_repos, routes_search
from app.core import paths
# Re-exported: the MCP server, scripts, evals and tests import these from here.
from app.state import RepoNotIndexed, RepoState, _loaded_repos, get_repo_state, load_graph  # noqa: F401


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


@app.exception_handler(RepoNotIndexed)
async def repo_not_indexed(request: Request, exc: RepoNotIndexed):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


_STATIC = paths.PROJECT_ROOT / "app" / "static"


@app.get("/", include_in_schema=False)
def web_ui():
    """A small browser UI over the API (search, definition, impact, ask)."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
