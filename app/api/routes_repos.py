import json

from fastapi import APIRouter, HTTPException

from app.core import paths
from app.core.validation import validate_repo_id
from app.models.schemas import DeleteRepoResponse, RepoInfo, RepoListResponse
from app.state import forget_repo

router = APIRouter()


@router.get("/repos", response_model=RepoListResponse)
def list_repos():
    """List every repo_id that has been ingested, with its last ingest stats."""
    repos: list[RepoInfo] = []
    for repo_id in paths.known_repo_ids():
        meta_file = paths.meta_path(repo_id)
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            repos.append(
                RepoInfo(
                    repo_id=repo_id,
                    files_indexed=meta.get("files_indexed", 0),
                    chunks_indexed=meta.get("chunks_indexed", 0),
                    symbols_extracted=meta.get("symbols_extracted", 0),
                    edges_in_graph=meta.get("edges_in_graph", 0),
                    embedding_backend=meta.get("embedding_backend"),
                    ingested_at=meta.get("ingested_at"),
                )
            )
        else:
            # Ingested before .meta.json existed, or the file was lost — still
            # list it so it's discoverable, just without historical stats.
            repos.append(
                RepoInfo(
                    repo_id=repo_id,
                    files_indexed=0,
                    chunks_indexed=0,
                    symbols_extracted=0,
                    edges_in_graph=0,
                )
            )
    return RepoListResponse(repos=repos)


@router.delete("/repos/{repo_id}", response_model=DeleteRepoResponse)
def delete_repo(repo_id: str):
    try:
        validate_repo_id(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    artifact_paths = paths.repo_artifact_paths(repo_id)
    existed = any(p.exists() for p in artifact_paths)
    if not existed:
        raise HTTPException(status_code=404, detail=f"No repo found with repo_id '{repo_id}'")

    for p in artifact_paths:
        p.unlink(missing_ok=True)

    forget_repo(repo_id)

    return DeleteRepoResponse(repo_id=repo_id, deleted=True)
