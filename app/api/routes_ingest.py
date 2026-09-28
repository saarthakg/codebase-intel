from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.pipeline import IngestError, run_ingestion
from app.models.schemas import IngestRequest, IngestResponse
from app.state import forget_repo

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not Path(request.repo_path).expanduser().exists():
        raise HTTPException(
            status_code=400, detail=f"repo_path does not exist: {request.repo_path}"
        )

    try:
        summary = run_ingestion(request.repo_path, request.repo_id)
    except IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Invalidate any cached in-memory state for this repo_id — the artifacts
    # on disk it points to have just been replaced.
    forget_repo(request.repo_id)

    return IngestResponse(
        repo_id=summary["repo_id"],
        files_indexed=summary["files_indexed"],
        chunks_indexed=summary["chunks_indexed"],
        symbols_extracted=summary["symbols_extracted"],
        edges_in_graph=summary["edges_in_graph"],
        files_skipped=summary["files_skipped"],
        chunks_embedded=summary["chunks_embedded"],
        chunks_reused=summary["chunks_reused"],
    )
