from fastapi import APIRouter, HTTPException

import app.core.embeddings as embeddings_module
from app.core.diff_impact import analyze_diff
from app.core.impact import analyze_impact, analyze_impact_batch
from app.models.schemas import (
    DiffImpactResponse,
    ImpactBatchRequest,
    ImpactBatchResponse,
    ImpactDiffRequest,
    ImpactRequest,
    ImpactResponse,
)

router = APIRouter()


@router.post("/impact", response_model=ImpactResponse)
def impact(request: ImpactRequest):
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return analyze_impact(
        target=request.target,
        repo_id=request.repo_id,
        graph=state.graph,
        faiss_store=state.faiss_store,
        metadata_store=state.metadata_store,
        embeddings_module=embeddings_module,
        depth=request.depth,
        cochange=state.cochange,
    )


@router.post("/impact/batch", response_model=ImpactBatchResponse)
def impact_batch(request: ImpactBatchRequest):
    """Diff-aware impact analysis over several changed files/symbols at once.

    Pass the output of `git diff --name-only` as `targets` to see everything a
    set of changes is likely to affect, merged into one ranked result.
    """
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not request.targets:
        raise HTTPException(status_code=400, detail="targets must be non-empty")

    return analyze_impact_batch(
        targets=request.targets,
        repo_id=request.repo_id,
        graph=state.graph,
        faiss_store=state.faiss_store,
        metadata_store=state.metadata_store,
        embeddings_module=embeddings_module,
        depth=request.depth,
        cochange=state.cochange,
    )


@router.post("/impact/diff", response_model=DiffImpactResponse)
def impact_diff(request: ImpactDiffRequest):
    """Symbol-level impact of a unified diff (e.g. `git diff HEAD`).

    Maps changed lines to the innermost functions/classes they touch, then
    ranks files that use those symbols above other importers of the file.
    """
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not request.diff.strip():
        raise HTTPException(status_code=400, detail="diff must be non-empty")

    return analyze_diff(
        request.diff, request.repo_id, state.graph, state.faiss_store, state.metadata_store,
        embeddings_module, depth=request.depth, cochange=state.cochange,
    )
