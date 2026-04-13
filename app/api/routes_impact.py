from fastapi import APIRouter, HTTPException

import app.core.embeddings as embeddings_module
from app.core.impact import analyze_impact
from app.models.schemas import ImpactRequest, ImpactResponse

router = APIRouter()


@router.post("/impact", response_model=ImpactResponse)
async def impact(request: ImpactRequest):
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    response = analyze_impact(
        target=request.target,
        repo_id=request.repo_id,
        graph=state.graph,
        faiss_store=state.faiss_store,
        metadata_store=state.metadata_store,
        embeddings_module=embeddings_module,
        depth=request.depth,
    )
    return response
