from fastapi import APIRouter, HTTPException, Query

from app.core.definitions import lookup_definition
from app.core.search import search_chunks
from app.core.validation import validate_repo_id
from app.models.schemas import DefinitionResponse, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = search_chunks(
        query=request.query,
        repo_id=request.repo_id,
        top_k=request.top_k,
        faiss_store=state.faiss_store,
        metadata_store=state.metadata_store,
        mode=request.mode,
    )
    return SearchResponse(results=results)


@router.get("/definition", response_model=DefinitionResponse)
def definition(
    repo_id: str = Query(...),
    symbol: str = Query(...),
):
    try:
        validate_repo_id(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from app.main import get_repo_state
    try:
        state = get_repo_state(repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    response = lookup_definition(symbol, state.metadata_store, repo_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in repo '{repo_id}'")
    return response
