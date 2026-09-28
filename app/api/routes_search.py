from fastapi import APIRouter, HTTPException, Query

from app.core.definitions import lookup_definition
from app.core.search import search_chunks
from app.core.validation import validate_repo_id
from app.models.schemas import DefinitionResponse, SearchRequest, SearchResponse
from app.state import get_repo_state

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    state = get_repo_state(request.repo_id)

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

    state = get_repo_state(repo_id)

    response = lookup_definition(symbol, state.metadata_store, repo_id, state.graph)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in repo '{repo_id}'")
    return response
