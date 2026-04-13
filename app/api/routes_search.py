from fastapi import APIRouter, HTTPException, Query

from app.core.search import search_chunks
from app.models.schemas import DefinitionResponse, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
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
    )
    return SearchResponse(results=results)


@router.get("/definition", response_model=DefinitionResponse)
async def definition(
    repo_id: str = Query(...),
    symbol: str = Query(...),
):
    from app.main import get_repo_state
    try:
        state = get_repo_state(repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = state.metadata_store.find_symbol(repo_id, symbol)
    if not results:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found in repo '{repo_id}'")

    defining = next(
        (r for r in results if r["kind"] in ("function", "class", "method")),
        results[0],
    )
    refs = state.graph.files_referencing_symbol(symbol, state.metadata_store, repo_id)
    return DefinitionResponse(
        symbol=symbol,
        defining_file=defining["file_path"],
        start_line=defining.get("start_line"),
        references=refs,
    )
