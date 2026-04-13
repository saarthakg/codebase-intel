from fastapi import APIRouter, HTTPException

from app.core.answer import generate_answer
from app.core.search import search_chunks
from app.models.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Semantic search for top_k chunks
    search_results = search_chunks(
        query=request.question,
        repo_id=request.repo_id,
        top_k=request.top_k,
        faiss_store=state.faiss_store,
        metadata_store=state.metadata_store,
    )
    chunk_metas = [state.metadata_store.get_chunk(r.chunk_id) for r in search_results]
    chunk_metas = [c for c in chunk_metas if c is not None]

    response = generate_answer(request.question, chunk_metas, request.repo_id)
    return response
