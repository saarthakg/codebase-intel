from fastapi import APIRouter, HTTPException

from app.core.answer import LLMCallError, LLMConfigError, answer_question
from app.models.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    from app.main import get_repo_state
    try:
        state = get_repo_state(request.repo_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        return answer_question(
            request.question, request.repo_id, state.faiss_store, state.metadata_store,
            top_k=request.top_k, use_cache=request.use_cache,
        )
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMCallError as e:
        raise HTTPException(status_code=502, detail=str(e))
