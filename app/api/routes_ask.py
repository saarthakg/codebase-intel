import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.answer import (
    LLMCallError,
    LLMConfigError,
    answer_question,
    check_llm_config,
    stream_answer_question,
)
from app.models.schemas import AskRequest, AskResponse
from app.state import get_repo_state

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    state = get_repo_state(request.repo_id)

    try:
        return answer_question(
            request.question, request.repo_id, state.faiss_store, state.metadata_store,
            top_k=request.top_k, use_cache=request.use_cache,
        )
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMCallError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/ask/stream")
def ask_stream(request: AskRequest):
    """Like POST /ask, but streams newline-delimited JSON events as the answer
    is generated: a `context` event (excerpts sent to the model), `delta`
    events with answer text, then an `answer` event with the full, checked
    AskResponse. Failures after the stream has started arrive as an `error`
    event: {"type": "error", "status": 502|503, "detail": "..."}.
    """
    state = get_repo_state(request.repo_id)
    try:
        check_llm_config()  # a missing key is a plain 503, not a stream error
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))

    def events():
        try:
            for event in stream_answer_question(
                request.question, request.repo_id, state.faiss_store, state.metadata_store,
                top_k=request.top_k, use_cache=request.use_cache,
            ):
                yield json.dumps(event) + "\n"
        except LLMConfigError as e:
            yield json.dumps({"type": "error", "status": 503, "detail": str(e)}) + "\n"
        except LLMCallError as e:
            yield json.dumps({"type": "error", "status": 502, "detail": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
