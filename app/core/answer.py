import os
import re

from app.models.schemas import AskResponse, Citation, ChunkMetadata

# Overridable via env so this doesn't need a code change when a new model ships.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

SYSTEM_PROMPT = """You are a codebase assistant. You answer questions about source code \
using ONLY the provided code excerpts. You must cite the specific files and line ranges \
that support your answer. If the evidence is insufficient, say so explicitly. \
Never invent code, function names, or behavior not present in the excerpts."""

# Upper bound per excerpt, as a guard against pathological chunks. Chunks are
# ~1600 chars today; this used to be a hard 800-char cut, which silently
# dropped half of every retrieved chunk before the model ever saw it.
MAX_EXCERPT_CHARS = 4000


class LLMConfigError(RuntimeError):
    """The LLM backend isn't configured (e.g. missing API key)."""


class LLMCallError(RuntimeError):
    """The LLM provider returned an error or an unusable response."""


_CITATION_RE = re.compile(r'\[(\d+)\]')
_UNCERTAINTY_PHRASES = (
    "insufficient", "unclear", "cannot determine", "not enough",
    "don't have enough", "no evidence", "not shown", "not present",
)


def build_prompt(question: str, chunks: list[ChunkMetadata]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        block = (
            f"[{i + 1}] File: {chunk.file_path} (lines {chunk.start_line}–{chunk.end_line})\n"
            f"```\n{chunk.content[:MAX_EXCERPT_CHARS]}\n```"
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)
    return (
        f"Code excerpts from the repository:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based strictly on the excerpts above. Cite by [N] number."
    )


def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMConfigError(
            f"{env_var} is not set. /ask needs an LLM API key — set it in .env "
            f"(search, definition and impact work without one)."
        )
    return key


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    try:
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        raise LLMCallError(f"Anthropic API error: {e}") from e
    text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
    if not text:
        raise LLMCallError("Anthropic returned no text content.")
    return text


def _call_gemini(prompt: str) -> str:
    import httpx
    gemini_key = _require_key("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1000},
    }
    try:
        # Key goes in a header, not the query string, so it can't leak into
        # proxy/access logs or exception messages that include the URL.
        resp = httpx.post(url, json=body, headers={"x-goog-api-key": gemini_key}, timeout=60)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise LLMCallError(f"Gemini API error: HTTP {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise LLMCallError(f"Gemini request failed: {type(e).__name__}") from e
    try:
        data = resp.json()
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (ValueError, KeyError, IndexError) as e:
        raise LLMCallError("Gemini returned no answer (the response may have been blocked).") from e


def generate_answer(
    question: str,
    retrieved_chunks: list[ChunkMetadata],
    repo_id: str,
) -> AskResponse:
    if not retrieved_chunks:
        return AskResponse(
            answer="No relevant code was found for this question.",
            citations=[],
            uncertainty="No code chunks were retrieved to answer from.",
        )

    prompt = build_prompt(question, retrieved_chunks)
    backend = os.environ.get("LLM_BACKEND", "anthropic").lower()

    if backend == "gemini":
        answer_text = _call_gemini(prompt)
    else:
        answer_text = _call_anthropic(prompt)

    # Parse [N] citation references
    cited_indices = set()
    for m in _CITATION_RE.finditer(answer_text):
        idx = int(m.group(1)) - 1  # 0-based
        if 0 <= idx < len(retrieved_chunks):
            cited_indices.add(idx)

    citations: list[Citation] = []
    for idx in sorted(cited_indices):
        chunk = retrieved_chunks[idx]
        citations.append(
            Citation(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                relevance=f"Cited as [{idx + 1}] in the answer",
            )
        )

    answer_lower = answer_text.lower()
    uncertainty: str | None = None
    for phrase in _UNCERTAINTY_PHRASES:
        if phrase in answer_lower:
            uncertainty = "The answer may be incomplete — the relevant code may not have been retrieved."
            break

    return AskResponse(
        answer=answer_text,
        citations=citations,
        uncertainty=uncertainty,
    )
