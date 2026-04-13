import os
import re

from app.models.schemas import AskResponse, Citation, ChunkMetadata

ANTHROPIC_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

SYSTEM_PROMPT = """You are a codebase assistant. You answer questions about source code \
using ONLY the provided code excerpts. You must cite the specific files and line ranges \
that support your answer. If the evidence is insufficient, say so explicitly. \
Never invent code, function names, or behavior not present in the excerpts."""

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
            f"```\n{chunk.content[:800]}\n```"
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)
    return (
        f"Code excerpts from the repository:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer based strictly on the excerpts above. Cite by [N] number."
    )


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_gemini(prompt: str) -> str:
    import openai
    client = openai.OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url=GEMINI_BASE_URL,
    )
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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
