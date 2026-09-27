"""Grounded Q&A: retrieve code, build a bounded prompt, call an LLM, check the answer.

Cost controls, since /ask is the only part of codebase-intel that can cost money:
- LLM_BACKEND=ollama runs a local model — no key, no per-call cost.
- Answers are cached by the exact prompt (question + excerpt text + model), so
  repeating a question against unchanged code never calls the LLM again.
- The context is assembled under a character budget, with overlapping or
  adjacent chunks from one file merged so shared lines are sent once.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from app.core.text import looks_like_identifier
from app.models.schemas import AskResponse, ChunkMetadata, Citation

if TYPE_CHECKING:
    from app.storage.faiss_store import FAISSStore
    from app.storage.metadata_store import MetadataStore

SYSTEM_PROMPT = """You are a codebase assistant. You answer questions about source code \
using ONLY the provided code excerpts. You must cite the specific files and line ranges \
that support your answer. If the evidence is insufficient, say so explicitly. \
Never invent code, function names, or behavior not present in the excerpts."""

# Bump whenever SYSTEM_PROMPT, the prompt layout or answer post-processing
# changes, so cached answers produced under the old format aren't served.
PROMPT_VERSION = "2"

# Total characters of code excerpts per prompt (~3-4K tokens). Excerpts are
# added best-first until the next one no longer fits.
CONTEXT_BUDGET_CHARS = 12_000

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-flash-latest",
    "ollama": "qwen2.5-coder:7b",
}
_MODEL_ENV = {"anthropic": "ANTHROPIC_MODEL", "gemini": "GEMINI_MODEL", "ollama": "OLLAMA_MODEL"}


class LLMConfigError(RuntimeError):
    """The LLM backend isn't configured or reachable (missing key, Ollama not running...)."""


class LLMCallError(RuntimeError):
    """The LLM provider returned an error or an unusable response."""


def llm_settings() -> tuple[str, str]:
    """(backend, model) from LLM_BACKEND and <BACKEND>_MODEL, read at call time."""
    backend = os.environ.get("LLM_BACKEND", "anthropic").strip().lower()
    if backend not in DEFAULT_MODELS:
        raise LLMConfigError(
            f"LLM_BACKEND={backend!r} is not supported; use one of {sorted(DEFAULT_MODELS)}."
        )
    model = os.environ.get(_MODEL_ENV[backend], "").strip() or DEFAULT_MODELS[backend]
    return backend, model


# ── Context assembly ──────────────────────────────────────────────────────────

@dataclass
class Excerpt:
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_ids: list[str] = field(default_factory=list)


def _merge_file_chunks(chunks: list[ChunkMetadata]) -> list[Excerpt]:
    """Merge one file's chunks whose line ranges overlap or touch."""
    chunks = sorted(chunks, key=lambda c: c.start_line)
    groups: list[list[ChunkMetadata]] = []
    for chunk in chunks:
        if groups and chunk.start_line <= max(c.end_line for c in groups[-1]) + 1:
            groups[-1].append(chunk)
        else:
            groups.append([chunk])
    excerpts = []
    for group in groups:
        lines: dict[int, str] = {}
        for chunk in group:
            for offset, text in enumerate(chunk.content.splitlines(keepends=True)):
                lines.setdefault(chunk.start_line + offset, text)
        start, end = min(lines), max(lines)
        excerpts.append(Excerpt(
            file_path=group[0].file_path, start_line=start, end_line=end,
            content="".join(lines[i] for i in range(start, end + 1) if i in lines),
            chunk_ids=[c.chunk_id for c in group],
        ))
    return excerpts


def assemble_context(
    chunks: list[ChunkMetadata], budget_chars: int = CONTEXT_BUDGET_CHARS
) -> tuple[list[Excerpt], int]:
    """Turn ranked chunks into prompt excerpts. Returns (excerpts, n_omitted).

    Overlapping/adjacent chunks from the same file become one excerpt (so
    shared lines are sent once), excerpts keep the rank of their best chunk,
    and they're added best-first while they fit in `budget_chars`. The top
    excerpt is always included, even if it alone is over budget.
    """
    rank = {c.chunk_id: i for i, c in enumerate(chunks)}
    by_file: dict[str, list[ChunkMetadata]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.file_path, []).append(chunk)
    merged = [e for file_chunks in by_file.values() for e in _merge_file_chunks(file_chunks)]
    merged.sort(key=lambda e: min(rank[cid] for cid in e.chunk_ids))

    selected: list[Excerpt] = []
    used = 0
    for excerpt in merged:
        size = len(excerpt.content)
        if selected and used + size > budget_chars:
            continue  # a later, smaller excerpt may still fit
        selected.append(excerpt)
        used += size
    return selected, len(merged) - len(selected)


def build_prompt(question: str, excerpts: list) -> str:
    """Accepts Excerpts or ChunkMetadata (anything with path/lines/content)."""
    blocks = [
        f"[{i}] File: {e.file_path} (lines {e.start_line}–{e.end_line})\n```\n{e.content.rstrip()}\n```"
        for i, e in enumerate(excerpts, 1)
    ]
    return (
        "Code excerpts from the repository:\n" + "\n\n".join(blocks) + "\n\n"
        f"Question: {question}\n\n"
        "Answer based strictly on the excerpts above. Cite by [N] number."
    )


# ── LLM backends ──────────────────────────────────────────────────────────────

def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise LLMConfigError(
            f"{env_var} is not set. /ask needs an LLM: set {env_var} in .env, or use "
            f"LLM_BACKEND=ollama for a free local model (search, definition and impact "
            f"work without any LLM)."
        )
    return key


def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
    try:
        # Current Claude models think adaptively and count thinking against
        # max_tokens, so leave headroom (billing is for tokens actually used).
        # Low effort: grounded lookup over supplied excerpts needs little
        # deliberation, and it's the main cost lever. No `temperature` — it's
        # rejected by current models.
        message = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as e:
        raise LLMCallError("Could not reach the Anthropic API.") from e
    except anthropic.APIStatusError as e:
        raise LLMCallError(f"Anthropic API error: HTTP {e.status_code}: {e.message}") from e
    if message.stop_reason == "refusal":
        raise LLMCallError("The model declined to answer this question.")
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text:
        raise LLMCallError(f"Anthropic returned no text (stop_reason={message.stop_reason}).")
    return text


def _call_gemini(prompt: str, model: str) -> str:
    import httpx
    gemini_key = _require_key("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        # Gemini 2.5-family thinking tokens count against maxOutputTokens;
        # the old limit of 1000 could be spent before any answer text.
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0},
    }
    try:
        # Key goes in a header, not the query string, so it can't leak into
        # proxy/access logs or exception messages that include the URL.
        resp = httpx.post(url, json=body, headers={"x-goog-api-key": gemini_key}, timeout=120)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise LLMCallError(f"Gemini API error: HTTP {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise LLMCallError(f"Gemini request failed: {type(e).__name__}") from e
    try:
        data = resp.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (ValueError, KeyError, IndexError) as e:
        raise LLMCallError("Gemini returned no answer (the response may have been blocked).") from e
    if not text:
        raise LLMCallError("Gemini returned an empty answer.")
    return text


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_timeout() -> float:
    return float(os.environ.get("OLLAMA_TIMEOUT", "600"))


def _call_ollama(prompt: str, model: str) -> str:
    """Local model via Ollama's /api/chat. Free; needs `ollama serve` + `ollama pull`."""
    import httpx
    host = _ollama_host()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # Ollama's default context window is small enough that a full prompt
        # would be silently truncated from the front, dropping the excerpts.
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    try:
        resp = httpx.post(f"{host}/api/chat", json=body, timeout=_ollama_timeout())
    except httpx.ConnectError as e:
        raise LLMConfigError(
            f"Can't reach Ollama at {host}. Start it with `ollama serve`, "
            f"then `ollama pull {model}` (or set OLLAMA_HOST)."
        ) from e
    except httpx.TimeoutException as e:
        # Seen with a 13 GB model on a 16 GB machine: it doesn't fit in GPU
        # memory, spills onto the CPU, and a ~3K-token prompt takes 10+ minutes.
        raise LLMCallError(
            f"Ollama model '{model}' didn't answer within {_ollama_timeout():.0f}s. It may be too "
            f"large for this machine (check `ollama ps` for CPU offload); try a smaller model "
            f"such as qwen2.5-coder:7b, or raise OLLAMA_TIMEOUT."
        ) from e
    except httpx.HTTPError as e:
        raise LLMCallError(f"Ollama request failed: {type(e).__name__}") from e
    if resp.status_code == 404:
        raise LLMConfigError(f"Ollama model '{model}' isn't available. Run `ollama pull {model}`.")
    if resp.status_code != 200:
        raise LLMCallError(f"Ollama error: HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        text = resp.json()["message"]["content"]
    except (ValueError, KeyError) as e:
        raise LLMCallError("Ollama returned an unexpected response.") from e
    if not text.strip():
        raise LLMCallError("Ollama returned an empty answer.")
    return text


_BACKENDS = {"anthropic": _call_anthropic, "gemini": _call_gemini, "ollama": _call_ollama}


def call_llm(prompt: str, backend: str, model: str) -> str:
    return _BACKENDS[backend](prompt, model)


# ── Answer checks ─────────────────────────────────────────────────────────────

# [1], [1][2], [3, 10] — groups are common in model output
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_UNCERTAINTY_PHRASES = (
    "insufficient", "unclear", "cannot determine", "not enough",
    "don't have enough", "no evidence", "not shown", "not present",
)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,80})`")
_PATH_RE = re.compile(r"\b[\w./-]+\.(?:py|pyi|ts|tsx|js|jsx|mjs|cjs|md|json|toml|ya?ml|txt)\b")
_NAME_RE = re.compile(r"[A-Za-z_][\w.]*[\w]")
_IGNORED_NAMES = frozenset({"self", "cls", "None", "True", "False", "kwargs", "args"})


def parse_citations(answer: str, n_excerpts: int) -> tuple[list[int], list[int]]:
    """(valid 1-based excerpt numbers, out-of-range numbers), each sorted and unique."""
    valid, invalid = set(), set()
    for m in _CITATION_RE.finditer(answer):
        for num in (int(n) for n in m.group(1).split(",")):
            (valid if 1 <= num <= n_excerpts else invalid).add(num)
    return sorted(valid), sorted(invalid)


def _mentions(answer: str) -> list[str]:
    """Code names and file paths the answer asserts exist: identifiers in
    backticks, code-looking identifiers anywhere, and file paths."""
    found: list[str] = []
    for span in _BACKTICK_RE.findall(answer):
        if _PATH_RE.fullmatch(span.strip()):
            continue  # collected with the other paths below
        found += [n for n in _NAME_RE.findall(span.split("(")[0]) if len(n) > 1]
    found += [t for t in _NAME_RE.findall(_PATH_RE.sub(" ", answer)) if looks_like_identifier(t)]
    found += _PATH_RE.findall(answer)
    return [m for m in dict.fromkeys(found) if m not in _IGNORED_NAMES]


def unverified_mentions(
    answer: str, excerpts: list, metadata_store: "MetadataStore", repo_id: str
) -> list[str]:
    """Names/paths in the answer that appear neither in the excerpts it was
    given nor anywhere in the repo's symbol table or file list — the
    signature of an invented function or file."""
    context = "\n".join(f"{e.file_path}\n{e.content}" for e in excerpts)
    unknown = []
    for mention in _mentions(answer):
        if mention in context:
            continue
        if _PATH_RE.fullmatch(mention):
            if metadata_store.file_exists(repo_id, mention):
                continue
        else:
            parts = [p for p in mention.split(".") if p and p not in _IGNORED_NAMES]
            if parts and all(p in context or metadata_store.symbol_exists(repo_id, p) for p in parts):
                continue
        unknown.append(mention)
    return unknown


def _uncertainty(answer: str, cited: list[int], invalid: list[int], unknown: list[str]) -> Optional[str]:
    notes = []
    lower = answer.lower()
    if any(p in lower for p in _UNCERTAINTY_PHRASES):
        notes.append("the model says the evidence may be insufficient")
    if not cited:
        notes.append("the answer cites none of the excerpts")
    if invalid:
        notes.append(f"it cites excerpts that don't exist ({', '.join(map(str, invalid))})")
    if unknown:
        notes.append(f"it names things not found in the repo: {', '.join(unknown[:5])}")
    if not notes:
        return None
    return "Check this answer: " + "; ".join(notes) + "."


# ── Entry points ──────────────────────────────────────────────────────────────

def _cache_key(backend: str, model: str, prompt: str) -> str:
    payload = json.dumps(
        {"v": PROMPT_VERSION, "backend": backend, "model": model, "system": SYSTEM_PROMPT, "prompt": prompt},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_answer(
    question: str,
    retrieved_chunks: list[ChunkMetadata],
    repo_id: str,
    metadata_store: Optional["MetadataStore"] = None,
    use_cache: bool = True,
) -> AskResponse:
    """Answer `question` from already-retrieved chunks (best-first)."""
    question = " ".join(question.split())
    if not retrieved_chunks:
        return AskResponse(
            answer="No relevant code was found for this question.",
            citations=[],
            uncertainty="No code chunks were retrieved to answer from.",
        )

    backend, model = llm_settings()
    excerpts, omitted = assemble_context(retrieved_chunks)
    prompt = build_prompt(question, excerpts)

    key = _cache_key(backend, model, prompt)
    if use_cache and metadata_store is not None:
        hit = metadata_store.get_cached_answer(key)
        if hit is not None:
            return AskResponse(**{**hit, "cached": True})

    answer_text = call_llm(prompt, backend, model)

    cited, invalid = parse_citations(answer_text, len(excerpts))
    unknown = (
        unverified_mentions(answer_text, excerpts, metadata_store, repo_id)
        if metadata_store is not None else []
    )
    response = AskResponse(
        answer=answer_text,
        citations=[
            Citation(
                file_path=excerpts[n - 1].file_path,
                start_line=excerpts[n - 1].start_line,
                end_line=excerpts[n - 1].end_line,
                relevance=f"Cited as [{n}] in the answer",
            )
            for n in cited
        ],
        uncertainty=_uncertainty(answer_text, cited, invalid, unknown),
        unverified_mentions=unknown,
        backend=backend,
        model=model,
        excerpts_used=len(excerpts),
        excerpts_omitted=omitted,
        context_chars=sum(len(e.content) for e in excerpts),
    )
    if metadata_store is not None:
        metadata_store.put_cached_answer(key, response.model_dump(exclude={"cached"}))
    return response


def answer_question(
    question: str,
    repo_id: str,
    faiss_store: "FAISSStore",
    metadata_store: "MetadataStore",
    top_k: int = 8,
    use_cache: bool = True,
) -> AskResponse:
    """Retrieve with hybrid search, then answer. Shared by POST /ask and the CLI."""
    from app.core.search import search_chunks

    results = search_chunks(question, repo_id, top_k, faiss_store, metadata_store)
    chunks = [c for c in (metadata_store.get_chunk(r.chunk_id) for r in results) if c is not None]
    return generate_answer(question, chunks, repo_id, metadata_store, use_cache=use_cache)
