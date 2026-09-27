from unittest.mock import MagicMock, patch

import pytest

from app.core import answer
from app.core.answer import LLMCallError, LLMConfigError, build_prompt, generate_answer
from app.models.schemas import ChunkMetadata


def _chunk(content: str, file_path: str = "a.py") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="c1", file_path=file_path, language="python", start_line=1,
        end_line=content.count("\n") + 1, symbols=[], imports=[], content=content,
    )


def test_prompt_includes_full_chunk_content():
    """Chunks are ~1600 chars; the prompt used to cut each one at 800."""
    body = "x = 1\n" * 250 + "THE_IMPORTANT_TAIL = True\n"   # ~1500 chars
    prompt = build_prompt("q?", [_chunk(body)])
    assert "THE_IMPORTANT_TAIL" in prompt


def test_missing_key_raises_config_error(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        generate_answer("q?", [_chunk("def f(): pass\n")], "r")


def test_gemini_key_sent_as_header_not_in_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    fake = MagicMock()
    fake.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok [1]"}]}}]}
    with patch("httpx.post", return_value=fake) as post:
        assert answer._call_gemini("prompt", "gemini-flash-latest") == "ok [1]"
    url = post.call_args.args[0]
    assert "secret-key" not in url
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "secret-key"


def test_gemini_blocked_response_raises_call_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake = MagicMock()
    fake.json.return_value = {"candidates": [{"finishReason": "SAFETY"}]}
    with patch("httpx.post", return_value=fake):
        with pytest.raises(LLMCallError):
            answer._call_gemini("prompt", "gemini-flash-latest")


def test_ask_without_key_returns_503(tmp_path, monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient
    from app.core import paths
    from app.main import _loaded_repos, app

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")

    fake_embed = lambda texts, backend=None, **kw: np.ones((len(texts), 8), dtype=np.float32)
    client = TestClient(app)
    with patch("app.core.pipeline.embed_texts", side_effect=fake_embed), \
         patch("app.core.search.embed_query", side_effect=lambda q, backend=None, **kw: fake_embed([q])):
        assert client.post("/ingest", json={"repo_path": str(repo), "repo_id": "askrepo"}).status_code == 200
        r = client.post("/ask", json={"repo_id": "askrepo", "question": "what does foo do?"})
    _loaded_repos.clear()
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


# ── Context assembly ──────────────────────────────────────────────────────────

from app.core.answer import assemble_context, parse_citations, unverified_mentions


def _c(cid: str, path: str, start: int, n_lines: int, tag: str = "x") -> ChunkMetadata:
    content = "".join(f"{tag}{start + i}\n" for i in range(n_lines))
    return ChunkMetadata(chunk_id=cid, file_path=path, language="python", start_line=start,
                         end_line=start + n_lines - 1, symbols=[], imports=[], content=content)


def test_overlapping_and_adjacent_chunks_are_merged_once():
    excerpts, omitted = assemble_context([
        _c("a", "f.py", 1, 10),     # lines 1-10
        _c("b", "f.py", 8, 5),      # 8-12, overlaps a
        _c("c", "f.py", 13, 3),     # 13-15, adjacent to b
        _c("d", "f.py", 40, 3),     # separate
    ])
    assert omitted == 0
    assert [(e.start_line, e.end_line) for e in excerpts] == [(1, 15), (40, 42)]
    assert excerpts[0].content.splitlines() == [f"x{i}" for i in range(1, 16)]  # no duplicated lines


def test_excerpts_keep_rank_order_and_respect_budget():
    big = _c("big", "b.py", 1, 400)          # ~2.4K chars
    excerpts, omitted = assemble_context(
        [_c("top", "a.py", 1, 5), big, _c("small", "c.py", 1, 5)], budget_chars=500,
    )
    assert [e.file_path for e in excerpts] == ["a.py", "c.py"]  # big skipped, smaller later one kept
    assert omitted == 1


def test_top_excerpt_always_included_even_if_over_budget():
    excerpts, _ = assemble_context([_c("big", "b.py", 1, 400)], budget_chars=10)
    assert len(excerpts) == 1


# ── Answer checks ─────────────────────────────────────────────────────────────

def test_parse_citations_handles_groups_and_out_of_range():
    valid, invalid = parse_citations("See [1][2] and [3, 10], also [2].", n_excerpts=3)
    assert valid == [1, 2, 3]
    assert invalid == [10]


def test_unverified_mentions_flags_invented_names_only(tmp_path):
    from app.storage.metadata_store import MetadataStore
    store = MetadataStore(str(tmp_path / "m.db"))
    store.add_chunks([_chunk_with("src/pkg/adapters.py", "def send(self):\n    return self.pool.urlopen()\n")], "r")
    store.upsert_symbol("HTTPAdapter", "r", "src/pkg/adapters.py", 1, "class")
    excerpts, _ = assemble_context(store.get_chunks_by_file("r", "src/pkg/adapters.py"))
    answer = (
        "`HTTPAdapter.send` calls `self.pool.urlopen()` in adapters.py [1], "
        "then `retry_with_backoff` in `src/pkg/retry.py`."
    )
    assert unverified_mentions(answer, excerpts, store, "r") == ["retry_with_backoff", "src/pkg/retry.py"]


def _chunk_with(path: str, content: str) -> ChunkMetadata:
    return ChunkMetadata(chunk_id="k1", file_path=path, language="python", start_line=1,
                         end_line=content.count("\n"), symbols=[], imports=[], content=content)


# ── Cache ─────────────────────────────────────────────────────────────────────

def test_answers_are_cached_by_prompt_content(tmp_path, monkeypatch):
    from app.storage.metadata_store import MetadataStore
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    store = MetadataStore(str(tmp_path / "m.db"))
    chunk = _chunk_with("a.py", "def foo():\n    return 1\n")
    calls = []

    def fake_llm(prompt, backend, model):
        calls.append(prompt)
        return "foo returns 1 [1]."

    with patch("app.core.answer.call_llm", side_effect=fake_llm):
        first = generate_answer("what does foo return?", [chunk], "r", store)
        second = generate_answer("what  does foo return?", [chunk], "r", store)   # whitespace-normalized
        forced = generate_answer("what does foo return?", [chunk], "r", store, use_cache=False)
        changed = generate_answer("what does foo return?",
                                  [_chunk_with("a.py", "def foo():\n    return 2\n")], "r", store)
    assert (first.cached, second.cached, forced.cached, changed.cached) == (False, True, False, False)
    assert second.answer == first.answer and second.citations == first.citations
    assert len(calls) == 3  # the cache hit made no LLM call; changed code missed the cache


# ── Backends ──────────────────────────────────────────────────────────────────

def test_unknown_backend_is_a_config_error(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt")
    with pytest.raises(LLMConfigError, match="not supported"):
        generate_answer("q", [_chunk("x = 1\n")], "r")


def test_anthropic_request_uses_low_effort_and_no_temperature(monkeypatch):
    """Current Claude models reject `temperature`, and adaptive thinking counts
    against max_tokens — so no sampling params and real output headroom."""
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    message = MagicMock(stop_reason="end_turn", content=[MagicMock(type="text", text="ok [1]")])
    with patch.object(anthropic.resources.messages.Messages, "create", return_value=message) as create:
        assert answer._call_anthropic("prompt", "claude-sonnet-5") == "ok [1]"
    kwargs = create.call_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["max_tokens"] >= 8000
    assert kwargs["model"] == "claude-sonnet-5"


def test_anthropic_refusal_is_reported(monkeypatch):
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    message = MagicMock(stop_reason="refusal", content=[])
    with patch.object(anthropic.resources.messages.Messages, "create", return_value=message):
        with pytest.raises(LLMCallError, match="declined"):
            answer._call_anthropic("prompt", "claude-sonnet-5")


def test_ollama_success_and_request_shape(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.local:1234/")
    fake = MagicMock(status_code=200)
    fake.json.return_value = {"message": {"content": "local answer [1]"}}
    with patch("httpx.post", return_value=fake) as post:
        assert answer._call_ollama("prompt", "qwen2.5-coder:7b") == "local answer [1]"
    assert post.call_args.args[0] == "http://ollama.local:1234/api/chat"
    body = post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5-coder:7b" and body["stream"] is False
    assert body["options"]["num_ctx"] >= 8192  # default context would truncate the excerpts


def test_ollama_not_running_or_model_missing_are_config_errors():
    import httpx
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(LLMConfigError, match="ollama serve"):
            answer._call_ollama("prompt", "m")
    with patch("httpx.post", return_value=MagicMock(status_code=404, text="model not found")):
        with pytest.raises(LLMConfigError, match="ollama pull m"):
            answer._call_ollama("prompt", "m")


def test_ollama_timeout_explains_what_to_do(monkeypatch):
    import httpx
    monkeypatch.setenv("OLLAMA_TIMEOUT", "5")
    with patch("httpx.post", side_effect=httpx.ReadTimeout("slow")) as post:
        with pytest.raises(LLMCallError, match="too large for this machine"):
            answer._call_ollama("prompt", "big-model")
    assert post.call_args.kwargs["timeout"] == 5.0
