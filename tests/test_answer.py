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


# ── Streaming ─────────────────────────────────────────────────────────────────

import contextlib
import json as _json

from app.core.answer import stream_answer


def _fake_stream_response(lines, status=200):
    resp = MagicMock(status_code=status)
    resp.iter_lines.return_value = iter(lines)

    @contextlib.contextmanager
    def cm(*args, **kwargs):
        cm.kwargs = kwargs
        cm.args = args
        yield resp
    return cm


def test_stream_answer_event_sequence_and_cache(tmp_path, monkeypatch):
    from app.storage.metadata_store import MetadataStore
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    store = MetadataStore(str(tmp_path / "m.db"))
    chunk = _chunk_with("a.py", "def foo():\n    return 1\n")
    with patch("app.core.answer.stream_llm", return_value=iter(["foo ", "returns 1 [1]."])) as llm:
        events = list(stream_answer("what does foo return?", [chunk], "r", store))
    assert [e["type"] for e in events] == ["context", "delta", "delta", "answer"]
    assert events[0]["excerpts"][0]["file_path"] == "a.py"
    final = events[-1]["response"]
    assert final["answer"] == "foo returns 1 [1]." and final["citations"][0]["file_path"] == "a.py"

    # Same question again: served from cache, no deltas, no LLM call
    with patch("app.core.answer.stream_llm") as llm:
        again = list(stream_answer("what does foo return?", [chunk], "r", store))
    llm.assert_not_called()
    assert [e["type"] for e in again] == ["context", "answer"]
    assert again[-1]["response"]["cached"] is True
    # non-streaming path shares the cache
    assert generate_answer("what does foo return?", [chunk], "r", store).cached


def test_ollama_stream_parses_ndjson():
    lines = [
        _json.dumps({"message": {"content": "Hel"}, "done": False}),
        "",
        _json.dumps({"message": {"content": "lo"}, "done": False}),
        _json.dumps({"message": {"content": ""}, "done": True}),
    ]
    fake = _fake_stream_response(lines)
    with patch("httpx.stream", fake):
        assert "".join(answer._stream_ollama("p", "m")) == "Hello"
    assert fake.kwargs["json"]["stream"] is True


def test_ollama_stream_error_event_raises():
    fake = _fake_stream_response([_json.dumps({"error": "out of memory"})])
    with patch("httpx.stream", fake):
        with pytest.raises(LLMCallError, match="out of memory"):
            list(answer._stream_ollama("p", "m"))


def test_gemini_stream_parses_sse(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    chunk = lambda t: "data: " + _json.dumps({"candidates": [{"content": {"parts": [{"text": t}]}}]})
    lines = [chunk("Gem"), "", chunk("ini"), "data: " + _json.dumps({"candidates": [{"finishReason": "STOP"}]})]
    fake = _fake_stream_response(lines)
    with patch("httpx.stream", fake):
        assert "".join(answer._stream_gemini("p", "gemini-flash-latest")) == "Gemini"
    assert "secret" not in fake.args[1]
    assert fake.kwargs["headers"]["x-goog-api-key"] == "secret"


def test_anthropic_stream_yields_text_and_checks_refusal(monkeypatch):
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def fake_stream(stop_reason):
        stream = MagicMock(text_stream=iter(["An", "swer"]))
        stream.get_final_message.return_value = MagicMock(stop_reason=stop_reason)

        @contextlib.contextmanager
        def cm(self, **kwargs):
            fake_stream.kwargs = kwargs
            yield stream
        return cm

    with patch.object(anthropic.resources.messages.Messages, "stream", fake_stream("end_turn")):
        assert "".join(answer._stream_anthropic("p", "claude-sonnet-5")) == "Answer"
    assert "temperature" not in fake_stream.kwargs
    assert fake_stream.kwargs["output_config"] == {"effort": "low"}
    with patch.object(anthropic.resources.messages.Messages, "stream", fake_stream("refusal")):
        with pytest.raises(LLMCallError, match="declined"):
            list(answer._stream_anthropic("p", "claude-sonnet-5"))


def _ask_client(tmp_path, monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient
    from app.core import paths
    from app.main import _loaded_repos, app

    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    _loaded_repos.clear()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")
    fake = lambda texts, backend=None, **kw: np.ones((len(texts), 8), dtype=np.float32)
    monkeypatch.setattr("app.core.pipeline.embed_texts", fake)
    monkeypatch.setattr("app.core.search.embed_query", lambda q, backend=None, **kw: fake([q]))
    client = TestClient(app)
    assert client.post("/ingest", json={"repo_path": str(repo), "repo_id": "st"}).status_code == 200
    return client


def test_ask_stream_endpoint(tmp_path, monkeypatch):
    client = _ask_client(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    with patch("app.core.answer.stream_llm", return_value=iter(["foo returns 1 [1]"])):
        r = client.post("/ask/stream", json={"repo_id": "st", "question": "what does foo return?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = [_json.loads(line) for line in r.text.splitlines()]
    assert [e["type"] for e in events] == ["context", "delta", "answer"]


def test_ask_stream_config_errors(tmp_path, monkeypatch):
    client = _ask_client(tmp_path, monkeypatch)
    # missing key: rejected before streaming starts
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/ask/stream", json={"repo_id": "st", "question": "q"})
    assert r.status_code == 503
    # Ollama unreachable: only discoverable mid-stream → error event
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    with patch("app.core.answer.stream_llm", side_effect=LLMConfigError("Can't reach Ollama")):
        r = client.post("/ask/stream", json={"repo_id": "st", "question": "q"})
    events = [_json.loads(line) for line in r.text.splitlines()]
    assert events[-1] == {"type": "error", "status": 503, "detail": "Can't reach Ollama"}



# ── Citations by file path (local models often skip [N]) ─────────────────────

from app.core.answer import Excerpt, path_citations


def _ex(path, start, end):
    return Excerpt(file_path=path, start_line=start, end_line=end, content="x\n")


def test_path_citations_match_file_and_lines():
    excerpts = [
        _ex("src/requests/sessions.py", 140, 190),   # 1
        _ex("src/requests/sessions.py", 300, 340),   # 2
        _ex("HISTORY.md", 1, 50),                    # 3
        _ex("src/requests/adapters.py", 1, 40),      # 4
    ]
    # Verbatim shape of a qwen2.5-coder:7b answer that the [N]-only parser missed
    answer = (
        "It happens in the `should_strip_auth` method, which is defined in "
        "`src/requests/sessions.py` (lines 154–184). This is described in the `HISTORY.md` file."
    )
    assert path_citations(answer, excerpts) == [1, 3]      # lines pick excerpt 1, not 2
    assert path_citations("see sessions.py:310-320", excerpts) == [2]  # suffix + colon form
    assert path_citations("see sessions.py", excerpts) == [1, 2]       # no lines: whole file
    assert path_citations("see sessions.py (line 999)", excerpts) == []  # lines outside every excerpt
    assert path_citations("see src/other.py", excerpts) == []


def test_answer_citing_by_path_is_not_flagged(tmp_path, monkeypatch):
    from app.storage.metadata_store import MetadataStore
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    store = MetadataStore(str(tmp_path / "m.db"))
    chunk = _chunk_with("src/pkg/auth.py", "def strip():\n    return True\n")
    answer_text = "`strip` in `src/pkg/auth.py` (lines 1–2) returns True."
    with patch("app.core.answer.call_llm", return_value=answer_text):
        response = generate_answer("what does strip return?", [chunk], "r", store)
    assert [(c.file_path, c.relevance) for c in response.citations] == [
        ("src/pkg/auth.py", "Referenced by file path in the answer")
    ]
    assert response.uncertainty is None


# ── Answer checks: fewer false alarms (from the /ask eval) ────────────────────

from app.core.answer import symbol_citations


def _store_with(tmp_path, path, content):
    from app.storage.metadata_store import MetadataStore
    store = MetadataStore(str(tmp_path / "m.db"))
    store.add_chunks([_chunk_with(path, content)], "r")
    return store


def test_example_code_placeholders_are_not_unverified(tmp_path):
    """Seen in the eval: an answer's own example defined `my_hook_function`,
    `MY_ENV_VAR` and `class MyAuth`, and the prose then referred to them."""
    store = _store_with(tmp_path, "src/pkg/hooks.py", "def register_hook(event, hook):\n    pass\n")
    excerpts, _ = assemble_context(store.get_chunks_by_file("r", "src/pkg/hooks.py"))
    answer = (
        "Use `register_hook` [1]:\n\n```python\nclass MyAuth(AuthBase):\n    pass\n\n"
        "def my_hook_function(r):\n    os.environ['MY_ENV_VAR'] = 'x'\n\nregister_hook('response', my_hook_function)\n```\n\n"
        "Here `my_hook_function` runs after `MyAuth` and reads `MY_ENV_VAR`. "
        "See https://github.com/pallets/flask/blob/main/src/flask/templating.py for more."
    )
    assert unverified_mentions(answer, excerpts, store, "r") == []


def test_names_used_elsewhere_in_repo_code_are_known(tmp_path):
    store = _store_with(tmp_path, "src/pkg/url.py", "def url_for(endpoint, _external=False, section=None):\n    pass\n")
    other = _chunk_with("src/pkg/other.py", "x = 1\n")
    excerpts, _ = assemble_context([other])
    assert unverified_mentions("Pass `section` to `url_for` [1].", excerpts, store, "r") == []
    assert unverified_mentions("Then call `retry_with_backoff` [1].", excerpts, store, "r") == ["retry_with_backoff"]


def test_naming_a_defined_symbol_cites_its_excerpt():
    a = Excerpt("src/app.py", 1, 40, "...", symbols=["Scaffold.add_url_rule", "Scaffold.route"])
    b = Excerpt("src/views.py", 1, 20, "...", symbols=["View.dispatch_request", "View.send"])
    c = Excerpt("src/net.py", 1, 20, "...", symbols=["Conn.send"])
    assert symbol_citations("It's handled by `Scaffold.add_url_rule`.", [a, b, c]) == [1]
    assert symbol_citations("The dispatch_request method does it.", [a, b, c]) == [2]
    assert symbol_citations("It calls `send`.", [a, b, c]) == []            # ambiguous: b and c
    assert symbol_citations("```python\nScaffold.route\n```", [a, b, c]) == []  # only inside example code


def test_hedges_count_only_when_about_the_evidence():
    from app.core.answer import _uncertainty

    describes_code = "It returns True when the Location header is not present and the status is 308 [1]."
    assert _uncertainty(describes_code, [1], [], []) is None
    hedge = "The provided excerpts are insufficient to say where retries happen [1]."
    assert "insufficient" in _uncertainty(hedge, [1], [], [])
