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
        assert answer._call_gemini("prompt") == "ok [1]"
    url = post.call_args.args[0]
    assert "secret-key" not in url
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "secret-key"


def test_gemini_blocked_response_raises_call_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    fake = MagicMock()
    fake.json.return_value = {"candidates": [{"finishReason": "SAFETY"}]}
    with patch("httpx.post", return_value=fake):
        with pytest.raises(LLMCallError):
            answer._call_gemini("prompt")


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
