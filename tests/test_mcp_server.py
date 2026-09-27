import anyio
import numpy as np
import pytest
from mcp import Client

from app.core import paths
from app.main import _loaded_repos
from app.mcp_server import mcp


def _fake_embed(texts, backend=None, **kwargs):
    rng = [np.random.default_rng(abs(hash(t)) % (2**32)) for t in texts]
    return np.vstack([r.random(8) for r in rng]).astype(np.float32)


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "metadata")
    monkeypatch.setattr("app.core.pipeline.embed_texts", _fake_embed)
    monkeypatch.setattr("app.core.search.embed_query", lambda q, backend=None, **kw: _fake_embed([q]))
    monkeypatch.setattr("app.core.embeddings.embed_query", lambda q, backend=None, **kw: _fake_embed([q]))
    monkeypatch.delenv("CODEBASE_INTEL_REPO_ID", raising=False)
    _loaded_repos.clear()
    yield tmp_path
    _loaded_repos.clear()


def _make_repo(root):
    repo = root / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "netrc.py").write_text(
        "def get_netrc_auth(url):\n    return None\n\n\ndef other():\n    return 1\n"
    )
    (repo / "pkg" / "session.py").write_text(
        "from pkg.netrc import get_netrc_auth\n\n\ndef request(url):\n    return get_netrc_auth(url)\n"
    )
    (repo / "tests" / "test_netrc.py").write_text("from pkg.netrc import get_netrc_auth\n")
    return repo


def call(tool, args=None):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(tool, args or {})
    return anyio.run(run)


def test_tools_and_annotations():
    async def run():
        async with Client(mcp) as client:
            return (await client.list_tools()).tools
    tools = {t.name: t for t in anyio.run(run)}
    assert set(tools) == {"list_repos", "search_code", "find_definition", "impact", "impact_of_diff", "ingest_repo"}
    assert all(tools[n].annotations.read_only_hint for n in tools if n != "ingest_repo")
    assert tools["ingest_repo"].annotations.read_only_hint is False


def test_no_repos_yet_tells_the_agent_to_ingest(data_dirs):
    r = call("search_code", {"query": "anything"})
    assert r.is_error and "Call ingest_repo first" in r.content[0].text


def test_ingest_then_query_end_to_end(data_dirs):
    repo = _make_repo(data_dirs)
    r = call("ingest_repo", {"repo_path": str(repo), "repo_id": "demo"})
    assert not r.is_error and r.structured_content["files_indexed"] >= 4

    assert call("list_repos").structured_content["repos"][0]["repo_id"] == "demo"

    # repo_id optional when only one repo is indexed
    found = call("search_code", {"query": "get_netrc_auth", "top_k": 1}).structured_content
    assert found["repo_id"] == "demo"
    assert found["results"][0]["file"] == "pkg/netrc.py" and "def get_netrc_auth" in found["results"][0]["code"]

    d = call("find_definition", {"symbol": "get_netrc_auth"}).structured_content
    assert (d["defined_in"], d["lines"]) == ("pkg/netrc.py", "1-2")
    assert {u["file"] for u in d["used_in"]} == {"pkg/session.py", "tests/test_netrc.py"}

    imp = call("impact", {"target": "pkg/netrc.py"}).structured_content
    assert imp["impacted"][0] == {"file": "tests/test_netrc.py", "confidence": 0.97,
                                  "reason": "test named for this file"}
    assert "pkg/session.py" in {i["file"] for i in imp["impacted"]}
    assert imp["tests_to_run"] == ["tests/test_netrc.py"]

    diff = "--- a/pkg/netrc.py\n+++ b/pkg/netrc.py\n@@ -2 +2 @@\n-    return None\n+    return ()\n"
    di = call("impact_of_diff", {"diff": diff}).structured_content
    assert di["changed_symbols"] == [{"file": "pkg/netrc.py", "symbol": "get_netrc_auth",
                                      "used_in": ["pkg/session.py", "tests/test_netrc.py"]}]


def test_errors_are_readable_by_the_agent(data_dirs):
    repo = _make_repo(data_dirs)
    call("ingest_repo", {"repo_path": str(repo), "repo_id": "one"})
    call("ingest_repo", {"repo_path": str(repo), "repo_id": "two"})

    r = call("search_code", {"query": "x"})
    assert r.is_error and "pass repo_id, one of: one, two" in r.content[0].text
    r = call("find_definition", {"symbol": "nope", "repo_id": "one"})
    assert r.is_error and "Try search_code" in r.content[0].text
    r = call("impact", {"target": "missing.py", "repo_id": "one"})
    assert r.is_error and "isn't a file or symbol" in r.content[0].text
    r = call("ingest_repo", {"repo_path": str(data_dirs / "nowhere"), "repo_id": "x"})
    assert r.is_error and "not a directory" in r.content[0].text
    r = call("impact_of_diff", {"diff": "   ", "repo_id": "one"})
    assert r.is_error and "diff is empty" in r.content[0].text
