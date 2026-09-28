import os
import subprocess

import anyio
import pytest
from mcp import Client

from app.core import paths
from app.mcp_server import mcp
from app.state import _loaded_repos

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_INDEXES", tmp_path / "data" / "indexes")
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "data" / "metadata")
    monkeypatch.delenv("CODEBASE_INTEL_REPO", raising=False)
    _loaded_repos.clear()
    repo = tmp_path / "proj"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, env=_ENV)
    git("init", "-q")
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "netrc.py").write_text("def get_netrc_auth(url):\n    return None\n")
    (repo / "pkg" / "session.py").write_text(
        "from pkg.netrc import get_netrc_auth\n\n\ndef request(url):\n    return get_netrc_auth(url)\n")
    (repo / "tests" / "test_netrc.py").write_text("from pkg.netrc import get_netrc_auth\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    yield repo
    _loaded_repos.clear()


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
    assert set(tools) == {"check_change", "impact"}
    assert all(t.annotations.read_only_hint for t in tools.values())


def test_check_change_and_impact(repo):
    (repo / "pkg" / "netrc.py").write_text("def get_netrc_auth(url):\n    return ()\n")
    r = call("check_change", {"repo_path": str(repo)}).structured_content
    assert r["changed_files"] == ["pkg/netrc.py"]
    assert r["callers_of_changed_code"] == {"get_netrc_auth": ["pkg/session.py", "tests/test_netrc.py"]}
    assert "tests/test_netrc.py" in r["tests_to_run"]
    assert r["compared_to"] == "HEAD, uncommitted changes"

    imp = call("impact", {"target": "pkg/netrc.py", "repo_path": str(repo)}).structured_content
    assert imp["impacted"][0]["file"] == "tests/test_netrc.py"
    assert imp["impacted"][0]["why"][0] == "test named for this file"


def test_repo_defaults_to_env_or_cwd(repo, monkeypatch):
    monkeypatch.setenv("CODEBASE_INTEL_REPO", str(repo))
    assert call("check_change").structured_content["changed_files"] == []


def test_errors_are_readable_by_the_agent(repo, tmp_path):
    r = call("check_change", {"repo_path": str(tmp_path / "elsewhere")})
    assert r.is_error and "not inside a git repository" in r.content[0].text
    r = call("impact", {"target": "missing.py", "repo_path": str(repo)})
    assert r.is_error and "isn't a file or symbol" in r.content[0].text
