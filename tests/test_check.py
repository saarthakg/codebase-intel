import json
import os
import subprocess

import pytest

from codebase_intel.core import paths
from codebase_intel.core.check import check_change
from codebase_intel.core.workspace import NotAGitRepo, ensure_index, indexed_head, repo_id_for
from codebase_intel.state import _loaded_repos

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_ENV).stdout


def _write(repo, rel, text):
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)


@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_METADATA", tmp_path / "data")
    _loaded_repos.clear()
    yield
    _loaded_repos.clear()


@pytest.fixture
def repo(tmp_path):
    """pkg/api.py is always changed together with docs/api.md and its test;
    pkg/client.py calls api.fetch."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/api.py", "def fetch(url):\n    return url\n\n\ndef other():\n    return 1\n")
    _write(repo, "pkg/client.py", "from pkg.api import fetch\n\n\ndef get(u):\n    return fetch(u)\n")
    _write(repo, "docs/api.md", "# fetch\n")
    _write(repo, "tests/test_api.py", "from pkg.api import fetch\n")
    _commit(repo)
    for i in range(4):
        _write(repo, "pkg/api.py", f"def fetch(url):\n    return url  # v{i}\n\n\ndef other():\n    return 1\n")
        _write(repo, "docs/api.md", f"# fetch v{i}\n")
        _write(repo, "tests/test_api.py", f"from pkg.api import fetch  # v{i}\n")
        _commit(repo, f"v{i}")
    return repo


def test_index_holds_head_and_is_rebuilt_only_when_head_moves(repo):
    root, repo_id, state = ensure_index(str(repo / "pkg"))  # any path inside the repo
    assert root == str(repo.resolve()) and repo_id == repo_id_for(root)
    assert indexed_head(repo_id) == _git(repo, "rev-parse", "HEAD").strip()
    assert state.metadata_store.find_symbol(repo_id, "fetch")

    # Uncommitted edits and new files don't touch the index
    _write(repo, "pkg/api.py", "def renamed(url):\n    return url\n")
    _write(repo, "pkg/new.py", "def brand_new():\n    pass\n")
    _, _, state = ensure_index(str(repo))
    assert state.metadata_store.find_symbol(repo_id, "fetch")
    assert "pkg/new.py" not in state.graph.G.nodes

    _commit(repo, "rename")
    _, _, state = ensure_index(str(repo))
    assert state.metadata_store.find_symbol(repo_id, "fetch") == []
    assert state.metadata_store.find_symbol(repo_id, "renamed")
    assert "pkg/new.py" in state.graph.G.nodes


def test_check_uncommitted_change(repo):
    _write(repo, "pkg/api.py", "def fetch(url):\n    return url.strip()\n\n\ndef other():\n    return 1\n")
    _write(repo, "pkg/extra.py", "x = 1\n")  # untracked
    r = check_change(str(repo))
    assert r.changed_files == ["pkg/api.py", "pkg/extra.py"]
    assert [(s.file, s.symbol, s.callers_outside_change) for s in r.changed_symbols] == [
        ("pkg/api.py", "fetch", ["pkg/client.py", "tests/test_api.py"])
    ]
    missing = {s.file: s for s in r.likely_missing}
    assert "docs/api.md" in missing and "tests/test_api.py" in missing
    assert missing["docs/api.md"].reasons[0] == "changed together with pkg/api.py in 5 of its 5 changes"
    assert missing["docs/api.md"].because_of == ["pkg/api.py"]
    assert "tests/test_api.py" in r.tests_to_run
    assert r.not_in_index == ["pkg/extra.py"]


def test_files_already_in_the_change_are_not_suggested(repo):
    _write(repo, "pkg/api.py", "def fetch(url):\n    return 2\n\n\ndef other():\n    return 1\n")
    _write(repo, "docs/api.md", "# changed too\n")
    r = check_change(str(repo))
    assert "docs/api.md" not in {s.file for s in r.likely_missing}
    assert "tests/test_api.py" in {s.file for s in r.likely_missing}


def test_pure_insertion_inside_a_function_is_attributed_to_it(repo):
    _write(repo, "pkg/api.py", "def fetch(url):\n    return url  # v3\n\n\ndef other():\n    y = 2\n    return 1\n")
    assert [s.symbol for s in check_change(str(repo)).changed_symbols] == ["other"]


def test_staged_only(repo):
    _write(repo, "pkg/api.py", "def fetch(url):\n    return 0\n\n\ndef other():\n    return 1\n")
    _write(repo, "pkg/client.py", "from pkg.api import fetch\n\n\ndef get(u):\n    return fetch(u) or 1\n")
    _git(repo, "add", "pkg/client.py")
    r = check_change(str(repo), staged=True)
    assert r.changed_files == ["pkg/client.py"]


def test_branch_against_base(repo):
    _git(repo, "checkout", "-qb", "feature")
    _write(repo, "pkg/api.py", "def fetch(url):\n    return url or ''\n\n\ndef other():\n    return 1\n")
    _commit(repo, "feature work")
    _write(repo, "pkg/client.py", "from pkg.api import fetch\n\n\ndef get(u):\n    return fetch(u)  # wip\n")
    r = check_change(str(repo), base="main")
    assert r.changed_files == ["pkg/api.py", "pkg/client.py"]
    assert ("pkg/api.py", "fetch") in {(s.file, s.symbol) for s in r.changed_symbols}   # committed part
    assert ("pkg/client.py", "get") in {(s.file, s.symbol) for s in r.changed_symbols}  # uncommitted part
    assert r.base == "main (merge base)"


def test_clean_tree_reports_nothing(repo):
    r = check_change(str(repo))
    assert r.changed_files == [] and r.likely_missing == []


def test_not_a_git_repo(tmp_path):
    with pytest.raises(NotAGitRepo):
        check_change(str(tmp_path))


def test_cli_check_text_json_and_exit_code(repo, capsys):
    from codebase_intel.cli import main
    _write(repo, "pkg/api.py", "def fetch(url):\n    return None\n\n\ndef other():\n    return 1\n")
    assert main(["check", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "Likely also needs changing:" in out and "docs/api.md" in out
    assert "fetch: pkg/client.py, tests/test_api.py" in out

    assert main(["check", str(repo), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["changed_files"] == ["pkg/api.py"]

    top = max(s["confidence"] for s in report["likely_missing"])
    assert main(["check", str(repo), "--fail-above", str(top)]) == 1
    assert main(["check", str(repo), "--fail-above", str(top + 0.001)]) == 0
    capsys.readouterr()
    assert main(["check", str(repo / ".." / "nowhere")]) == 2


def test_cli_impact(repo, capsys):
    from codebase_intel.cli import main
    assert main(["impact", "pkg/api.py", "--path", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "docs/api.md" in out and "changed together" in out
    assert main(["impact", "nope.py", "--path", str(repo)]) == 2
