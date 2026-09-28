"""The static engine: linter/type-checker diffs against session start, and
added suppression comments."""
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from notyet import config, snapshot, store
from notyet.engines import static
from notyet.findings import Context

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}
BIN = os.path.dirname(sys.executable)
RUFF = os.path.join(BIN, "ruff") if os.path.exists(os.path.join(BIN, "ruff")) else shutil.which("ruff")
PYRIGHT = os.path.join(BIN, "pyright") if os.path.exists(os.path.join(BIN, "pyright")) else shutil.which("pyright")


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_ENV).stdout


def _write(repo, rel, text):
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "pyproject.toml", '[tool.ruff.lint]\nselect = ["F"]\n\n[tool.pyright]\ntypeCheckingMode = "basic"\n')
    _write(repo, "pkg/__init__.py", "")
    # one pre-existing lint problem (unused import) that must never be reported
    _write(repo, "pkg/calc.py", "import os\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n")
    _write(repo, "pkg/use.py", "from pkg.calc import add\n\n\ndef total() -> int:\n    return add(1, 2)\n")
    lines = [f"{name} = {json.dumps(shlex.quote(path))}" for name, path in (("ruff", RUFF), ("pyright", PYRIGHT)) if path]
    _write(repo, ".notyet.toml", "[static]\n" + "\n".join(lines) + "\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def check(repo):
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="t", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, base), baseline_tree=base,
                  current_tree=current, deltas=snapshot.diff_trees(root, base, current))
    return static.run(ctx)


def rules(result):
    return sorted((f.rule, f.location) for f in result.findings)


needs_ruff = pytest.mark.skipif(not RUFF, reason="ruff not installed")
needs_pyright = pytest.mark.skipif(not PYRIGHT, reason="pyright not installed")


@needs_ruff
def test_only_lint_problems_the_session_added_are_reported(repo):
    _write(repo, "pkg/calc.py", "import os\nimport sys\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n")
    result = check(repo)
    assert [f for f in rules(result) if f[0] == "lint-new"] == [("lint-new", "pkg/calc.py:2")]
    assert any(c.startswith("ruff: 1 new") for c in result.checks)


@needs_ruff
def test_moving_existing_problems_reports_nothing(repo):
    _write(repo, "pkg/calc.py", '"""Doc."""\n\nimport os\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n')
    assert [f for f in rules(check(repo)) if f[0] == "lint-new"] == []


@needs_ruff
def test_loosening_lint_config_mid_session_does_not_hide_problems(repo):
    _write(repo, "pyproject.toml", '[tool.ruff.lint]\nselect = []\n')
    _write(repo, "pkg/calc.py", "import os\nimport sys\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n")
    result = check(repo)
    assert ("lint-new", "pkg/calc.py:2") in rules(result)
    assert any("pyproject.toml" in a for a in result.advice)


@needs_pyright
def test_a_signature_change_that_breaks_a_caller_is_found_in_the_caller(repo):
    _write(repo, "pkg/calc.py", "import os\n\n\ndef add(a: int, b: int, c: int) -> int:\n    return a + b + c\n")
    result = check(repo)
    found = [f for f in result.findings if f.rule == "type-error-new"]
    assert [f.location for f in found] == ["pkg/use.py:5"]       # the caller, not the changed file
    assert "pkg/use.py" in found[0].title


def test_added_suppressions_are_reported_moved_ones_are_not(repo):
    _write(repo, "pkg/calc.py", "import os  # noqa: F401\n\n\ndef add(a: int, b: int) -> int:\n"
                                "    return a + b  # type: ignore\n")
    _git(repo, "commit", "-qam", "suppressions exist before")
    # move the existing noqa, add a new type: ignore and a pragma
    _write(repo, "pkg/calc.py", '"""Doc."""\nimport os  # noqa: F401\n\n\ndef add(a: int, b: int) -> int:\n'
                                "    x = a  # type: ignore\n    return x + b  # type: ignore\n\n\n"
                                "def unused():  # pragma: no cover\n    pass\n")
    result = check(repo)
    found = sorted((f.location, f.title.split("`")[1]) for f in result.findings if f.rule == "suppression-added")
    assert len(found) == 2
    assert ("pkg/calc.py:10", "# pragma: no cover") in found
    assert any(t == "# type: ignore" for _, t in found)


def test_no_tools_configured_still_checks_suppressions(repo):
    _write(repo, ".notyet.toml", "")
    _git(repo, "commit", "-qam", "no tools")
    _write(repo, "pkg/use.py", "from pkg.calc import add\n\n\ndef total() -> int:\n    return add(1, '2')  # type: ignore\n")
    result = check(repo)
    assert rules(result) == [("suppression-added", "pkg/use.py:5")] and result.checks == []
