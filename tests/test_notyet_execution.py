"""The execution engine against real pytest runs in scratch repos."""
import json
import os
import shlex
import subprocess
import sys

import pytest

from notyet import config, gate, snapshot, store
from notyet.engines import execution
from notyet.findings import Context

_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t"}
PYTEST = f"{shlex.quote(sys.executable)} -m pytest"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_ENV).stdout


def _write(repo, rel, text):
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


@pytest.fixture
def repo(tmp_path):
    """calc (used by tests/test_calc.py) and shapes (uses calc; tested by test_shapes.py);
    strings is unrelated (tests/test_strings.py)."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "pkg/__init__.py", "")
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    _write(repo, "pkg/shapes.py", "from pkg.calc import add\n\n\ndef perimeter(w, h):\n    return add(add(w, h), add(w, h))\n")
    _write(repo, "pkg/strings.py", "def shout(s):\n    return s.upper()\n")
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    _write(repo, "tests/test_shapes.py", "from pkg.shapes import perimeter\n\n\ndef test_perimeter():\n"
                                         "    assert perimeter(2, 3) == 10\n")
    _write(repo, "tests/test_strings.py", "from pkg.strings import shout\n\n\ndef test_shout():\n    assert shout('a') == 'A'\n")
    _write(repo, ".notyet.toml", f'[test]\ncommand = {PYTEST!r}\nbudget_seconds = 60\n[gate]\nmode = "enforce"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def check(repo, baseline_tree=None):
    root = str(repo)
    session = store.Session(session_id="t", started=0, baseline_tree=baseline_tree or snapshot.head_tree(root),
                            baseline_head=snapshot.head_tree(root), baseline_source="session-start")
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, session.baseline_tree),
                  baseline_tree=session.baseline_tree, current_tree=current,
                  deltas=snapshot.diff_trees(root, session.baseline_tree, current))
    return execution.run(ctx)


def rules(result):
    return sorted((f.rule, f.severity, f.location) for f in result.findings)


def test_a_regression_blocks_and_only_relevant_tests_run(repo):
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n")
    result = check(repo)
    assert rules(result) == [("test-regression", "block", "tests/test_calc.py::test_add"),
                             ("test-regression", "block", "tests/test_shapes.py::test_perimeter")]
    assert "test_strings" not in result.checks[0]             # unrelated test file not run
    assert "ran 3 test(s) in 2 file(s)" in result.checks[0]    # found through pkg.shapes too
    regression = result.findings[0]
    assert any("assert" in e for e in regression.evidence)


def test_failures_that_existed_at_session_start_do_not_block(repo):
    _write(repo, "pkg/strings.py", "def shout(s):\n    return s.lower()\n")   # already broken...
    _git(repo, "commit", "-qam", "broken before the session")
    _write(repo, "pkg/strings.py", "def shout(s):\n    return s.lower()  # agent touched it\n")
    result = check(repo)
    assert rules(result) == [("test-failing-before", "note", "tests/test_strings.py::test_shout")]


def test_new_failing_test_blocks(repo):
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text()
           + "\n\ndef test_add_negative():\n    assert add(-1, -1) == -3\n")
    assert rules(check(repo)) == [("new-test-failing", "block", "tests/test_calc.py::test_add_negative")]


def test_flaky_failure_is_a_note(repo):
    counter = repo / "counter.txt"
    _write(repo, "tests/test_flaky.py", f"import pathlib\n\n\ndef test_sometimes():\n"
                                        f"    p = pathlib.Path({str(counter)!r})\n"
                                        f"    n = int(p.read_text()) if p.exists() else 0\n"
                                        f"    p.write_text(str(n + 1))\n    assert n % 2 == 1\n")
    assert rules(check(repo)) == [("test-flaky", "note", "tests/test_flaky.py::test_sometimes")]


def test_removed_test_blocks(repo):
    _write(repo, "tests/test_calc.py", "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    assert rules(check(repo)) == [("test-removed", "block", "tests/test_calc.py::test_sub")]


def test_added_test_that_pytest_never_collects(repo):
    _write(repo, "pytest.ini", "[pytest]\npython_functions = check_*\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "custom collection")
    _write(repo, "tests/test_calc.py", (repo / "tests/test_calc.py").read_text()
           + "\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n")
    found = rules(check(repo))
    assert ("test-not-collected", "resolve", "tests/test_calc.py::test_add_zero") in found


def test_docs_only_changes_run_nothing(repo):
    _write(repo, "README.md", "# hi\n")
    result = check(repo)
    assert result.findings == [] and "only documentation changed" in result.checks[0]


def test_no_relevant_tests_is_reported_not_passed(repo):
    _write(repo, "pkg/orphan.py", "def lonely():\n    return 1\n")
    result = check(repo)
    assert result.checks == [] and any("no tests exercise" in n for n in result.not_checked)


def test_budget_is_respected_and_reported(repo):
    _write(repo, "tests/test_slow.py", "import time\nfrom pkg import calc\n\n\ndef test_slow():\n    time.sleep(30)\n")
    _write(repo, ".notyet.toml", f'[test]\ncommand = {PYTEST!r}\nbudget_seconds = 3\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "slow suite")
    _write(repo, "tests/test_slow.py", "import time\nfrom pkg import calc\n\n\ndef test_slow():\n    time.sleep(30)  # touched\n")
    result = check(repo)
    assert any("budget ran out" in n for n in result.not_checked)


def test_baseline_that_imports_the_working_tree_is_not_trusted(repo):
    """Simulate an editable install: the interpreter puts the working tree
    ahead of everything else on sys.path, so a checkout of the session-start
    tree would silently import (and test) the new code."""
    code = f"import sys; sys.path.insert(0, {str(repo)!r}); import pytest; raise SystemExit(pytest.main())"
    forced = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    _write(repo, ".notyet.toml", f"[test]\ncommand = {json.dumps(forced)}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "forced path")
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n")
    result = check(repo)
    assert {f.severity for f in result.findings} == {"resolve"}       # failing, but not provably a regression
    assert any("imports the working tree's code" in n for n in result.not_checked)


def test_end_to_end_through_the_gate(repo):
    from notyet.hooks import claude
    payload = {"cwd": str(repo), "session_id": "e2e"}
    claude.handle("session-start", {**payload, "source": "startup"})
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n")
    out = claude.handle("stop", payload)
    assert out["decision"] == "block" and "passed at session start and fails now" in out["reason"]
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    _write(repo, "pkg/extra.py", "from pkg.calc import add\n\n\ndef twice(x):\n    return add(x, x)\n")
    out = claude.handle("stop", {**payload, "stop_hook_active": True})
    assert "decision" not in out
