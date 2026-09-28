"""Shared fixtures: a small git repo with a package and its tests."""
import os
import shlex
import subprocess
import sys

import pytest

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
