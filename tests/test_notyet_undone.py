"""Undone work: earlier turns' changes and the user's edits removed later."""
import time

from notyet import config, snapshot, store
from notyet.engines import undone
from notyet.findings import Context
from tests.conftest import _write

FIXED = "def add(a, b):\n    if a is None or b is None:\n        raise ValueError('missing operand')\n    return a + b\n\n\n" \
        "def sub(a, b):\n    return a - b\n"


def session_for(repo):
    root = str(repo)
    head = snapshot.head_tree(root)
    return store.Session(session_id="u", started=0, baseline_tree=snapshot.snapshot(root), baseline_head=head,
                         baseline_source="session-start")


def new_turn(repo, session):
    session.turns.append({"at": time.time(), "tree": snapshot.snapshot(str(repo))})


def check(repo, session):
    root = str(repo)
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, session.baseline_tree),
                  baseline_tree=session.baseline_tree, current_tree=current,
                  deltas=snapshot.diff_trees(root, session.baseline_tree, current))
    return undone.run(ctx)


def test_reverting_an_earlier_turns_fix_is_reported(repo):
    session = session_for(repo)
    new_turn(repo, session)                       # turn 1: "handle missing operands"
    _write(repo, "pkg/calc.py", FIXED)
    new_turn(repo, session)                       # turn 2: "make add faster"
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    found = check(repo, session).findings
    assert [(f.rule, f.location) for f in found] == [("undone-work", "pkg/calc.py")]
    assert "from an earlier turn" in found[0].title
    assert found[0].evidence == ["if a is None or b is None:", "raise ValueError('missing operand')"]


def test_the_users_uncommitted_edits_are_protected(repo):
    _write(repo, "pkg/strings.py", "def shout(s):\n    return s.upper() + '!'  # user's edit\n")
    session = session_for(repo)
    new_turn(repo, session)
    _write(repo, "pkg/strings.py", "def shout(s):\n    return s.upper()\n")
    found = check(repo, session).findings
    assert [f.location for f in found] == ["pkg/strings.py"] and "your uncommitted edits" in found[0].title


def test_moving_code_or_reworking_this_turns_own_lines_is_not_undone(repo):
    session = session_for(repo)
    new_turn(repo, session)
    _write(repo, "pkg/calc.py", FIXED)
    new_turn(repo, session)
    # the check moves into a helper in another file: same lines, new place
    _write(repo, "pkg/checks.py", "def require(a, b):\n    if a is None or b is None:\n"
                                  "        raise ValueError('missing operand')\n")
    _write(repo, "pkg/calc.py", "from pkg.checks import require\n\n\ndef add(a, b):\n    require(a, b)\n"
                                "    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    assert check(repo, session).findings == []


def test_lines_added_and_removed_within_one_turn_are_not_undone(repo):
    session = session_for(repo)
    new_turn(repo, session)
    _write(repo, "pkg/calc.py", FIXED)
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n")
    _write(repo, "pkg/calc.py", FIXED.replace("missing operand", "operand is None"))
    assert check(repo, session).findings == []


def test_prompt_hook_records_the_tree(repo):
    from notyet.hooks import claude
    payload = {"cwd": str(repo), "session_id": "turns"}
    claude.handle("session-start", {**payload, "source": "startup"})
    claude.handle("prompt", {**payload, "prompt": "fix add"})
    _write(repo, "pkg/calc.py", FIXED)
    claude.handle("prompt", {**payload, "prompt": "now speed it up"})
    session = store.load_session(str(repo), "turns")
    assert len(session.turns) == 2 and session.turns[0]["tree"] != session.turns[1]["tree"]
    assert session.turns[1]["tree"] == snapshot.snapshot(str(repo))
