"""The integrity engine: tests weakened instead of code fixed."""
from notyet.engines import integrity
from tests.conftest import _write


def check(repo):
    from notyet import config, snapshot, store
    from notyet.findings import Context
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="t", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    current = snapshot.snapshot(root)
    ctx = Context(root=root, session=session, config=config.load(root, base), baseline_tree=base,
                  current_tree=current, deltas=snapshot.diff_trees(root, base, current))
    return integrity.run(ctx)


def rules(result):
    return sorted((f.rule, f.severity, f.location) for f in result.findings)


BROKEN_ADD = "def add(a, b):\n    return a + b + 1\n\n\ndef sub(a, b):\n    return a - b\n"


def test_skipping_a_failing_test_blocks(repo):
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", "import pytest\nfrom pkg.calc import add, sub\n\n\n@pytest.mark.skip(reason='flaky')\n"
                                       "def test_add():\n    assert add(1, 2) == 3\n\n\ndef test_sub():\n    assert sub(3, 1) == 2\n")
    assert rules(check(repo)) == [("test-disabled", "block", "tests/test_calc.py::test_add")]


def test_skip_call_in_body_and_module_pytestmark(repo):
    _write(repo, "tests/test_calc.py", "import pytest\nfrom pkg.calc import add, sub\n\n\ndef test_add():\n"
                                       "    pytest.skip('later')\n    assert add(1, 2) == 3\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    _write(repo, "tests/test_strings.py", "import pytest\nfrom pkg.strings import shout\n\npytestmark = pytest.mark.xfail\n\n\n"
                                          "def test_shout():\n    assert shout('a') == 'A'\n")
    found = [r[2] for r in rules(check(repo)) if r[0] == "test-disabled"]
    assert found == ["tests/test_calc.py::test_add", "tests/test_strings.py::test_shout"]


def test_editing_the_expected_value_to_match_a_bug_blocks(repo):
    _write(repo, "pkg/calc.py", BROKEN_ADD)
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 4\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    result = check(repo)
    assert rules(result) == [("test-changed-to-pass", "block", "tests/test_calc.py::test_add")]
    assert any("assert" in e for e in result.findings[0].evidence)


def test_removing_assertions_is_fix_or_justify(repo):
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    add(1, 2)\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    assert rules(check(repo)) == [("assertions-removed", "resolve", "tests/test_calc.py::test_add")]


def test_strengthening_or_refactoring_a_test_is_fine(repo):
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    total = add(1, 2)\n"
                                       "    assert total == 3\n    assert add(0, 0) == 0\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == 2\n")
    result = check(repo)
    assert result.findings == [] and "ran the session-start version of 1 edited test(s)" in result.checks[0]


def test_a_requested_behavior_change_blocks_until_handed_to_the_user(repo):
    """Changing a test's expectation is legitimate when the user asked for the
    new behavior; the agent then hands it over with needs-human, and the gate
    lets it stop."""
    from notyet import gate, snapshot, store
    root = str(repo)
    base = snapshot.head_tree(root)
    session = store.Session(session_id="req", started=0, baseline_tree=base, baseline_head=base,
                            baseline_source="session-start")
    _write(repo, "pkg/calc.py", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return b - a\n")
    _write(repo, "tests/test_calc.py", "from pkg.calc import add, sub\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
                                       "def test_sub():\n    assert sub(3, 1) == -2\n")
    _write(repo, ".notyet.toml", (repo / ".notyet.toml").read_text())
    first = gate.check(root, session)
    assert first.verdict == "blocked" and [f.rule for f in first.findings] == ["test-changed-to-pass"]
    fid = first.findings[0].id
    session.acks[fid] = {"reason": "user asked for sub to be reversed", "category": "needs-human"}
    assert gate.check(root, session).verdict == "passed"
